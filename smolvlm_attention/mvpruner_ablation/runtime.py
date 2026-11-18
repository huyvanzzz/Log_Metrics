"""Kaggle measurement, resumable per-frame outputs, and dataset-level summaries."""

import csv
from hashlib import sha256
from io import BytesIO, StringIO
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .data import (CAMERAS, TEST_COUNT, atomic_output, file_sha256, load_manifest,
                   rows_by_frame, write_json)


LAYERS = 30
TOKENS_PER_CAMERA = 64
VISUAL_TOKENS = len(CAMERAS) * TOKENS_PER_CAMERA


def read_image(root, path, asset):
    payload = (Path(root) / path).read_bytes()
    if sha256(payload).hexdigest() != asset['sha256']:
        raise ValueError(f'Image changed since preparation: {path}')
    with Image.open(BytesIO(payload)) as image:
        return image.convert('RGB')


def frame_features(model, paths, assets, image_root):
    """Extract native pre-connector features in canonical camera order."""
    import torch

    images = [read_image(image_root, path, assets[path]) for path in paths]
    processor = model.processor.image_processor
    pixels = processor(images=[images], return_tensors='pt')
    vision = model.native.model.vision_model
    device = next(vision.parameters()).device
    values = pixels['pixel_values'][0].to(device)
    masks = pixels['pixel_attention_mask'][0].to(device)
    patch = model.native.config.vision_config.patch_size
    patches = masks.unfold(1, patch, patch).unfold(2, patch, patch)
    patch_mask = patches.any(dim=-1).any(dim=-1)
    with torch.inference_mode():
        return vision(pixel_values=values, patch_attention_mask=patch_mask).last_hidden_state


def _source_sha256():
    package = Path(__file__).resolve().parent
    roots = [package, package.parent, package.parents[1] / 'smolvlm_inference']
    value = sha256()
    for folder in roots:
        for file in sorted(folder.glob('*.py')):
            value.update(file.name.encode())
            value.update(bytes.fromhex(file_sha256(file)))
    return value.hexdigest()


def _bind_run(output, identity):
    path = output / 'measurement.json'
    if path.exists():
        if json.loads(path.read_bytes()) != identity:
            raise ValueError('Existing outputs belong to a different measurement; use a new output directory')
    else:
        if any((output / 'frames').glob('*.npz')):
            raise ValueError('Frame outputs are missing their measurement metadata')
        write_json(path, identity)


def _check_frame(path, rows, example_id):
    expected_heatmap = any(row['id'] == example_id for row in rows)
    with np.load(path, allow_pickle=False) as data:
        if not np.array_equal(data['ids'], [row['id'] for row in rows]):
            raise ValueError(f'Wrong question IDs in {path}')
        predicted = data['predicted']
        entropy = data['entropy']
        heatmap = data['heatmap']
    if (predicted.shape != (len(rows), LAYERS) or not np.issubdtype(predicted.dtype, np.integer)
            or (predicted < 0).any() or (predicted >= len(CAMERAS)).any()
            or entropy.shape != (len(rows), LAYERS, 2) or not np.isfinite(entropy).all()
            or (entropy < 0).any() or (entropy > 1).any()
            or heatmap.shape != ((LAYERS, VISUAL_TOKENS) if expected_heatmap else (0,))
            or not np.isfinite(heatmap).all() or (heatmap < 0).any()):
        raise ValueError(f'Invalid measurement arrays in {path}')
    return predicted, entropy, heatmap


def _measure_frame(model, encoder, features, rows, example_id, batch_size):
    import torch

    from .collector import collect

    encoded = [encoder.encode(row['question']) for row in rows]
    predicted, entropy, heatmap = [], [], np.empty(0, dtype=np.float32)
    start = 0
    while start < len(rows):
        stop = min(start + batch_size, len(rows))
        try:
            result = collect(model, encoder.batch(encoded[start:stop], features))
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
        else:
            shape = (stop - start, LAYERS, len(CAMERAS), TOKENS_PER_CAMERA)
            predicted.append(result['visual'].reshape(shape).mean(-1).argmax(-1).numpy())
            entropy.append(torch.stack((result['token_entropy'], result['question_entropy']), -1).numpy())
            for idx, row in enumerate(rows[start:stop]):
                if row['id'] == example_id:
                    heatmap = result['visual'][idx].numpy()
            start = stop
            continue
        torch.cuda.empty_cache()
        batch_size = max(1, batch_size // 2)
        print(f'CUDA memory limit reached. Retrying with batch size {batch_size}.', flush=True)
    return dict(ids=[row['id'] for row in rows], predicted=np.concatenate(predicted),
                entropy=np.concatenate(entropy), heatmap=heatmap), batch_size


def measure(manifest_path, image_root, output, *, checkpoint=None, batch_size=8):
    from filelock import FileLock
    from huggingface_hub import hf_hub_download
    import torch
    from tqdm.auto import tqdm
    import transformers

    from smolvlm_attention.config import MODELS
    from smolvlm_inference.model import CAMERA_ORDER, load_model
    from .collector import QuestionBatch, collect, reference

    if batch_size <= 0 or not torch.cuda.is_available():
        raise ValueError('Use a positive batch size and enable a Kaggle GPU')
    manifest = load_manifest(manifest_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = MODELS['base']
    with FileLock(str(output / 'measure.lock'), timeout=0):
        checkpoint_origin = ({'repo': spec['repo'], 'revision': spec['revision']}
                             if checkpoint is None else None)
        checkpoint = checkpoint or hf_hub_download(
            repo_id=spec['repo'], revision=spec['revision'], filename='finetune_best.pt', token=False)
        identity = dict(manifest_sha256=file_sha256(manifest_path),
                        checkpoint_sha256=file_sha256(checkpoint), source_sha256=_source_sha256(),
                        torch=torch.__version__, transformers=transformers.__version__,
                        gpu=torch.cuda.get_device_name(), precision='fp32',
                        questions=TEST_COUNT, layers=LAYERS, visual_tokens=VISUAL_TOKENS,
                        camera_order=list(CAMERAS), checkpoint_origin=checkpoint_origin)
        _bind_run(output, identity)
        groups = rows_by_frame(manifest)
        directory = output / 'frames'
        example_id = manifest['illustration']['id']
        pending = {}
        total_done = 0
        for frame_id, rows in groups.items():
            path = directory / f'{frame_id}.npz'
            if path.exists():
                _check_frame(path, rows, example_id)
                total_done += len(rows)
            else:
                pending[frame_id] = rows
        if not pending:
            print(f'All {total_done:,} questions are already measured.')
            return

        torch.set_float32_matmul_precision('highest')
        torch.backends.cudnn.allow_tf32 = False
        model = load_model(checkpoint, device='cuda:0', expected_prompt_camera_order=CAMERA_ORDER)
        if (len(model.native.model.text_model.layers) != LAYERS
                or model.geometry['visual_tokens_per_view'] != TOKENS_PER_CAMERA
                or tuple('CAM_' + camera.upper().replace('-', '_') for camera in CAMERA_ORDER) != CAMERAS):
            raise ValueError('Unexpected checkpoint layer count or camera token geometry')
        encoder = QuestionBatch(model)
        with tqdm(total=TEST_COUNT, initial=total_done, unit='question', desc='SmolVLM ablation') as bar:
            for index, (frame_id, rows) in enumerate(pending.items()):
                features = frame_features(model, manifest['frames'][frame_id], manifest['assets'], image_root)
                if index == 0:
                    encoded = encoder.encode(min(rows, key=lambda row: len(row['question']))['question'])
                    batch = encoder.batch([encoded], features)
                    actual, expected = collect(model, batch), reference(model, batch)
                    for key, value in actual.items():
                        torch.testing.assert_close(value, expected[key],
                                                   atol=1e-6 if key == 'visual' else 1e-5, rtol=1e-4)
                    del actual, expected, batch
                    bar.write('Native prefill attention parity passed.')
                arrays, batch_size = _measure_frame(model, encoder, features, rows, example_id, batch_size)
                with atomic_output(directory / f'{frame_id}.npz') as stream:
                    np.savez_compressed(stream, **arrays)
                bar.update(len(rows))
        print(f'Measurements saved to {output}')


def summarize(manifest_path, output):
    manifest = load_manifest(manifest_path)
    output = Path(output)
    metadata = json.loads((output / 'measurement.json').read_bytes())
    if metadata['manifest_sha256'] != file_sha256(manifest_path):
        raise ValueError('Measurement and plotting manifest differ')
    groups = rows_by_frame(manifest)
    example_id = manifest['illustration']['id']
    sources = ('tag', 'prose')
    counts = np.zeros((len(sources), len(CAMERAS)), dtype=np.int64)
    correct = np.zeros((len(sources), len(CAMERAS), LAYERS), dtype=np.int64)
    entropy_sum = np.zeros((LAYERS, 2), dtype=np.float64)
    heatmap, total = None, 0
    for frame_id, rows in groups.items():
        path = output / 'frames' / f'{frame_id}.npz'
        if not path.is_file():
            raise ValueError(f'Measurement incomplete: missing frame {frame_id}. Run measure to continue.')
        predicted, entropies, maps = _check_frame(path, rows, example_id)
        total += len(rows)
        entropy_sum += entropies.sum(axis=0, dtype=np.float64)
        if maps.size:
            heatmap = maps
        for row, prediction in zip(rows, predicted):
            if row['source'] in sources:
                source = sources.index(row['source'])
                label = CAMERAS.index(row['camera'])
                counts[source, label] += 1
                correct[source, label] += prediction == label
    if total != TEST_COUNT or heatmap is None or not counts.sum(axis=1).all():
        raise ValueError('Results must cover both labeled groups, the full test set, and the illustration')
    accuracy = correct.sum(axis=1) / counts.sum(axis=1, keepdims=True)
    baseline = counts.max(axis=1) / counts.sum(axis=1)
    entropy = entropy_sum / total

    table = StringIO(newline='')
    writer = csv.writer(table)
    camera_columns = [f'{source}_{camera.lower()}_accuracy' for source in sources for camera in CAMERAS]
    writer.writerow(['layer', 'tag_accuracy', 'prose_accuracy', 'tag_majority_baseline',
                     'prose_majority_baseline', 'token_level_entropy', 'question_level_entropy', *camera_columns])
    per_camera = np.divide(correct, counts[..., None], where=counts[..., None] > 0,
                           out=np.full(correct.shape, np.nan))
    for layer in range(LAYERS):
        writer.writerow([layer + 1, *accuracy[:, layer], *baseline, *entropy[layer],
                         *per_camera[:, :, layer].ravel()])
    with atomic_output(output / 'metrics_by_layer.csv') as stream:
        stream.write(table.getvalue().encode('utf-8'))
    write_json(output / 'summary.json', dict(questions=total, labeled_counts=counts.sum(axis=1).tolist(),
               sources=sources, camera_order=CAMERAS, camera_counts=counts.tolist(),
               majority_baselines=baseline.tolist(), illustration=manifest['illustration']))
    return manifest, accuracy, baseline, entropy, heatmap
