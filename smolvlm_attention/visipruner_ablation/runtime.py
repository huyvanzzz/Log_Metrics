"""Prepare the paired test subset and persist Kaggle generations by variant and frame."""

from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
import logging
from pathlib import Path

from smolvlm_attention.mvpruner_ablation.data import (
    CAMERAS, TEST_COUNT, _relative_image, atomic_output, file_sha256, write_json,
)

from .design import MERGE_SEEDS, RANK_LAYERS, SAMPLE_COUNT, SEED, merge_positions, select_indices, yes_no
from .suites import SUITES, suite_variants


# Source digest of runtime.py in the released single-GPU runner.
_SINGLE_GPU_RUNTIME = 'dd6f4634e1a5857ee2e3b75095af984e88ff56cc1e5aa935ee53f413c9212ea3'
_PRE_SWEEP_RUNTIME = '7de511b8237294b4ac565df5f02b5c29172f1e7e67ba26d66f6556b86126beb7'
_PRE_SWEEP_DECODER = '659ab8491304258e694aa085d48a0ddfc1acb4b9cea16948c757b0b000e97247'


def prepare(test_file, image_root, output, count=None, *, suite='original'):
    from tqdm.auto import tqdm

    payload = Path(test_file).read_bytes()
    records = json.loads(payload)
    if len(records) != TEST_COUNT:
        raise ValueError(f'Expected {TEST_COUNT} questions in the input split')
    if suite == 'deep-full':
        if count not in (None, TEST_COUNT):
            raise ValueError('The deep-full suite requires the complete test split')
        count, indices = TEST_COUNT, range(TEST_COUNT)
    else:
        count = SAMPLE_COUNT if count is None else count
        indices = select_indices(records, count)
    frames = {}
    for index in indices:
        qa, images = records[index]
        paths = [_relative_image(images[camera], camera) for camera in CAMERAS]
        frame_id = sha256('\n'.join(paths).encode()).hexdigest()
        frame = frames.setdefault(frame_id, dict(paths=paths, rows=[]))
        frame['rows'].append(dict(id=index, question=qa['Q'], answer=qa['A'],
                                  question_type=qa['question_type'], yes_no=yes_no(qa['A'])))
    all_paths = sorted({path for frame in frames.values() for path in frame['paths']})
    assets = {path: dict(sha256=file_sha256(Path(image_root) / path))
              for path in tqdm(all_paths, desc='Hash test images', unit='image')}
    positions = merge_positions() if suite == 'original' else {}
    variants = [dict(asdict(v), layers=list(v.layers)) for v in SUITES[suite]]
    manifest = dict(version=1, seed=SEED, samples=count, test_sha256=sha256(payload).hexdigest(),
                    frames=frames, assets=assets, variants=variants,
                    merge={str(seed): positions[seed] for seed in MERGE_SEEDS if seed in positions})
    if suite != 'original':
        manifest['suite'] = suite
    destination = output / 'manifest.json'
    if destination.exists():
        if json.loads(destination.read_bytes()) != manifest:
            raise ValueError('An output manifest already exists for different inputs; choose a new output directory')
    else:
        write_json(destination, manifest)
    types = Counter(row['question_type'] for frame in frames.values() for row in frame['rows'])
    print(f'{count:,} questions · {len(frames)} six-camera frames · {dict(types)}')
    for seed, entry in positions.items():
        print(f'Merge seed {seed}: {CAMERAS[entry["camera"]]}, patch row {entry["row"] + 1}, col {entry["col"] + 1}')


def _logger(output, variant):
    folder = output / 'logs'
    folder.mkdir(parents=True, exist_ok=True)
    logger = logging.Logger(variant)
    formatter = logging.Formatter('%(asctime)s %(levelname)s [%(name)s] %(message)s')
    for handler in (logging.StreamHandler(), logging.FileHandler(folder / f'{variant}.log', encoding='utf-8')):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _load_frame(path, rows):
    data = json.loads(path.read_bytes())
    if [entry['id'] for entry in data] != [row['id'] for row in rows]:
        raise ValueError(f'Invalid saved prediction IDs in {path}')
    return data


def _code_identity(runtime_digest=None):
    package = Path(__file__).resolve().parent
    code = [package / name for name in ('design.py', 'decoder.py', 'runtime.py')]
    code += [package.parent / 'mvpruner_ablation' / name for name in ('collector.py', 'runtime.py')]
    code += [package.parents[1] / 'smolvlm_inference' / name for name in ('model.py', 'text.py')]
    if runtime_digest is None:
        code.append(package / 'suites.py')
    overrides = ({package / 'runtime.py': runtime_digest, package / 'decoder.py': _PRE_SWEEP_DECODER}
                 if runtime_digest is not None else {})
    digests = [overrides.get(path) or file_sha256(path) for path in code]
    return sha256(''.join(digests).encode()).hexdigest()


def _bind(path, identity):
    if path.exists():
        existing = json.loads(path.read_bytes())
        # Both earlier runners used the same generation and intervention math.
        legacy_codes = (_code_identity(_SINGLE_GPU_RUNTIME), _code_identity(_PRE_SWEEP_RUNTIME))
        if existing != identity and not any(existing == dict(identity, code=code) for code in legacy_codes):
            raise ValueError('Checkpoint, data, decoding settings or measurement code changed; use a new output directory')
    else:
        write_json(path, identity)


def _projected(model, output, frame_id, frame, manifest, image_root):
    from filelock import FileLock
    import numpy as np
    import torch

    from smolvlm_attention.mvpruner_ablation.runtime import frame_features

    path = output / 'projected' / f'{frame_id}.npy'
    if path.exists():
        return torch.from_numpy(np.load(path, allow_pickle=False)).to(next(model.parameters()).device)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path.with_suffix('.lock'))):
        if path.exists():
            return torch.from_numpy(np.load(path, allow_pickle=False)).to(next(model.parameters()).device)
        with torch.inference_mode():
            raw = frame_features(model, frame['paths'], manifest['assets'], image_root)
            projected = model._project(raw.unsqueeze(0), 1)
        with atomic_output(path) as stream:
            np.save(stream, projected.cpu().numpy(), allow_pickle=False)
    return projected


def _measure_frame(decoder, encoder, rows, projected, variant, limit, rank_file, recipient, batch_size, logger,
                   rank_layers):
    import numpy as np
    import torch

    encoded = [encoder.encode(row['question']) for row in rows]
    rank_layers = tuple(layer for layer in rank_layers if layer <= len(decoder.text.layers))
    rank_parts = {layer: [] for layer in rank_layers}
    ranking = {}
    if variant.mode in ('top10', 'keep10', 'bottom10', 'keep5'):
        with np.load(rank_file, allow_pickle=False) as saved:
            if not np.array_equal(saved['ids'], [row['id'] for row in rows]):
                raise ValueError(f'Mismatched baseline rankings in {rank_file}')
            ranking = {layer: torch.from_numpy(saved[str(layer)]).to(decoder.device, dtype=torch.long)
                       for layer in variant.layers}
    results, start = [], 0
    while start < len(rows):
        stop = min(len(rows), start + batch_size)
        try:
            tokens, ranks, mass, truncated = decoder.generate(
                encoded[start:stop], projected, variant, limit,
                rankings={layer: order[start:stop] for layer, order in ranking.items()} or None,
                recipient=recipient, rank_layers=rank_layers,
            )
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
        else:
            for layer, order in ranks.items():
                rank_parts[layer].append(order.numpy().astype(np.uint16))
            predictions = encoder.tokenizer.batch_decode(tokens, skip_special_tokens=True)
            for row, prediction, stats, cutoff in zip(rows[start:stop], predictions, mass.tolist(), truncated.tolist()):
                results.append(dict(id=row['id'], prediction=prediction.strip(), truncated=cutoff,
                                    removed_mass=stats[0], visual_mass=stats[1],
                                    fallback_rows=int(stats[2]), attention_rows=int(stats[3])))
            start = stop
            continue
        torch.cuda.empty_cache()
        batch_size = max(1, batch_size // 2)
        logger.warning('CUDA OOM. Retrying the same questions with batch size %s', batch_size)
    if variant.mode == 'baseline' and rank_layers:
        with atomic_output(rank_file) as stream:
            np.savez_compressed(stream, ids=[row['id'] for row in rows],
                                **{str(layer): np.concatenate(parts) for layer, parts in rank_parts.items()})
    return results, batch_size


def run(output, image_root, variant, *, batch_size, limit, checkpoint, gpu=0):
    from filelock import FileLock
    from huggingface_hub import hf_hub_download
    import torch
    import transformers
    from tqdm.auto import tqdm

    from smolvlm_attention.config import MODELS
    from smolvlm_attention.mvpruner_ablation.collector import QuestionBatch
    from smolvlm_inference.model import CAMERA_ORDER, load_model
    from .decoder import Decoder

    if transformers.__version__ != '4.46.3' or not torch.cuda.is_available():
        raise RuntimeError('Use a Kaggle GPU with Transformers 4.46.3')
    if batch_size < 1 or limit < 1:
        raise ValueError('Batch size and generation limit must be positive')
    if not 0 <= gpu < torch.cuda.device_count():
        raise ValueError(f'GPU index {gpu} is unavailable')
    device = torch.device(f'cuda:{gpu}')
    torch.cuda.set_device(device)
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    variants = suite_variants(manifest)
    if variant not in variants:
        raise ValueError(f'{variant.name} is not part of the suite in {manifest_path}')
    rank_layers = RANK_LAYERS if manifest.get('suite', 'original') == 'original' else ()
    logger = _logger(output, variant.name)
    spec = MODELS['base']
    lock_folder = output / 'locks'
    lock_folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_folder / f'{variant.name}.lock'), timeout=0):
        if variant.mode != 'baseline' and not (output / 'predictions/baseline/complete.json').is_file():
            raise ValueError('Finish the baseline before starting the intervention cells')
        if checkpoint is None:
            checkpoint = hf_hub_download(repo_id=spec['repo'], revision=spec['revision'],
                                         filename='finetune_best.pt')
        identity = dict(manifest=file_sha256(manifest_path), checkpoint=file_sha256(checkpoint),
                        code=_code_identity(), torch=str(torch.__version__), gpu=torch.cuda.get_device_name(device),
                        generation='greedy', max_new_tokens=limit)
        with FileLock(str(output / 'run.lock')):
            _bind(output / 'run.json', identity)
        directory = output / 'predictions' / variant.name
        frames = manifest['frames']
        pending, done = [], 0
        for frame_id, frame in frames.items():
            path = directory / f'{frame_id}.json'
            if path.is_file():
                done += len(_load_frame(path, frame['rows']))
            else:
                pending.append((frame_id, frame))
        total = manifest['samples']
        number = variants.index(variant) + 1
        logger.info('Experiment %s/%s | %s | %s/%s questions already measured',
                    number, len(variants), device, done, total)
        if pending:
            torch.set_float32_matmul_precision('highest')
            torch.backends.cudnn.allow_tf32 = False
            model = load_model(checkpoint, device=device, expected_prompt_camera_order=CAMERA_ORDER)
            if len(model.native.model.text_model.layers) != 30 or model.visual_token_len != 384:
                raise ValueError('The experiment expects 30 decoder layers and 384 image tokens')
            encoder = QuestionBatch(model)
            decoder = Decoder(model, encoder)
            recipient = manifest['merge'][str(variant.seed)]['token'] if variant.seed is not None else None
            with tqdm(total=total, initial=done, desc=variant.name, unit='question') as bar:
                for index, (frame_id, frame) in enumerate(pending):
                    projected = _projected(model, output, frame_id, frame, manifest, image_root)
                    if index == 0 and variant.mode == 'baseline':
                        probes = [encoder.encode(row['question']) for row in frame['rows'][:2]]
                        decoder.verify_native(probes, projected, variant)
                        logger.info('Baseline greedy-token parity with native SmolVLM passed')
                    results, batch_size = _measure_frame(
                        decoder, encoder, frame['rows'], projected, variant, limit,
                        output / 'rankings' / f'{frame_id}.npz', recipient, batch_size, logger, rank_layers,
                    )
                    write_json(directory / f'{frame_id}.json', results)
                    done += len(results)
                    bar.update(len(results))
                    logger.info('%s/%s questions | frame %s', done, total, frame_id[:12])
        if done != total:
            raise RuntimeError(f'Incomplete run: {done}/{total} questions')
        write_json(directory / 'complete.json', dict(samples=done, variant=variant.name))
        logger.info('Complete | %s/%s questions', done, total)
