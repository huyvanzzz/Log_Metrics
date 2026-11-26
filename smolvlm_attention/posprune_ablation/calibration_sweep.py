"""Compare shallow-layer PosPrune bias fits from one CUDA prefill per batch."""

import argparse
from collections import defaultdict
from contextlib import ExitStack
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from tqdm.auto import tqdm

from smolvlm_attention.collector import _prepare
from smolvlm_attention.plotting import _STYLE, plt
from smolvlm_inference.model import CAMERA_ORDER, load_model
from smolvlm_inference.cache import record_key
from smolvlm_inference.text import CachedBatchCollator
from .decoder import _visual_attention
from .image_features import encode_frame, frame_plan, image_cache
from .runtime import _inputs, _scene, _write_json, select_rows


SHALLOW_LAYERS = (2, 3, 4, 5)
REFERENCE_LAYER = 22


@torch.inference_mode()
def measure_attention(model, inputs, layers):
    text, fused, mask, positions, last, visual = _prepare(model, inputs)
    observed = []

    def capture(attn, args, kwargs):
        observed.append(_visual_attention(attn, kwargs, mask, last, visual))

    with ExitStack() as hooks:
        for layer in layers:
            handle = text.layers[layer - 1].self_attn.register_forward_pre_hook(capture, with_kwargs=True)
            hooks.callback(handle.remove)
        text(inputs_embeds=fused, attention_mask=mask, position_ids=positions, use_cache=False,
             output_attentions=False, output_hidden_states=False)
    return torch.stack(observed, dim=1).cpu().numpy()


def split_by_recording(rows):
    recordings = defaultdict(list)
    for index, row in enumerate(rows):
        recordings[_scene(row) or record_key(row)].append(index)
    groups = [[], []]
    for name, indices in sorted(recordings.items(), key=lambda item: (-len(item[1]), item[0])):
        groups[min(range(2), key=lambda i: len(groups[i]))].extend(indices)
    if not all(groups):
        raise ValueError('The split check requires at least two distinct recording logs or frames')
    return [sorted(group) for group in groups]


def _fit(x, observations):
    return np.polynomial.Polynomial.fit(x, observations, deg=3)(x)


def _r_squared(observed, fitted):
    variance = np.square(observed - observed.mean()).sum()
    return float(1 - np.square(observed - fitted).sum() / variance) if variance else None


def _summarize(attention, groups, cameras):
    contrasts = attention[:, :-1].astype(np.float64) - attention[:, -1:, :]
    overall = contrasts.mean(axis=0)
    group_a, group_b = (contrasts[indices].mean(axis=0) for indices in groups)
    normalized = np.linspace(0, 1, overall.shape[-1])
    fits = np.stack([_fit(normalized, observed) for observed in overall])
    metrics = []
    for i, layer in enumerate(SHALLOW_LAYERS):
        observed, fitted, a, b = overall[i], fits[i], group_a[i], group_b[i]
        minimum = int(observed.argmin())
        camera, patch = divmod(minimum, len(observed) // len(cameras))
        correlation = float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else None
        metrics.append(dict(layer=layer, reference=REFERENCE_LAYER, cubic_r2=_r_squared(observed, fitted),
            cubic_rmse=float(np.sqrt(np.square(observed - fitted).mean())), split_pearson=correlation,
            cubic_r2_a_to_b=_r_squared(b, _fit(normalized, a)), cubic_r2_b_to_a=_r_squared(a, _fit(normalized, b)),
            minimum_index=minimum, minimum_camera=cameras[camera], camera_token_index=patch,
            minimum_difference=float(observed[minimum]), split_a_at_minimum=float(a[minimum]),
            split_b_at_minimum=float(b[minimum]),
            visual_mass=float(attention[:, i].sum(axis=1, dtype=np.float64).mean()),
            reference_visual_mass=float(attention[:, -1].sum(axis=1, dtype=np.float64).mean())))
    return overall, fits, group_a, group_b, metrics


def _plots(output, cameras, overall, fits, a, b, metrics, split_sizes):
    count = overall.shape[1]
    per_camera = count // len(cameras)
    x = np.arange(count)
    centers = (np.arange(len(cameras)) + .5) * per_camera - .5
    with plt.rc_context({**_STYLE, 'font.size': 10}):
        for name, curves, splits in (
            ('positional_bias_by_layer', (overall, fits), False),
            ('positional_bias_recording_split', (a, b), True),
        ):
            fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True, sharey=True, layout='constrained')
            labels = ((f'Recording group A ({split_sizes[0]} questions)', f'Recording group B ({split_sizes[1]} questions)')
                      if splits else ('Calibration mean', 'Cubic fit'))
            for i, ax in enumerate(axes.flat):
                for curve, label, color, width in zip(curves, labels, ('#35668c', '#bf601d'), (1.0, 1.6)):
                    ax.plot(x, curve[i], color=color, lw=width, label=label)
                for boundary in range(per_camera, count, per_camera):
                    ax.axvline(boundary - .5, color='0.82', lw=.7, zorder=0)
                ax.axhline(0, color='0.55', lw=.65, zorder=0)
                metric = metrics[i]
                statistic = metric['split_pearson'] if splits else metric['cubic_r2']
                label = 'Pearson r' if splits else 'cubic R²'
                value = f'{statistic:.3f}' if statistic is not None else 'n/a'
                ax.set_title(f'Layer {SHALLOW_LAYERS[i]} − layer {REFERENCE_LAYER}  ·  {label} {value}', loc='left')
                top = ax.secondary_xaxis('top')
                top.set_xticks(centers, cameras)
                top.tick_params(labelsize=8, length=0)
                top.spines['top'].set_visible(False)
            for ax in axes[-1]:
                ax.set_xlabel('Original visual token index (zero-based)')
            for ax in axes[:, 0]:
                ax.set_ylabel('Attention probability difference')
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='outside lower center', ncol=2)
            fig.suptitle('Positional Bias by Decoder Layer' if not splits else 'Positional Bias Across Disjoint Recordings',
                         fontsize=16)
            fig.savefig(output / f'{name}.png', dpi=180)
            plt.close(fig)


def _csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(checkpoint, prepared, images, output, *, calibration_size=800, seed=42, batch_size=8):
    if not torch.cuda.is_available():
        raise RuntimeError('This sweep requires a CUDA-capable PyTorch installation')
    if batch_size < 1:
        raise ValueError('Batch size must be positive')
    start = perf_counter()
    calibration, _, policy, _ = select_rows(prepared, calibration_size, 1, seed)
    groups = split_by_recording(calibration)
    frames = frame_plan(calibration)
    missing = [path for _, paths in frames for path in paths if not (images / path).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} selected camera images missing under {images}: {missing[0]}')
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda:0')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    model = load_model(checkpoint, device=device, expected_prompt_camera_order=CAMERA_ORDER)
    collator = CachedBatchCollator(model.processor, image_seq_len=model.geometry['visual_tokens_per_view'],
                                   max_length=model.native.config.text_config.max_position_embeddings,
                                   max_new_tokens=512, prompt_camera_order=model.prompt_camera_order)
    layers = (*SHALLOW_LAYERS, REFERENCE_LAYER)
    with TemporaryDirectory(prefix='frames-', dir=output) as scratch:
        for key, paths in tqdm(frames, desc='Encode local calibration frames', unit='frame'):
            with ExitStack() as files:
                views = []
                for path in paths:
                    with Image.open(images / path) as image:
                        views.append(files.enter_context(image.convert('RGB')))
                np.save(Path(scratch) / f'{key}.npy', encode_frame(model, views), allow_pickle=False)
        cache = image_cache(scratch, frames, {'calibration': len(calibration)})
        session = SimpleNamespace(model=model, device=device, cache=cache, collator=collator)
        batches = []
        index = 0
        with tqdm(total=len(calibration), desc='Collect five decoder layers', unit='question') as bar:
            while index < len(calibration):
                rows = calibration[index:index + batch_size]
                try:
                    batches.append(measure_attention(model, _inputs(session, rows), layers))
                except torch.cuda.OutOfMemoryError:
                    if len(rows) == 1:
                        raise
                    batch_size = len(rows) // 2
                else:
                    index += len(rows)
                    bar.update(len(rows))
                    continue
                torch.cuda.empty_cache()
    overall, fits, a, b, metrics = _summarize(np.concatenate(batches), groups, model.prompt_camera_order)
    _plots(output, model.prompt_camera_order, overall, fits, a, b, metrics, [len(group) for group in groups])
    _csv(output / 'layer_metrics.csv', metrics)
    _csv(output / 'attention_by_position.csv', [dict(layer=layer, index=j,
        camera=model.prompt_camera_order[j // model.geometry['visual_tokens_per_view']],
        mean=overall[i, j], cubic=fits[i, j], group_a=a[i, j], group_b=b[i, j])
        for i, layer in enumerate(SHALLOW_LAYERS) for j in range(overall.shape[1])])
    _write_json(output / 'calibration_config.json', dict(checkpoint=str(checkpoint.resolve()),
        prepared=str(prepared.resolve()), image_root=str(images.resolve()), gpu=torch.cuda.get_device_name(device),
        torch=torch.__version__, cuda=torch.version.cuda, seed=seed, precision='FP32 with FP16 feature storage',
        layers=list(layers), overlap_policy=policy, processing_seconds=perf_counter() - start,
        unique_frames=len(frames), sample_ids=[row['sample_id'] for row in calibration],
        recording_groups=[dict(sample_ids=[calibration[i]['sample_id'] for i in group],
                               logs=sorted({_scene(calibration[i]) for i in group if _scene(calibration[i])}))
                          for group in groups]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'prepared', 'images', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    run(args.checkpoint, args.prepared, args.images, args.output, batch_size=args.batch_size)
