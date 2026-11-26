"""Publication figures for sample/head-averaged image attention."""
from pathlib import Path
import json
import zipfile

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
import numpy as np

_STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12, 'axes.linewidth': 2, 'axes.spines.top': False,
    'axes.spines.right': False, 'legend.frameon': False,
    'text.usetex': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
}
_LAYERS = tuple(range(0, 30, 2))
_NAMES = ('token_atlas', 'spatial_atlas', 'layer_histogram')


def _layer_ids(layer_ids):
    if layer_ids is None:
        return None
    result = tuple(layer_ids)
    if (not result or len(result) > 15 or len(set(result)) != len(result)
            or any(not isinstance(x, (int, np.integer)) or isinstance(x, bool)
                   or not 0 <= x < 30 for x in result)):
        raise ValueError('Expected up to fifteen distinct layer indices in 0..29')
    return result


def _mean(value, layer_ids=None):
    result = np.asarray(value, dtype=float)
    layers = _layer_ids(layer_ids)
    expected = (30 if layers is None else len(layers), 384)
    if result.shape != expected:
        raise ValueError(f'Expected attention means with shape {expected}')
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError('Attention means must be finite and nonnegative')
    return result


def _cameras(camera_order):
    names = list(camera_order)
    if len(names) != 6 or len(set(names)) != 6:
        raise ValueError('Expected six distinct camera names in prompt order')
    return names


def shared_scales(means: list[np.ndarray], *, layer_ids=None) -> dict:
    """Use all supplied rows; explicit layer_ids label an already selected array."""
    if not means:
        raise ValueError('At least one model is required')
    arrays = [_mean(mean, layer_ids) for mean in means]
    maximum = max(float(a.max()) for a in arrays)
    # An all-zero model still needs nondegenerate axes and linear bin edges.
    maximum = maximum if maximum > 0 else 1.0
    positive_minima = [float(a[a > 0].min()) for a in arrays if np.any(a > 0)]
    minimum = min(positive_minima) if positive_minima else maximum * 1e-6
    log_minimum = max(np.finfo(float).tiny, 10. ** np.floor(np.log10(minimum)))
    if log_minimum >= maximum:
        log_minimum = maximum / 10
    edges = np.linspace(0, maximum, 101)
    count_maximum = max(int(np.histogram(row, bins=edges)[0].max())
                        for array in arrays for row in array)
    return {'vmax': maximum, 'log_vmin': log_minimum, 'attention_scale': 'log',
            'zero_display': 'no bar; gray heatmap cell', 'histogram_edges': edges,
            'histogram_vmax': count_maximum}


def _export(fig, out_dir, name, dpi):
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    files = [folder / f'{name}.{extension}' for extension in ('png', 'pdf')]
    try:
        for path in files:
            fig.savefig(path, dpi=dpi, facecolor='white')
    finally:
        plt.close(fig)
    return files


def plot_token_atlas(mean, camera_order, out_dir, *, title, scales, dpi=300,
                     layer_ids=None, value_label=None) -> list[Path]:
    """Plot 384 bars per panel; explicit layer_ids map directly to supplied rows."""
    mean, names = _mean(mean, layer_ids), _cameras(camera_order)
    layers = _LAYERS if layer_ids is None else _layer_ids(layer_ids)
    row_indices = layers if layer_ids is None else range(len(layers))
    norm = LogNorm(scales['log_vmin'], scales['vmax'])
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(5, 3, figsize=(24, 20), layout='constrained')
        for ax, layer, row_index in zip(axes.flat, layers, row_indices):
            values = mean[row_index]
            ax.bar(np.arange(384), values, width=1, linewidth=0,
                   color=plt.colormaps['viridis'](norm(values)))
            for boundary in range(64, 384, 64):
                ax.axvline(boundary - .5, color='0.5', linewidth=.7)
            ax.set_yscale('log')
            ax.set(xlim=(-.5, 383.5), ylim=(scales['log_vmin'], scales['vmax']),
                   title=f'Layer {layer}', ylabel=(value_label + ' (log)') if value_label else 'Mean attention (log)')
            ax.set_xticks([0, 64, 128, 192, 256, 320, 383])
            ax.set_xlabel('Image-token index')
            camera_axis = ax.secondary_xaxis('top')
            camera_axis.set_xticks(np.arange(6) * 64 + 31.5, names,
                                   rotation=25, ha='left', fontsize=9)
            camera_axis.tick_params(length=0, pad=3)
            camera_axis.spines['top'].set_visible(False)
            ax.set_title(f'Layer {layer}', pad=43)
        fig.suptitle(title, fontsize=22)
        fig.colorbar(ScalarMappable(norm=norm, cmap='viridis'), ax=axes,
                     label=(value_label + ' (log; zero has no bar)') if value_label else 'Mean attention (log; zero has no bar)', shrink=.65, pad=.02)
        return _export(fig, out_dir, 'token_atlas', dpi)


def plot_spatial_atlas(mean, camera_order, out_dir, *, title, scales, dpi=300,
                     layer_ids=None, value_label=None) -> list[Path]:
    """Preserve camera/token order: six untransformed 8x8 maps per panel row."""
    mean, names = _mean(mean, layer_ids), _cameras(camera_order)
    layers = _LAYERS if layer_ids is None else _layer_ids(layer_ids)
    row_indices = layers if layer_ids is None else range(len(layers))
    norm = LogNorm(scales['log_vmin'], scales['vmax'])
    cmap = plt.colormaps['viridis'].with_extremes(bad='lightgray')
    with plt.rc_context(_STYLE):
        fig = plt.figure(figsize=(24, 12), layout='constrained')
        grid = fig.add_gridspec(5, 4, width_ratios=[1, 1, 1, .045],
                                wspace=.03, hspace=.04)
        axes = []
        for index, (layer, row_index) in enumerate(zip(layers, row_indices)):
            panel = fig.add_subfigure(grid[index // 3, index % 3])
            panel.suptitle(f'Layer {layer}', fontsize=15)
            row = panel.subplots(1, 6, gridspec_kw={'wspace': .08})
            for camera, (ax, name) in enumerate(zip(row, names)):
                ax.imshow(np.ma.masked_equal(mean[row_index, camera * 64:(camera + 1) * 64].reshape(8, 8), 0),
                          cmap=cmap, norm=norm, origin='upper',
                          interpolation='none', aspect='equal')
                ax.set_title(name, fontsize=10, rotation=30, pad=8)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                axes.append(ax)
        fig.suptitle(title, fontsize=22)
        # A separate outer colorbar avoids changing any individual camera width.
        colorbar_ax = fig.add_subplot(grid[:, 3])
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_ax,
                     label=(value_label + ' (log; zero = gray)') if value_label else 'Mean attention (log; zero = gray)')
        return _export(fig, out_dir, 'spatial_atlas', dpi)


def plot_layer_histogram(mean, camera_order, out_dir, *, title, scales, dpi=300,
                     layer_ids=None, value_label=None) -> list[Path]:
    """Count 384 tokens per supplied layer (all 30 layers by default)."""
    mean = _mean(mean, layer_ids)
    layers = tuple(range(30)) if layer_ids is None else _layer_ids(layer_ids)
    _cameras(camera_order)
    edges = np.asarray(scales['histogram_edges'])
    counts = np.stack([np.histogram(row, bins=edges)[0] for row in mean])
    if not np.all(counts.sum(axis=1) == 384):
        raise ValueError('Histogram edges must contain every attention value')
    norm = LogNorm(vmin=1, vmax=max(2, scales['histogram_vmax']))
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(14, 10), layout='constrained')
        im = ax.imshow(np.ma.masked_equal(counts.T, 0), origin='lower',
                       aspect='auto', interpolation='none', cmap='magma', norm=norm,
                       extent=(-.5, len(layers) - .5, edges[0], edges[-1]))
        ax.set(title=title, xlabel='Layer', ylabel=value_label or 'Mean image-token attention')
        ax.set_xticks(np.arange(len(layers)), [str(layer) for layer in layers])
        fig.colorbar(im, ax=ax, label='Token count (log scale)')
        return _export(fig, out_dir, 'layer_histogram', dpi)


def make_figures_zip(figures_dir, output_zip, metadata: dict, *,
                     models=('base', 'reverse'), caption=None) -> Path:
    """Archive only selected models' figures, metadata, and an optional caption."""
    root, destination = Path(figures_dir), Path(output_zip)
    models = tuple(models)
    if not models or len(set(models)) != len(models) or any(model not in ('base', 'reverse') for model in models):
        raise ValueError('Expected distinct base/reverse model names')
    expected = [Path(model) / f'{name}.{extension}'
                for model in models for name in _NAMES
                for extension in ('png', 'pdf')]
    missing = [str(path) for path in expected
               if not (root / path).is_file() or (root / path).stat().st_size == 0]
    if missing:
        raise ValueError('Missing or empty required figures: ' + ', '.join(missing))
    payload = json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in expected:
            archive.write(root / path, path.as_posix())
        archive.writestr('metadata.json', payload + '\n')
        if caption is not None:
            archive.writestr('caption.txt', str(caption).rstrip() + '\n')
    with zipfile.ZipFile(destination) as archive:
        damaged = archive.testzip()
        if damaged is not None:
            raise ValueError(f'Archive integrity check failed: {damaged}')
    return destination
