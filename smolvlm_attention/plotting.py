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


def _mean(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (30, 384):
        raise ValueError('Expected attention means with shape (30, 384)')
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError('Attention means must be finite and nonnegative')
    return result


def _cameras(camera_order):
    names = list(camera_order)
    if len(names) != 6 or len(set(names)) != 6:
        raise ValueError('Expected six distinct camera names in prompt order')
    return names


def shared_scales(means: list[np.ndarray]) -> dict:
    """Build common raw-value and histogram-count scales across every layer."""
    if not means:
        raise ValueError('At least one model is required')
    arrays = [_mean(mean) for mean in means]
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


def plot_token_atlas(mean, camera_order, out_dir, *, title, scales, dpi=300) -> list[Path]:
    """Plot 384 raw-valued bars in each of fifteen even-layer panels."""
    mean, names = _mean(mean), _cameras(camera_order)
    norm = LogNorm(scales['log_vmin'], scales['vmax'])
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(5, 3, figsize=(24, 20), layout='constrained')
        for ax, layer in zip(axes.flat, _LAYERS):
            values = mean[layer]
            ax.bar(np.arange(384), values, width=1, linewidth=0,
                   color=plt.colormaps['viridis'](norm(values)))
            for boundary in range(64, 384, 64):
                ax.axvline(boundary - .5, color='0.5', linewidth=.7)
            ax.set_yscale('log')
            ax.set(xlim=(-.5, 383.5), ylim=(scales['log_vmin'], scales['vmax']),
                   title=f'Layer {layer}', ylabel='Mean attention (log)')
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
                     label='Mean attention (log; zero has no bar)', shrink=.65, pad=.02)
        return _export(fig, out_dir, 'token_atlas', dpi)


def plot_spatial_atlas(mean, camera_order, out_dir, *, title, scales, dpi=300) -> list[Path]:
    """Preserve camera/token order: six untransformed 8x8 maps per panel row."""
    mean, names = _mean(mean), _cameras(camera_order)
    norm = LogNorm(scales['log_vmin'], scales['vmax'])
    cmap = plt.colormaps['viridis'].with_extremes(bad='lightgray')
    with plt.rc_context(_STYLE):
        fig = plt.figure(figsize=(24, 12), layout='constrained')
        grid = fig.add_gridspec(5, 4, width_ratios=[1, 1, 1, .045],
                                wspace=.03, hspace=.04)
        axes = []
        for index, layer in enumerate(_LAYERS):
            panel = fig.add_subfigure(grid[index // 3, index % 3])
            panel.suptitle(f'Layer {layer}', fontsize=15)
            row = panel.subplots(1, 6, gridspec_kw={'wspace': .08})
            for camera, (ax, name) in enumerate(zip(row, names)):
                ax.imshow(np.ma.masked_equal(mean[layer, camera * 64:(camera + 1) * 64].reshape(8, 8), 0),
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
                     label='Mean attention (log; zero = gray)')
        return _export(fig, out_dir, 'spatial_atlas', dpi)


def plot_layer_histogram(mean, camera_order, out_dir, *, title, scales, dpi=300) -> list[Path]:
    """Count the 384 averaged tokens in shared linear bins for all 30 layers."""
    mean = _mean(mean)
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
                       extent=(-.5, 29.5, edges[0], edges[-1]))
        ax.set(title=title, xlabel='Layer', ylabel='Mean image-token attention')
        ax.set_xticks(np.arange(30))
        fig.colorbar(im, ax=ax, label='Token count (log scale)')
        return _export(fig, out_dir, 'layer_histogram', dpi)


def make_figures_zip(figures_dir, output_zip, metadata: dict) -> Path:
    """Archive exactly both models' six figure files plus reproducibility metadata."""
    root, destination = Path(figures_dir), Path(output_zip)
    expected = [Path(model) / f'{name}.{extension}'
                for model in ('base', 'reverse') for name in _NAMES
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
    with zipfile.ZipFile(destination) as archive:
        damaged = archive.testzip()
        if damaged is not None:
            raise ValueError(f'Archive integrity check failed: {damaged}')
    return destination
