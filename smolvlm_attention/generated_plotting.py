"""Paper-style figures for attention averaged over generated answer tokens."""
from pathlib import Path
import json
import zipfile

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

_STYLE = {
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif', 'font.size': 14,
    'axes.labelsize': 17, 'axes.linewidth': 1,
    'text.usetex': False, 'pdf.fonttype': 42,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
}
_NAMES = ('attention_fractions', 'visual_attention_histogram')
_CAPTION = """Generated-token attention, SmolVLM base

Query positions are generated answer tokens; EOS and padding are excluded.
Generated keys include the current query token (causal self-attention).
Attention is softmax-normalized over all valid causal context keys.
Prompt text includes every prompt position except the 384 image tokens.

Average attention over heads, then generated tokens within each sample,
then samples with equal sample weights. Histogram: each layer contains
384 image-token values after these averages, NOT pooled sample values.
Histogram color is bin count on a logarithmic scale; empty bins are white.
The three curves sum to one; shaded bands show mean +/- 1 SD across samples
after averaging heads and generated tokens, clipped to the interval [0, 1].
See metadata.json for sample counts, seed, revisions and generation settings.
"""


def _array(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f'{name} must have shape {shape}')
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f'{name} must be finite and nonnegative')
    return array


def _export(fig, out_dir, name, dpi):
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / f'{name}.{extension}' for extension in ('png', 'pdf')]
    try:
        for path in paths:
            fig.savefig(path, dpi=dpi, facecolor='white')
    finally:
        plt.close(fig)
    return paths


def plot_attention_fractions(mean_groups, std_groups, out_dir, dpi=300) -> list[Path]:
    """Plot three group means and sample SD after head/token averaging."""
    mean = _array(mean_groups, (30, 3), 'Group means')
    std = _array(std_groups, (30, 3), 'Group standard deviations')
    if not np.allclose(mean.sum(axis=1), 1, rtol=1e-5, atol=1e-6):
        raise ValueError('The three attention group means must sum to one')
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
        layers = np.arange(30)
        labels = (r'$a_{\mathrm{img}}$ (Image)',
                  r'$a_{\mathrm{txt}}$ (Prompt text)',
                  r'$a_{\mathrm{gen}}$ (Generated)')
        for index, (color, label) in enumerate(zip(
                ('#1f77b4', '#ff7f0e', '#2ca02c'), labels)):
            ax.fill_between(layers, np.clip(mean[:, index] - std[:, index], 0, 1),
                            np.clip(mean[:, index] + std[:, index], 0, 1),
                            color=color, alpha=.2, linewidth=0)
            ax.plot(layers, mean[:, index], color=color, label=label, linewidth=1.5)
        ax.set(xlabel='Layer', ylabel='Attention fraction', xlim=(0, 29), ylim=(0, 1))
        ax.set_xticks([0, 5, 10, 15, 20, 25, 29])
        ax.set_axisbelow(True)
        ax.grid(color='.85', linewidth=.6)
        ax.legend(loc='best', framealpha=.85, fontsize=12,
                  title=r'Shading: $\pm$1 SD', title_fontsize=11)
        return _export(fig, out_dir, 'attention_fractions', dpi)


def plot_generated_histogram(mean_visual, out_dir, dpi=300) -> list[Path]:
    """Histogram 384 already-averaged visual positions for each of 30 layers."""
    mean = _array(mean_visual, (30, 384), 'Visual attention means')
    maximum = float(mean.max()) or 1.0
    edges = np.linspace(0, maximum, 101)
    counts = np.stack([np.histogram(row, bins=edges)[0] for row in mean], axis=1)
    cmap = plt.colormaps['magma'].with_extremes(bad='white')
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(9, 5.2), layout='constrained')
        image = ax.imshow(np.ma.masked_equal(counts, 0), cmap=cmap,
                          norm=LogNorm(vmin=1, vmax=max(2, int(counts.max()))),
                          aspect='auto', origin='lower', interpolation='none',
                          extent=(-.5, 29.5, 0, maximum))
        ax.set(xlabel='Layer', ylabel='Mean image-token attention',
               xlim=(-.5, 29.5), ylim=(0, maximum))
        ax.set_xticks([0, 5, 10, 15, 20, 25, 29])
        colorbar = fig.colorbar(image, ax=ax, pad=.025, fraction=.045)
        colorbar.set_label('Image-token count (log scale)', fontsize=12)
        return _export(fig, out_dir, 'visual_attention_histogram', dpi)


def make_generated_zip(figures_dir, output_zip, metadata) -> Path:
    """Package exactly four figures, provenance, and measurement captions."""
    folder = Path(figures_dir)
    paths = [folder / f'{name}.{extension}' for name in _NAMES
             for extension in ('png', 'pdf')]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Missing generated-attention figure: {path}')
    encoded = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)
    output = Path(output_zip)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, arcname=path.name)
        archive.writestr('metadata.json', encoded)
        archive.writestr('README.txt', _CAPTION)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise OSError('Generated-attention ZIP failed CRC verification')
    return output
