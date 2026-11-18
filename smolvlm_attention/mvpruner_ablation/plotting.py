"""Export four standalone Figure 2-style figures."""

from pathlib import Path
from textwrap import fill

from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import PercentFormatter
import numpy as np

from smolvlm_attention.plotting import _STYLE, _export, plt
from .data import CAMERAS, MAP_LAYERS
from .runtime import LAYERS, read_image, summarize


_BLUE = '#185782'
_ORANGE = '#a24d17'


def _layer_axis(ax):
    ax.set(xlabel='Decoder layer', xlim=(0.25, LAYERS + 0.75),
           xticks=[1, *range(5, LAYERS + 1, 5)])
    ax.grid(axis='y', color='#dadfe3', linewidth=0.7)
    ax.set_axisbelow(True)


def _accuracy(output, accuracy, baseline, counts):
    layers = np.arange(1, LAYERS + 1)
    fig, ax = plt.subplots(figsize=(8.7, 4.4), layout='constrained')
    ax.bar(layers, accuracy[0], color=_BLUE, width=0.78, label=f'Object tag (n={counts["tag"]:,})')
    ax.plot(layers, accuracy[1], color=_ORANGE, linewidth=1.8, marker='o',
            markersize=3, label=f'Direction phrase (n={counts["prose"]:,})')
    for color, value, style, group in zip((_BLUE, _ORANGE), baseline, ('--', ':'), ('Tag', 'Phrase')):
        ax.axhline(value, color=color, linestyle=style, linewidth=1.25,
                   label=f'{group} majority baseline ({value:.1%})')
    _layer_axis(ax)
    ax.set(ylabel='Top-1 camera accuracy', ylim=(0, 1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncols=2, fontsize=9)
    return _export(fig, output, 'view_recognition_accuracy', 300)


def _entropy(output, values, name, ylabel, count):
    fig, ax = plt.subplots(figsize=(8.7, 4.2), layout='constrained')
    layers = np.arange(1, LAYERS + 1)
    ax.plot(layers, values, color=_BLUE, linewidth=2, marker='o', markersize=3)
    _layer_axis(ax)
    ax.set(ylabel=ylabel, ylim=(0, 1))
    ax.set_title(f'DriveLM test · {count:,} questions', loc='left', fontsize=11)
    return _export(fig, output, name, 300)


def _maps(output, manifest, image_root, heatmap):
    example = manifest['illustration']
    row = manifest['rows'][example['id']]
    # Saved attention covers every layer, including those omitted by older manifests.
    layers = sorted(set(manifest['map_layers']).union(MAP_LAYERS))
    maps = heatmap[np.asarray(layers) - 1].reshape(len(layers), len(CAMERAS), 8, 8)
    positive = maps[maps > 0]
    if positive.size and positive.min() < positive.max():
        norm = LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
    else:
        norm = Normalize(vmin=0, vmax=float(maps.max()) or 1)
    images = [read_image(image_root, path, manifest['assets'][path])
              for path in manifest['frames'][row['frame']]]
    fig, axes = plt.subplots(len(layers), len(CAMERAS), figsize=(14.5, 2 * len(layers)),
                             layout='constrained')
    cmap = plt.get_cmap('magma')
    for layer_idx, (layer, axes_row) in enumerate(zip(layers, axes)):
        for camera_idx, (camera, image, ax) in enumerate(zip(CAMERAS, images, axes_row)):
            width, height = image.size
            extent = (0, width, height, 0)
            ax.imshow(image, extent=extent)
            ax.imshow(np.ma.masked_less_equal(maps[layer_idx, camera_idx], 0),
                      extent=extent, origin='upper', cmap=cmap, norm=norm,
                      interpolation='nearest', alpha=0.6)
            if camera == example['camera']:
                ax.scatter(example['x'], example['y'], s=60, facecolors='none',
                           edgecolors='#46e3ff', linewidths=1.7)
            if layer_idx == 0:
                ax.set_title(camera.removeprefix('CAM_').replace('_', ' ').title(), fontsize=11)
            if camera_idx == 0:
                ax.set_ylabel(f'Layer {layer}', fontsize=11)
            ax.set(xticks=[], yticks=[])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle(fill(row['question'], width=120), fontsize=13)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist(),
                 fraction=0.023, pad=0.02, label='Mean attention probability')
    return _export(fig, output, 'text_to_vision_attention_maps', 300)


def plot(manifest_path, image_root, output):
    manifest, accuracy, baseline, entropy, heatmap = summarize(manifest_path, output)
    figures = Path(output) / 'figures'
    with plt.rc_context(_STYLE):
        paths = _accuracy(figures, accuracy, baseline, manifest['counts'])
        paths += _entropy(figures, entropy[:, 0], 'token_level_attention_entropy',
                          'Normalized token-level entropy', len(manifest['rows']))
        paths += _entropy(figures, entropy[:, 1], 'question_level_attention_entropy',
                          'Normalized question-level entropy', len(manifest['rows']))
        paths += _maps(figures, manifest, image_root, heatmap)
    return paths
