from pathlib import Path
import json
import zipfile

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from smolvlm_attention.plotting import (
    shared_scales, plot_token_atlas, plot_spatial_atlas,
    plot_layer_histogram, make_figures_zip,
)

CAMERAS = ['front', 'front_left', 'front_right', 'back', 'back_left', 'back_right']


def test_shared_scales_use_both_models_and_all_layers():
    a = np.zeros((30, 384))
    b = np.ones((30, 384))
    b[29, 383] = 2
    scales = shared_scales([a, b])
    assert scales['vmax'] == 2
    assert len(scales['histogram_edges']) == 101
    assert np.allclose(np.diff(scales['histogram_edges']), .02)
    assert scales['histogram_vmax'] == 384


@pytest.mark.parametrize('function,basename', [
    (plot_token_atlas, 'token_atlas'),
    (plot_spatial_atlas, 'spatial_atlas'),
    (plot_layer_histogram, 'layer_histogram'),
])
def test_figures_encode_correct_values_and_export(function, basename, tmp_path, monkeypatch):
    mean = np.arange(30 * 384, dtype=float).reshape(30, 384) / (30 * 384)
    scales = shared_scales([mean, mean * 2])
    saved = []
    original = matplotlib.figure.Figure.savefig
    def capture(fig, *args, **kwargs):
        if not saved:
            saved.append(fig)
        return original(fig, *args, **kwargs)
    monkeypatch.setattr(matplotlib.figure.Figure, 'savefig', capture)
    files = function(mean, CAMERAS, tmp_path, title='Test', scales=scales, dpi=15)
    assert {p.name for p in files} == {basename + '.png', basename + '.pdf'}
    assert all(p.stat().st_size > 100 for p in files)
    fig = saved[0]
    if basename == 'token_atlas':
        panels = [ax for ax in fig.axes if len(ax.patches) == 384]
        assert len(panels) == 15
        assert np.allclose([p.get_height() for p in panels[-1].patches], mean[28])
        assert all(ax.get_ylim() == (0, scales['vmax']) for ax in panels)
        assert np.array_equal(panels[0].get_xticks(), [0, 64, 128, 192, 256, 320, 383])
        assert panels[0].get_xlabel() == 'Image-token index'
        assert [t.get_text() for t in panels[0].child_axes[0].get_xticklabels()] == CAMERAS
    elif basename == 'spatial_atlas':
        panels = [ax for ax in fig.axes if ax.images]
        assert len(panels) == 90
        for camera, ax in enumerate(panels[:6]):
            im = ax.images[0]
            assert np.array_equal(im.get_array(), mean[0, camera*64:(camera+1)*64].reshape(8, 8))
            assert im.origin == 'upper'
            assert im.get_interpolation() == 'none'
            assert im.norm.vmin == 0 and im.norm.vmax == scales['vmax']
        assert len({round(ax.get_position().y0, 6) for ax in panels[:6]}) == 1
    else:
        ax = next(ax for ax in fig.axes if ax.images)
        im = ax.images[0]
        assert isinstance(im.norm, LogNorm)
        counts = np.ma.filled(im.get_array(), 0)
        assert counts.shape == (100, 30)
        assert np.all(counts.sum(axis=0) == 384)
        assert im.origin == 'lower'
        assert tuple(im.get_extent()) == (-.5, 29.5, 0, scales['vmax'])
        assert ax.get_xlabel() == 'Layer'
        assert ax.get_ylabel() == 'Mean image-token attention'
        assert np.array_equal(ax.get_xticks(), np.arange(30))
        assert np.ma.getmaskarray(im.get_array()).any()
    assert not plt.fignum_exists(fig.number)


def test_zip_requires_complete_models_and_only_archives_figures(tmp_path):
    root = tmp_path / 'figures'
    root.mkdir()
    with pytest.raises(ValueError):
        make_figures_zip(root, tmp_path / 'out.zip', {})
    for model in ['base', 'reverse']:
        folder = root / model
        folder.mkdir()
        for name in ['token_atlas', 'spatial_atlas', 'layer_histogram']:
            for ext in ['png', 'pdf']:
                (folder / f'{name}.{ext}').write_bytes(b'figure')
    (root / 'base' / 'cache.bin').write_bytes(b'cache')
    result = make_figures_zip(root, tmp_path / 'out.zip', {'seed': 7})
    with zipfile.ZipFile(result) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == 13
        assert json.loads(archive.read('metadata.json')) == {'seed': 7}
        assert 'base/token_atlas.png' in archive.namelist()
        assert not any('cache' in name for name in archive.namelist())
    (root / 'reverse' / 'token_atlas.pdf').unlink()
    with pytest.raises(ValueError):
        make_figures_zip(root, tmp_path / 'out.zip', {})
