import json
import zipfile

import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt
import numpy as np
import pytest

from dinov2_attention.plotting import (shared_scales, plot_token_atlas,
    plot_spatial_atlas, plot_layer_histogram, make_figures_zip)

CAMERAS = ['Front', 'Front-Left', 'Front-Right', 'Back', 'Back-Left', 'Back-Right']


@pytest.mark.parametrize('value', [0., 1e-8])
def test_scales_accept_one_or_multiple_means_and_degenerate_values(value):
    mean = np.full((4, 384), value)
    a, b = shared_scales(mean), shared_scales([mean])
    assert 0 < a['log_vmin'] < a['vmax']
    assert np.array_equal(a['histogram_edges'], b['histogram_edges'])
    assert len(a['histogram_edges']) == 101
    assert a['histogram_vmax'] == 384
    if value:
        assert a['log_vmin'] <= value <= a['vmax']


@pytest.mark.parametrize('bad', [np.zeros((30,384)), np.full((4,384), -1),
                                 np.full((4,384), np.nan)])
def test_reject_invalid_means(bad):
    with pytest.raises(ValueError):
        shared_scales(bad)


@pytest.mark.parametrize('function,name', [(plot_token_atlas,'token_atlas'),
    (plot_spatial_atlas,'spatial_atlas'), (plot_layer_histogram,'layer_histogram')])
def test_figures_preserve_values_layout_scales_and_export(function, name, tmp_path, monkeypatch):
    mean = np.arange(4*384).reshape(4,384) / 1e7
    scales = shared_scales(mean)
    captured = []
    original = matplotlib.figure.Figure.savefig
    def capture(fig, *args, **kwargs):
        if not captured:
            captured.append(fig)
        return original(fig, *args, **kwargs)
    monkeypatch.setattr(matplotlib.figure.Figure, 'savefig', capture)
    files = function(mean, CAMERAS, tmp_path, scales=scales, dpi=25)
    assert {p.name for p in files} == {name+'.png', name+'.pdf'}
    assert all(p.stat().st_size > 100 for p in files)
    fig = captured[0]
    if name == 'token_atlas':
        axes = [ax for ax in fig.axes if ax.patches]
        assert len(axes) == 4
        for layer, ax in enumerate(axes):
            positive = mean[layer] > 0
            assert np.allclose([p.get_height() for p in ax.patches], mean[layer,positive])
            assert ax.get_yscale() == 'log'
            assert ax.get_ylim() == (scales['log_vmin'], scales['vmax'])
            assert [t.get_text() for t in ax.child_axes[0].get_xticklabels()] == CAMERAS
        assert len({round(ax.get_position().x0,4) for ax in axes}) == 2
        assert len({round(ax.get_position().y0,4) for ax in axes}) == 2
    elif name == 'spatial_atlas':
        axes = [ax for ax in fig.axes if ax.images]
        assert len(axes) == 24
        for index, ax in enumerate(axes):
            layer, camera = divmod(index,6)
            im = ax.images[0]
            assert np.array_equal(np.ma.filled(im.get_array(),0), mean[layer,camera*64:(camera+1)*64].reshape(8,8))
            assert im.origin == 'upper' and im.get_interpolation() == 'none'
            assert isinstance(im.norm, LogNorm)
            assert (im.norm.vmin,im.norm.vmax) == (scales['log_vmin'],scales['vmax'])
        assert len({round(ax.get_position().y0,5) for ax in axes[:6]}) == 1
        assert np.ma.getmaskarray(axes[0].images[0].get_array())[0,0]
    else:
        ax = next(ax for ax in fig.axes if ax.images)
        im = ax.images[0]
        counts = np.ma.filled(im.get_array(),0)
        assert counts.shape == (100,4)
        assert np.all(counts.sum(axis=0) == 384)
        assert isinstance(im.norm,LogNorm)
        assert tuple(im.get_extent()) == (-.5,3.5,0,scales['vmax'])
        assert np.array_equal(ax.get_xticks(),np.arange(4))
        assert np.array_equal(im.cmap.get_bad(),[1,1,1,1])
    assert not plt.fignum_exists(fig.number)


def test_zip_exact_figures_metadata_caption_only(tmp_path):
    with pytest.raises(ValueError):
        make_figures_zip(tmp_path,tmp_path/'out.zip',{})
    for name in ['token_atlas','spatial_atlas','layer_histogram']:
        for ext in ['png','pdf']:
            (tmp_path/f'{name}.{ext}').write_bytes(b'figure')
    (tmp_path/'checkpoint.pt').write_bytes(b'weights')
    result = make_figures_zip(tmp_path,tmp_path/'out.zip',{'query':'decoder-start'})
    with zipfile.ZipFile(result) as archive:
        assert len(archive.namelist()) == 8
        assert archive.testzip() is None
        assert json.loads(archive.read('metadata.json')) == {'query':'decoder-start'}
        assert b'encoder visual positions' in archive.read('caption.txt')
        assert 'checkpoint.pt' not in archive.namelist()
