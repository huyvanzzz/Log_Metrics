import json
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.figure
from matplotlib.colors import LogNorm
import numpy as np
import pytest

from smolvlm_attention.generated_plotting import (
    plot_attention_fractions, plot_generated_histogram, make_generated_zip,
)


def capture_figure(monkeypatch):
    figures = []
    original = matplotlib.figure.Figure.savefig
    def capture(fig, *args, **kwargs):
        if not figures:
            figures.append(fig)
        return original(fig, *args, **kwargs)
    monkeypatch.setattr(matplotlib.figure.Figure, 'savefig', capture)
    return figures


def test_fraction_curves_and_sd_band_preserve_probability_semantics(tmp_path, monkeypatch):
    figures = capture_figure(monkeypatch)
    mean = np.tile([.1, .7, .2], (30, 1))
    sd = np.full_like(mean, .15)
    files = plot_attention_fractions(mean, sd, tmp_path, dpi=20)
    assert {p.name for p in files} == {'attention_fractions.png', 'attention_fractions.pdf'}
    assert all(p.stat().st_size > 100 for p in files)
    fig, = figures
    assert np.allclose(fig.get_size_inches(), [9, 4.8])
    ax = fig.axes[0]
    assert ax.get_xlabel() == 'Layer'
    assert ax.get_ylabel() == 'Attention fraction'
    assert ax.get_ylim() == (0, 1)
    assert [line.get_color() for line in ax.lines] == ['#1f77b4', '#ff7f0e', '#2ca02c']
    for index, line in enumerate(ax.lines):
        assert np.array_equal(line.get_xdata(), np.arange(30))
        assert np.array_equal(line.get_ydata(), mean[:, index])
    assert len(ax.collections) == 3
    for band in ax.collections:
        assert band.get_alpha() == .2
        y = band.get_paths()[0].vertices[:, 1]
        assert y.min() >= 0 and y.max() <= 1


@pytest.mark.parametrize('bad', ['shape', 'nan', 'negative', 'sum', 'sd'])
def test_fraction_input_validation(bad, tmp_path):
    mean = np.tile([.1, .7, .2], (30, 1))
    sd = np.zeros_like(mean)
    if bad == 'shape': mean = mean[:29]
    if bad == 'nan': mean[0, 0] = np.nan
    if bad == 'negative': mean[0] = [-.1, .9, .2]
    if bad == 'sum': mean[0] = [.2, .7, .2]
    if bad == 'sd': sd[0, 0] = -.1
    with pytest.raises(ValueError):
        plot_attention_fractions(mean, sd, tmp_path)


@pytest.mark.parametrize('zero', [False, True])
def test_histogram_has_384_averaged_values_per_layer(tmp_path, monkeypatch, zero):
    figures = capture_figure(monkeypatch)
    mean = np.linspace(0, .002, 30 * 384).reshape(30, 384)
    if zero: mean[:] = 0
    paths = plot_generated_histogram(mean, tmp_path, dpi=20)
    assert all(p.stat().st_size > 100 for p in paths)
    fig, = figures
    assert np.allclose(fig.get_size_inches(), [9, 5.2])
    ax = fig.axes[0]
    image = ax.images[0]
    counts = np.ma.filled(image.get_array(), 0)
    assert counts.shape == (100, 30)
    assert np.all(counts.sum(axis=0) == 384)
    assert isinstance(image.norm, LogNorm)
    assert image.norm.vmin == 1
    assert image.norm.vmax <= 384
    assert image.get_cmap().name == 'magma'
    assert np.allclose(image.get_cmap().get_bad(), [1, 1, 1, 1])
    assert ax.get_ylabel() == 'Mean image-token attention'
    assert ax.get_xlabel() == 'Layer'


def test_zip_contains_only_figures_metadata_and_measurement_caption(tmp_path):
    for stem in ('attention_fractions', 'visual_attention_histogram'):
        for ext in ('png', 'pdf'):
            (tmp_path / f'{stem}.{ext}').write_bytes(b'figure')
    (tmp_path / 'checkpoint.pt').write_bytes(b'never include')
    path = make_generated_zip(tmp_path, tmp_path / 'figures.zip', {'seed': 42})
    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist()) == {
            'attention_fractions.png', 'attention_fractions.pdf',
            'visual_attention_histogram.png', 'visual_attention_histogram.pdf',
            'metadata.json', 'README.txt',
        }
        assert json.loads(archive.read('metadata.json')) == {'seed': 42}
        caption = archive.read('README.txt').decode()
        for phrase in ('384', 'heads', 'generated tokens', 'samples', 'SD', 'EOS'):
            assert phrase in caption
    (tmp_path / 'attention_fractions.pdf').unlink()
    with pytest.raises(FileNotFoundError):
        make_generated_zip(tmp_path, path, {})
