import numpy as np
import pytest
from matplotlib import pyplot as plt
from smolvlm_attention import plotting as p

LAYERS = tuple(range(0,29,2))
CAMERAS = ['Front','Front-Left','Front-Right','Back','Back-Left','Back-Right']
LABEL = 'Mean combined attention score'

@pytest.mark.parametrize('function', [p.plot_token_atlas,p.plot_spatial_atlas,p.plot_layer_histogram])
def test_selected_layers_preserve_values_and_layout(function,tmp_path,monkeypatch):
    values=np.arange(15*384,dtype=float).reshape(15,384)/100000
    scales=p.shared_scales([values],layer_ids=LAYERS)
    figures=[]
    def capture(fig,*args):
        figures.append(fig)
        plt.close(fig)
        return []
    monkeypatch.setattr(p,'_export',capture)
    function(values,CAMERAS,tmp_path,title='Combined',scales=scales,layer_ids=LAYERS,value_label=LABEL)
    fig=figures[0]
    if function is p.plot_token_atlas:
        panels=[a for a in fig.axes if len(a.patches)==384]
        assert len(panels)==15
        assert [a.get_title() for a in panels]==[f'Layer {x}' for x in LAYERS]
        np.testing.assert_array_equal([x.get_height() for x in panels[-1].patches],values[-1])
        assert all(LABEL in a.get_ylabel() and a.get_yscale()=='log' for a in panels)
    elif function is p.plot_spatial_atlas:
        panels=[a for a in fig.axes if a.images]
        assert len(panels)==90
        for i,a in enumerate(panels):
            np.testing.assert_array_equal(np.ma.filled(a.images[0].get_array(),0),values[i//6,(i%6)*64:(i%6+1)*64].reshape(8,8))
            assert a.images[0].origin=='upper'
        assert [s._suptitle.get_text() for s in fig.subfigs]==[f'Layer {x}' for x in LAYERS]
    else:
        ax=next(a for a in fig.axes if a.images)
        counts=np.ma.filled(ax.images[0].get_array(),0)
        assert counts.shape==(100,15)
        assert np.all(counts.sum(0)==384)
        assert [t.get_text() for t in ax.get_xticklabels()]==[str(x) for x in LAYERS]
        assert ax.get_ylabel()==LABEL


def test_selected_scales_and_validation():
    values=np.ones((15,384))*.1
    assert p.shared_scales([values],layer_ids=LAYERS)['vmax']==.1
    with pytest.raises(ValueError):p.shared_scales([values])
    for layers in [(0,0),(),(-1,), (30,)]:
        with pytest.raises(ValueError):p.shared_scales([np.zeros((len(layers),384))],layer_ids=layers)


def test_base_only_zip_has_caption_and_six_figures(tmp_path):
    import zipfile
    root=tmp_path/'figures'
    (root/'base').mkdir(parents=True)
    for name in ['token_atlas','spatial_atlas','layer_histogram']:
        for ext in ['png','pdf']:(root/'base'/f'{name}.{ext}').write_bytes(b'figure')
    (root/'base'/'cache.bin').write_bytes(b'not a figure')
    result=p.make_figures_zip(root,tmp_path/'out.zip',{},models=('base',),caption='Formula caption')
    with zipfile.ZipFile(result) as archive:
        assert len(archive.namelist())==8
        assert archive.read('caption.txt').decode()=='Formula caption\n'
        assert not any('cache' in name for name in archive.namelist())
