import importlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.error
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LogNorm, to_rgba
import numpy as np
from PIL import Image
import pytest


def plotting():
    return importlib.import_module('smolvlm_probing.plotting')


def capture(monkeypatch):
    figures = []
    original = matplotlib.figure.Figure.savefig
    def save(fig, path, **kwargs):
        if fig not in figures:
            figures.append(fig)
        assert kwargs['dpi'] == 300
        return original(fig, path, **{**kwargs, 'dpi': 18})
    monkeypatch.setattr(matplotlib.figure.Figure, 'savefig', save)
    return figures


def summary():
    return {p: {'total_targets': 10, 'eligible_targets': 4, 'coverage': .4,
                'reasons': {'eligible': 4, 'missing_phrase': 6},
                'groups': [{'layers': [g*5, g*5+4], 'correct': g%3,
                            'total': 4, 'accuracy': (g%3)/4} for g in range(6)],
                'unreachable_total': 2, 'unreachable_eligible': 1}
            for p in ('with_coordinates', 'without_coordinates')}


def records():
    results=[]
    for i in range(5):
        maps=np.zeros((6,384)); maps[:, 0]=.02
        maps[:, 192 if i==1 else 0]=.04
        results.append({'sample_id': str(i), 'frame_id': str(i//2),
                        'partition': 'with_coordinates', 'question': 'What?',
                        'response': 'White sedan.',
                        'camera_paths': {'Front': 'data/nuscenes/samples/CAM_FRONT/a.jpg',
                                         'Back': 'data/nuscenes/samples/CAM_BACK/b.jpg'},
                        'targets': [{'status': 'eligible',
                                     'target': {'camera': 'Front', 'bbox': [0,0,40,30],
                                                'image_size':[80,60], 'category':'Vehicle',
                                                'size_fraction': .01 if i==2 else .3,
                                                'reference_span': {'text':'White sedan'}},
                                     'token_indices': [0,1], 'matched_span': [0,11],
                                     'group_results': [{'layer_start': g*5, 'layer_end':g*5+4,
                                                       'peak_index': 192 if i==1 else 0,
                                                       'peak_camera': 'Back' if i==1 else 'Front',
                                                       'peak_center':[5,3.75], 'correct':i!=1,
                                                       'bbox_contains_grid_center':True}
                                                      for g in range(6)],
                                     'attention_maps': maps.tolist()}]})
    return results


def test_synthetic_fixture_maps_agree_with_saved_global_peaks():
    for record in records():
        for evaluated in record['targets']:
            for attention, decision in zip(evaluated['attention_maps'], evaluated['group_results']):
                assert int(np.argmax(attention)) == decision['peak_index']
                assert ('Front', 'Front-Left', 'Front-Right', 'Back', 'Back-Left', 'Back-Right')[decision['peak_index'] // 64] == decision['peak_camera']


def test_causal_five_exports_shared_axes_and_unclipped_values(tmp_path, monkeypatch):
    figures=capture(monkeypatch)
    records_=[{'sample_id':'a','random':{'comp':-.3,'sgap': .4},
               'layers':[{'layer':l,'top':{'comp': -9 if l==25 else .7,'sgap':.8},
                          'bottom':{'comp':.01,'sgap':4 if l==19 else .02}}
                         for l in (1,7,13,19,25)]}]
    paths=plotting().plot_causal(records_,tmp_path)
    assert len(paths)==10 and all(p.is_file() for p in paths)
    assert len(figures)==5
    axes=[f.axes[0] for f in figures]
    assert len({a.get_xlim() for a in axes})==1
    assert len({a.get_ylim() for a in axes})==1
    assert axes[0].get_xlim()[0]<-9 and axes[0].get_ylim()[1]>4
    for ax in axes:
        assert ax.get_xscale()==ax.get_yscale()=='linear'
        assert [tuple(c.get_facecolors()[0][:3]) for c in ax.collections] == [to_rgba(c)[:3] for c in ('#1f77b4','#ff7f0e','#2ca02c')]
        assert all(0<c.get_alpha()<1 for c in ax.collections)
        assert np.allclose(ax.collections[1].get_offsets(), [[-.3,.4]])
    assert 'embeddings' in (tmp_path/'causal_caption.txt').read_text()


def test_localization_six_groups_and_zero_denominator_na(tmp_path,monkeypatch):
    figures=capture(monkeypatch); sums=summary()
    sums['without_coordinates'].update(eligible_targets=0,coverage=0)
    for g in sums['without_coordinates']['groups']:
        g.update(correct=0,total=0,accuracy=None)
    paths=plotting().plot_localization(sums,tmp_path)
    assert len(paths)==2
    ax=figures[0].axes[0]
    assert len(ax.patches)==12
    assert sum(t.get_text()=='N/A' for t in ax.texts)==6
    assert any('50.0%'==t.get_text() for t in ax.texts)
    caption=(tmp_path/'localization_caption.txt').read_text()
    assert '4/10' in caption and '0/10' in caption and 'grid' in caption


def test_selection_reproducible_and_diverse_and_empty_safe():
    p=plotting(); sums=summary()
    a=p.select_examples(records(),sums); b=p.select_examples(list(reversed(records())),sums)
    assert [r['sample_id'] for r in a['with_coordinates']]==[r['sample_id'] for r in b['with_coordinates']]
    assert len(a['with_coordinates'])==3 and a['without_coordinates']==[]
    assert len({r['frame_id'] for r in a['with_coordinates']})==3
    assert {r['selection_reason'] for r in a['with_coordinates']}=={'hit','miss','small'}
    assert all(r['group_index']==2 and r['layer_start']==10 for r in a['with_coordinates'])
    assert p.select_examples([],sums)=={'with_coordinates':[],'without_coordinates':[]}


def test_fetch_selected_only_deduplicates_and_uses_cache(tmp_path):
    p=plotting(); examples=p.select_examples(records(),summary()); calls=[]
    def download(url,path):
        calls.append(url); Image.new('RGB',(80,60)).save(path,format='JPEG')
    mapping,missing=p.fetch_example_images(examples,tmp_path,downloader=download)
    assert not missing and len(mapping)==2 and len(calls)==2
    assert all('/6c4336a5279eaba3f3dfc5b525cc6ba57d55d61a/nuscenes/' in u for u in calls)
    assert all('data/nuscenes' not in u for u in calls)
    mapping2,missing=p.fetch_example_images(examples,tmp_path,downloader=lambda *_:pytest.fail('cache missed'))
    assert mapping2==mapping and not missing
    assert (tmp_path/'image_fetch_receipt.json').is_file()


def test_fetch_retry_429_then_404_is_recorded(tmp_path):
    p=plotting(); examples=p.select_examples(records()[:1],summary()); calls=[]; waits=[]
    def download(url,path):
        calls.append(url)
        raise urllib.error.HTTPError(url,429,'limited',{'Retry-After':'2'},None)
    mapping,missing=p.fetch_example_images(examples,tmp_path,downloader=download,sleep=waits.append)
    assert not mapping and len(calls)==3 and waits==[2.,2.]
    assert missing[0]['attempts']==3
    calls.clear()
    def absent(url,path):
        calls.append(url); raise urllib.error.HTTPError(url,404,'missing',{},None)
    mapping,missing=p.fetch_example_images(examples,tmp_path,downloader=absent,sleep=waits.append)
    assert not mapping and len(calls)==1 and '404' in missing[0]['error']


@pytest.mark.parametrize('bad',['../a.jpg','https://evil/a.jpg','data/nuscenes/../a.jpg','other/a.jpg'])
def test_fetch_rejects_paths_outside_selected_dataset(tmp_path,bad):
    p=plotting(); examples=p.select_examples(records()[:1],summary())
    examples['with_coordinates'][0]['camera_paths']['Front']=bad
    mapping,missing=p.fetch_example_images(examples,tmp_path,downloader=lambda *_:pytest.fail('unexpected request'))
    assert not mapping and len(missing)==1


def test_overlays_share_lognorm_render_raw_grid_and_wrong_camera(tmp_path,monkeypatch):
    p=plotting(); figures=capture(monkeypatch)
    examples=p.select_examples(records(),summary())
    source=tmp_path/'source.jpg'; Image.new('RGB',(80,60),'gray').save(source)
    paths=p.plot_examples(examples, {v:source for r in examples['with_coordinates'] for v in r['camera_paths'].values()},tmp_path)
    assert len(paths)==2
    heatmaps=[im for ax in figures[0].axes for im in ax.images if isinstance(im.norm,LogNorm)]
    assert len(heatmaps)==4
    assert len({id(im.norm) for im in heatmaps})==1
    assert all(im.get_array().shape==(8,8) for im in heatmaps)
    assert all(np.ma.is_masked(im.get_array()) for im in heatmaps)
    assert any('Global peak' in ax.get_title() for ax in figures[0].axes)
    assert any('10–14' in ax.get_title() for ax in figures[0].axes)
    figures[0].set_dpi(300)
    canvas = FigureCanvasAgg(figures[0])
    canvas.draw()
    renderer = canvas.get_renderer()
    for row_index in range(2):
        current = figures[0].axes[row_index*3+1]
        following = figures[0].axes[(row_index+1)*3+1]
        phrase = current.texts[0].get_window_extent(renderer)
        next_title = following.title.get_window_extent(renderer)
        assert phrase.y0 > next_title.y1 + 80
    assert p.plot_examples({'with_coordinates':[]}, {},tmp_path)==[]


def test_bundle_explicit_whitelist_only(tmp_path):
    p=plotting(); figure=tmp_path/'causal_layer_1.png'; figure.write_bytes(b'png')
    caption=tmp_path/'causal_caption.txt';caption.write_text('caption')
    summary_=tmp_path/'summary.json';summary_.write_text('{}')
    bad=tmp_path/'checkpoint.pt';bad.write_bytes(b'weights')
    source=tmp_path/'source_image.png';source.write_bytes(b'pixels')
    archive=p.bundle_figures(tmp_path,[figure],{'seed':42},[caption,summary_])
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist())=={'causal_layer_1.png','causal_caption.txt','summary.json','metadata.json'}
        assert json.loads(z.read('metadata.json'))=={'seed':42}
    for item in (bad,source):
        with pytest.raises(ValueError):p.bundle_figures(tmp_path,[figure],{},[item])


def test_plotting_import_does_not_import_torch():
    script="import sys; import smolvlm_probing.plotting; assert 'torch' not in sys.modules"
    result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_normalized_aliases_share_one_failed_request_sequence(tmp_path):
    p=plotting(); examples=p.select_examples(records()[:1],summary())
    second=dict(examples['with_coordinates'][0])
    second['camera_paths']={'Front':'nuscenes/samples/CAM_FRONT/a.jpg'}
    examples['with_coordinates'].append(second)
    calls=[]
    def absent(url,path):
        calls.append(url); raise urllib.error.HTTPError(url,404,'missing',{},None)
    mapping,missing=p.fetch_example_images(examples,tmp_path,downloader=absent)
    assert mapping=={} and len(missing)==2
    assert len(calls)==1


def test_local_original_image_precedes_network(tmp_path,monkeypatch):
    p=plotting(); monkeypatch.chdir(tmp_path)
    original=tmp_path/'data/nuscenes/samples/CAM_FRONT/a.jpg'
    original.parent.mkdir(parents=True); Image.new('RGB',(80,60)).save(original)
    examples=p.select_examples(records()[:1],summary())
    mapping,missing=p.fetch_example_images(examples,tmp_path/'images',downloader=lambda *_:pytest.fail('unexpected network'))
    assert not missing and Path(mapping['data/nuscenes/samples/CAM_FRONT/a.jpg']).is_file()


def test_zero_attention_and_missing_images_render_without_fake_targets(tmp_path,monkeypatch):
    p=plotting();capture(monkeypatch); examples=p.select_examples(records()[:1],summary())
    target=examples['with_coordinates'][0]['targets'][0]
    target['attention_maps']=np.zeros((6,384)).tolist()
    assert len(p.plot_examples(examples,{},tmp_path))==2


def test_causal_refuses_nonfinite_metrics(tmp_path):
    p=plotting()
    data=[{'sample_id':'bad','random':{'comp':float('nan'),'sgap':0},
           'layers':[{'layer':l,'top':{'comp':0,'sgap':0},'bottom':{'comp':0,'sgap':0}}
                     for l in (1,7,13,19,25)]}]
    with pytest.raises(ValueError,match='finite'):
        p.plot_causal(data,tmp_path)
