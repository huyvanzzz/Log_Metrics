import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from dinov2_attention.data import (validate_cache, open_validated_cache, prepare_test,
                                   make_batch, record_key, tokenizer_policy)

CAMERAS = ['Front', 'Front-Left', 'Front-Right', 'Back', 'Back-Left', 'Back-Right']

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fixture_cache(tmp_path):
    cache, prepared = tmp_path / 'cache', tmp_path / 'prepared'
    cache.mkdir(); prepared.mkdir()
    row = dict(sample_id='test-a', eval_id=0, question='Where?', answer='SECRET',
               camera_paths={c: f'{c}.jpg' for c in CAMERAS})
    (prepared / 'manifest.json').write_text('{}')
    for split in ('train', 'val', 'test'):
        (prepared / f'{split}.jsonl').write_text(json.dumps(dict(row, sample_id=f'{split}-a')) + '\n')
    (cache / 'index.json').write_text(json.dumps({record_key(row): 0}))
    values = np.arange(6 * 4 * 8, dtype=np.float16).reshape(1,6,4,8)
    values.tofile(cache / 'features.float16.memmap')
    manifest = dict(version=2,dataset_revision='fixture',model_id='facebook/dinov2-small',
                    model_revision='revision',preprocessing='dinov2_224',dtype='float16',
                    image_size=224,view_order=CAMERAS,shape=list(values.shape),
                    processor_policy={'fixture': True},feature_extraction={'fixture': True},
                    load_dtype='float32',data_identity={'prepared_sha256':{
                        n: digest(prepared/n) for n in ('manifest.json','train.jsonl','val.jsonl','test.jsonl')},
                        'index_sha256':digest(cache/'index.json')})
    manifest['fingerprint']=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (cache/'manifest.json').write_text(json.dumps(manifest))
    return cache,prepared,values

class Tokenizer:
    pad_token_id=0
    eos_token_id=1
    padding_side='right'
    special_tokens_map={'pad_token':'<pad>', 'eos_token':'</s>'}
    def get_vocab(self): return {'<pad>':0,'</s>':1,'?':2}
    def __call__(self, text, **kwargs):
        assert text == 'Question: Where? Answer:'
        assert kwargs == {'add_special_tokens':True,'truncation':False}
        return {'input_ids':[4,5,1]}

def test_receipt_prep_and_batch(tmp_path):
    cache, prepared, values = fixture_cache(tmp_path)
    receipt=validate_cache(cache,prepared,tmp_path/'out',expected_fingerprint=None)
    reader=open_validated_cache(receipt)
    assert reader.counts == {'train':1,'val':1,'test':1}
    assert not reader.array.flags.writeable
    path=prepare_test(reader,Tokenizer(),tmp_path/'out',progress=False)
    data=json.loads(path.read_text())
    assert data['records'][0]['input_ids']==[4,5,1]
    assert 'SECRET' not in path.read_text()
    original=path.read_bytes()
    assert prepare_test(reader,Tokenizer(),tmp_path/'out',progress=True).read_bytes()==original
    batch=make_batch(reader,[data['records'][0],dict(data['records'][0],input_ids=[1])],Tokenizer())
    assert batch['input_ids'].tolist()==[[4,5,1],[1,0,0]]
    assert batch['attention_mask'].tolist()==[[1,1,1],[1,0,0]]
    np.testing.assert_array_equal(batch['visual_features'][0],values[0])
    assert str(batch['visual_features'].dtype)=='torch.float32'

@pytest.mark.parametrize('change',['prepared','payload','manifest','index'])
def test_receipt_rejects_changed_input(tmp_path,change):
    cache,prepared,_=fixture_cache(tmp_path)
    receipt=validate_cache(cache,prepared,tmp_path/'out',expected_fingerprint=None)
    target={'prepared':prepared/'test.jsonl','payload':cache/'features.float16.memmap',
            'manifest':cache/'manifest.json','index':cache/'index.json'}[change]
    with target.open('ab') as f: f.write(b' ')
    with pytest.raises(ValueError): open_validated_cache(receipt)

@pytest.mark.parametrize('change',['fingerprint','nan','missing','duplicate'])
def test_invalid_cache(tmp_path,change):
    cache,prepared,_=fixture_cache(tmp_path)
    if change=='fingerprint':
        m=json.loads((cache/'manifest.json').read_text());m['fingerprint']='wrong';(cache/'manifest.json').write_text(json.dumps(m))
    elif change=='nan':
        v=np.memmap(cache/'features.float16.memmap',dtype=np.float16,mode='r+');v[0]=np.nan;v.flush();del v
    elif change=='missing':
        (cache/'index.json').write_text('{}')
    else:
        p=prepared/'test.jsonl';p.write_text(p.read_text()*2)
    with pytest.raises(ValueError): validate_cache(cache,prepared,tmp_path/'out',expected_fingerprint=None)

def test_key_and_tokenizer_policy():
    assert record_key({'camera_paths':{'z':'raw/z','a':'raw/a'}})=='raw/a|raw/z'
    assert tokenizer_policy(Tokenizer())==tokenizer_policy(Tokenizer())

def refresh_manifest(cache, prepared):
    path=cache/'manifest.json'
    manifest=json.loads(path.read_text())
    manifest['data_identity']['prepared_sha256']={n:digest(prepared/n) for n in ('manifest.json','train.jsonl','val.jsonl','test.jsonl')}
    manifest['data_identity']['index_sha256']=digest(cache/'index.json')
    manifest.pop('fingerprint')
    manifest['fingerprint']=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    path.write_text(json.dumps(manifest))

@pytest.mark.parametrize('change',['duplicate','missing_camera','missing_key','nondense'])
def test_record_coverage_and_index_semantics(tmp_path,change):
    cache,prepared,_=fixture_cache(tmp_path)
    p=prepared/'test.jsonl'
    row=json.loads(p.read_text())
    if change=='duplicate': p.write_text(p.read_text()*2)
    elif change=='missing_camera':
        row['camera_paths'].pop('Front');p.write_text(json.dumps(row)+'\n')
    elif change=='missing_key':
        row['camera_paths']['Front']='different.jpg';p.write_text(json.dumps(row)+'\n')
    else: (cache/'index.json').write_text(json.dumps({record_key(row):1}))
    refresh_manifest(cache,prepared)
    with pytest.raises(ValueError): validate_cache(cache,prepared,tmp_path/'out',expected_fingerprint=None)

def test_receipt_reuse_skips_finite_scan(tmp_path,monkeypatch):
    cache,prepared,_=fixture_cache(tmp_path)
    receipt=validate_cache(cache,prepared,tmp_path/'out',expected_fingerprint=None,progress=False)
    def fail(*args,**kwargs): raise AssertionError('finite scan repeated')
    monkeypatch.setattr(np,'isfinite',fail)
    assert validate_cache(cache,prepared,tmp_path/'out',expected_fingerprint=None)==receipt
    assert open_validated_cache(receipt).counts['test']==1

def test_preparation_length_stats_and_missing_question(tmp_path):
    cache,prepared,_=fixture_cache(tmp_path)
    reader=open_validated_cache(validate_cache(cache,prepared,tmp_path/'out',expected_fingerprint=None,progress=False))
    value=json.loads(prepare_test(reader,Tokenizer(),tmp_path/'out',progress=False).read_text())
    assert value['token_lengths']=={'min':3,'max':3}
    row=json.loads((prepared/'test.jsonl').read_text());row.pop('question')
    (prepared/'test.jsonl').write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='question'):
        prepare_test(reader,Tokenizer(),tmp_path/'out',progress=False)


def test_digest_progress_bytes(tmp_path,monkeypatch):
    from dinov2_attention import data
    payload=tmp_path/'payload.bin';payload.write_bytes(b'hello world')
    updates=[]
    class Bar:
        def __init__(self,**kwargs):
            assert kwargs['total']==11
            assert kwargs['unit']=='B'
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def update(self,count): updates.append(count)
    monkeypatch.setattr(data,'tqdm',Bar)
    assert data.digest(payload,progress=True)==hashlib.sha256(b'hello world').hexdigest()
    assert sum(updates)==11
