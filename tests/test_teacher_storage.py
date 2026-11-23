import json
import shutil
import numpy as np
import pytest
from smolvlm_attention.teacher_targets.storage import TargetShard, finalize_split, validate_export


@pytest.fixture(autouse=True)
def current_storage_api():
    # Source-activation regression evicts package imports. Fault injection must
    # patch the same module instance used by the class under test.
    from smolvlm_attention.teacher_targets import storage
    for name in ('TargetShard', 'finalize_split', 'validate_export'):
        globals()[name] = getattr(storage, name)


def probs(n, v=4):
    x = np.arange(n*v, dtype=np.float32).reshape(n,v) / 10
    return (x - np.log(np.exp(x).sum(1,keepdims=True))).astype('<f4')


def test_commit_resume_and_finalize(tmp_path):
    identity = {'score_layer_index':22, 'split':'train'}
    a = TargetShard(tmp_path/'a', identity, [2,0], 4)
    b = TargetShard(tmp_path/'b', identity, [1], 4)
    a.commit([0,2], probs(2))
    b.commit([1], probs(1))
    assert TargetShard(tmp_path/'a', identity, [2,0], 4).pending_rows.size == 0
    samples = ({'row':i,'sample_id':str(i),'split':'train'} for i in range(3))
    finalize_split(tmp_path/'export',identity,[a,b],samples,3,4)
    result = np.load(tmp_path/'export/log_probs.npy')
    np.testing.assert_array_equal(result[[0,2]],probs(2))
    shutil.move(str(tmp_path/'export'), str(tmp_path/'moved'))
    assert validate_export(tmp_path/'moved',identity)['complete']


def test_uncommitted_recalculated_and_corruption_rejected(tmp_path):
    s = TargetShard(tmp_path, {'x':1}, [0,1],4)
    s.array[0] = probs(1)[0]
    s.array.flush()
    assert TargetShard(tmp_path, {'x':1}, [0,1],4).pending_rows.tolist() == [0,1]
    s.commit([0],probs(1))
    s.array[0,0] += 1
    s.array.flush()
    with pytest.raises(ValueError):
        TargetShard(tmp_path, {'x':1}, [0,1],4)
    with pytest.raises(ValueError):
        TargetShard(tmp_path, {'x':2}, [0,1],4)


def test_missing_duplicate_invalid_targets_and_lock(tmp_path):
    s=TargetShard(tmp_path/'s',{},[0,1],4)
    with s.lock():
        with pytest.raises(RuntimeError):
            with s.lock(): pass
    with pytest.raises(ValueError): s.commit([0],np.zeros((1,4),dtype=np.float32))
    with pytest.raises(ValueError): s.commit([0,0],probs(2))
    with pytest.raises(ValueError):
        finalize_split(tmp_path/'out',{},[s],[],2,4)
    s.commit([0,1],probs(2))
    with pytest.raises(ValueError):
        finalize_split(tmp_path/'out',{},[s],[{'row':0,'sample_id':'x'},{'row':1,'sample_id':'x'}],2,4)

def test_finalize_rechecks_committed_bytes(tmp_path):
    s=TargetShard(tmp_path/'s',{},[0],4)
    s.commit([0],probs(1))
    s.array[0] = probs(1)[0][::-1]
    s.array.flush()
    with pytest.raises(ValueError,match='checksum'):
        finalize_split(tmp_path/'out',{},[s],[{'row':0,'sample_id':'x'}],1,4)

def test_receipt_crash_retries_only_uncommitted(tmp_path, monkeypatch):
    import smolvlm_attention.teacher_targets.storage as storage
    s=TargetShard(tmp_path/'s',{},[0,1],4)
    original=storage.atomic_json
    def crash(path,value):
        if path.parent.name == 'commits': raise OSError('simulated receipt crash')
        return original(path,value)
    monkeypatch.setattr(storage,'atomic_json',crash)
    with pytest.raises(OSError): s.commit([0],probs(1))
    monkeypatch.setattr(storage,'atomic_json',original)
    recovered=TargetShard(tmp_path/'s',{},[0,1],4)
    assert recovered.pending_rows.tolist() == [0,1]
    recovered.commit([0],probs(1))
    assert TargetShard(tmp_path/'s',{},[0,1],4).pending_rows.tolist() == [1]


def test_constructor_respects_writer_lock(tmp_path):
    s=TargetShard(tmp_path/'s',{},[0],4)
    with s.lock():
        with pytest.raises(RuntimeError): TargetShard(tmp_path/'s',{},[0],4)


@pytest.mark.parametrize('field,value',[('axes',['visual_token','sample']),('dtype','<f8'),('normalization','softmax'),('identity',{'score_layer_index':21})])
def test_export_semantics_rejected(tmp_path,field,value):
    s=TargetShard(tmp_path/'s',{'score_layer_index':22,'split':'train'},[0],4)
    s.commit([0],probs(1))
    finalize_split(tmp_path/'out',s.identity,[s],[{'row':0,'sample_id':'x','split':'train'}],1,4)
    path=tmp_path/'out/manifest.json'
    metadata=json.loads(path.read_text())
    metadata[field]=value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError): validate_export(tmp_path/'out')


@pytest.mark.parametrize('addition',[{'answer':'private'},{'input_ids':[1]},{'split':'val'},{'sample_id':[]}])
def test_portable_mapping_revalidated(tmp_path,addition):
    from smolvlm_attention.sample_store import digest
    s=TargetShard(tmp_path/'s',{'split':'train'},[0],4)
    s.commit([0],probs(1))
    finalize_split(tmp_path/'out',s.identity,[s],[{'row':0,'sample_id':'x','split':'train'}],1,4)
    mapping=tmp_path/'out/samples.jsonl'
    mapping.write_text(json.dumps({'row':0,'sample_id':'x','split':'train',**addition})+'\n')
    path=tmp_path/'out/manifest.json'
    metadata=json.loads(path.read_text()); metadata['checksums']['samples.jsonl']=digest(mapping)
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError): validate_export(tmp_path/'out')

def test_duplicate_shards_and_non_fp32_rejected(tmp_path):
    s=TargetShard(tmp_path/'s',{},[0],4)
    with pytest.raises(ValueError): s.commit([0],probs(1).astype(np.float16))
    s.commit([0],probs(1))
    with pytest.raises(ValueError,match='Duplicate'):
        finalize_split(tmp_path/'out',{},[s,s],[{'row':0,'sample_id':'x'}],1,4)


def test_stale_pid_file_does_not_block_os_lock(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path/'writer.lock').write_text(str(__import__('os').getpid()))
    s=TargetShard(tmp_path,{},[0],4)
    s.commit([0],probs(1))
    assert s.pending_rows.size == 0

def test_initialization_crash_is_resumable_without_deleting_orphan(tmp_path,monkeypatch):
    import smolvlm_attention.teacher_targets.storage as storage
    original=storage.atomic_json
    def crash(path,value):
        if path.name == 'shard.json': raise OSError('metadata publication crash')
        return original(path,value)
    monkeypatch.setattr(storage,'atomic_json',crash)
    with pytest.raises(OSError): TargetShard(tmp_path,{},[0],4)
    monkeypatch.setattr(storage,'atomic_json',original)
    recovered=TargetShard(tmp_path,{},[0],4)
    assert recovered.pending_rows.tolist() == [0]
    assert len(list(tmp_path.glob('log_probs.uncommitted-*.npy'))) == 1

def test_portable_protocol_geometry_checked(tmp_path):
    identity={'split':'train','score_layer_index':22,'score_layer_number':23,
              'axes':['sample','visual_token'],'dtype':'<f4','normalization':'native',
              'protocol':{'score_layer_index':22,'score_layer_number':23,'visual_tokens':384,
                          'splits':{'train':1,'val':1},'normalization':'native'}}
    s=TargetShard(tmp_path/'s',identity,[0],384)
    s.commit([0],probs(1,384))
    finalize_split(tmp_path/'out',identity,[s],[{'row':0,'sample_id':'x','split':'train'}],1,384)
    path=tmp_path/'out/manifest.json'
    metadata=json.loads(path.read_text()); metadata['identity']['protocol']['splits']['train']=2
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='protocol'): validate_export(tmp_path/'out')
