import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def test_plot_runtime_does_not_import_inference_libraries():
    import subprocess,sys
    script='''
import builtins
real_import=builtins.__import__
def checked(name,*args,**kwargs):
    if name.split('.')[0] in ('torch','transformers','tokenizers'):
        raise RuntimeError('inference import in plot path: '+name)
    return real_import(name,*args,**kwargs)
builtins.__import__=checked
from dinov2_attention.runtime import resolve_run_mode,load_results
'''
    result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_batch32_oom_and_resume_commit_once(tmp_path):
    from dinov2_attention.runtime import process_pending
    from smolvlm_attention.aggregate import AttentionAccumulator
    rows = [dict(sample_id=str(i), input_ids=[2,1]) for i in range(40)]
    accumulator = AttentionAccumulator((4,384),'run',[r['sample_id'] for r in rows])
    accumulator.add(['0'],np.full((1,4,384),.001))
    calls,events = [],[]
    def measure(batch):
        calls.append(len(batch))
        if len(batch)>16:
            raise torch.cuda.OutOfMemoryError('fixture')
        return torch.full((len(batch),4,384),.001)
    final = process_pending(rows,accumulator,measure,checkpoint=tmp_path/'state.npz',progress=events.append)
    assert calls == [32,16,16,7] and final==16
    assert accumulator.count == 40 and events[-1]['processed']==40
    process_pending(rows,accumulator,measure,checkpoint=tmp_path/'state.npz')
    assert calls == [32,16,16,7]


def test_measurement_hash_excludes_plotting(tmp_path):
    from dinov2_attention.runtime import measurement_source_hash
    for name in ('config.py','model.py','collector.py','data.py','plotting.py'):
        (tmp_path/name).write_text(name)
    original=measurement_source_hash(tmp_path)
    (tmp_path/'plotting.py').write_text('new colors')
    assert measurement_source_hash(tmp_path)==original
    (tmp_path/'collector.py').write_text('different query')
    assert measurement_source_hash(tmp_path)!=original


def _run(tmp_path):
    from dinov2_attention.runtime import atomic_json, measurement_source_hash
    from smolvlm_attention.aggregate import AttentionAccumulator
    rows=[dict(sample_id=str(i),input_ids=[2,1]) for i in range(3)]
    inputs=atomic_json(tmp_path/'prepared_test.json',dict(records=rows))
    identity=dict(measurement_source=measurement_source_hash(),prepared_test_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest())
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    run=dict(output_root=str(tmp_path),identity=identity,fingerprint=fingerprint,prepared_test=str(inputs),
             shard_count=2,test_count=3,source_commit='old-commit',source_sha256='old-source')
    atomic_json(tmp_path/'run.json',run)
    for shard,ids,value in ((0,['0','1'],.001),(1,['2'],.002)):
        a=AttentionAccumulator((4,384),fingerprint,ids)
        a.add(ids,np.full((len(ids),4,384),value))
        a.save(tmp_path/'data'/f'shard_{shard}'/'attention.npz')
        atomic_json(tmp_path/'logs'/f'shard_{shard}.json',dict(status='complete',stage='complete',processed=len(ids),
                                                           fingerprint=fingerprint))
    return run


def test_auto_plot_merges_unequal_shards_without_artifact_download(tmp_path):
    from dinov2_attention.runtime import resolve_run_mode,load_results
    _run(tmp_path)
    assert resolve_run_mode(tmp_path,'auto')=='plot'
    mean,metadata=load_results(tmp_path/'run.json',plot_source_commit='new-plot')
    np.testing.assert_allclose(mean,.004/3)
    assert metadata['count']==3 and metadata['measurement_source_commit']=='old-commit'
    assert metadata['plot_source_commit']=='new-plot'
    with pytest.raises(ValueError,match='exists'):
        resolve_run_mode(tmp_path,'measure')


def test_partial_resume_and_changed_measurement_rejected_before_writes(tmp_path,monkeypatch):
    from dinov2_attention import runtime
    assert runtime.resolve_run_mode(tmp_path,'auto')=='measure'
    with pytest.raises(ValueError):
        runtime.resolve_run_mode(tmp_path,'resume')
    _run(tmp_path)
    state=tmp_path/'data/shard_1/attention.npz'
    state.unlink()
    assert runtime.resolve_run_mode(tmp_path,'auto')=='resume'
    before=(tmp_path/'run.json').read_bytes()
    monkeypatch.setattr(runtime,'measurement_source_hash',lambda *args:'changed')
    with pytest.raises(ValueError,match='measurement'):
        runtime.prepare_run(tmp_path,mode='resume',gpu_count=2)
    assert (tmp_path/'run.json').read_bytes()==before


def test_wait_accepts_completed_status_after_process_exit(tmp_path,monkeypatch):
    from dinov2_attention import runtime
    _run(tmp_path)
    class Done:
        def poll(self): return 0
    monkeypatch.setattr(runtime,'_PROCESSES',{runtime._process_key(tmp_path/'run.json',i):Done() for i in (0,1)})
    runtime.wait_workers(tmp_path/'run.json',poll_seconds=0)


def test_wait_reports_worker_error_and_log_path(tmp_path,monkeypatch):
    from dinov2_attention import runtime
    run=_run(tmp_path)
    runtime.atomic_json(tmp_path/'logs/shard_0.json',dict(status='failed',error='fixture OOM',fingerprint=run['fingerprint']))
    class Failed:
        def poll(self): return 1
    monkeypatch.setattr(runtime,'_PROCESSES',{runtime._process_key(tmp_path/'run.json',0):Failed()})
    with pytest.raises(RuntimeError,match='fixture OOM.*shard_0.log'):
        runtime.wait_workers(tmp_path/'run.json',poll_seconds=0)


def test_preflight_checks_native_first_token_and_batch_parity():
    from dinov2_attention.runtime import numerical_preflight
    from dinov2_attention.model import CachedDinoT5
    from transformers import T5Config,T5ForConditionalGeneration
    cfg=T5Config(vocab_size=30,d_model=16,d_ff=24,num_layers=4,num_decoder_layers=4,
                 num_heads=2,d_kv=8,dropout_rate=0,decoder_start_token_id=0,pad_token_id=0,eos_token_id=1)
    model=CachedDinoT5(T5ForConditionalGeneration(cfg),vision_dim=8).eval()
    batch=dict(input_ids=torch.tensor([[3,4,1],[5,1,0]]),attention_mask=torch.tensor([[1,1,1],[1,1,0]]),
               visual_features=torch.randn(2,6,64,8))
    result=numerical_preflight(model,batch)
    assert result['shape']==[2,4,384] and result['native_first_token_parity']


def test_worker_cpu_fixture_end_to_end(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from dinov2_attention import runtime,data,model as adapter
    from transformers import T5Config,T5ForConditionalGeneration
    cfg=T5Config(vocab_size=30,d_model=16,d_ff=24,num_layers=4,num_decoder_layers=4,
                 num_heads=2,d_kv=8,dropout_rate=0,decoder_start_token_id=0,pad_token_id=0,eos_token_id=1)
    model=adapter.CachedDinoT5(T5ForConditionalGeneration(cfg),vision_dim=8).eval()
    run=_run(tmp_path)
    for state in tmp_path.rglob('attention.npz'): state.unlink()
    prepared=json.loads(Path(run['prepared_test']).read_text())
    for row in prepared['records']: row['feature_index']=0
    prepared['tokenizer_policy']={'fixture':True}
    runtime.atomic_json(run['prepared_test'],prepared)
    run['identity'].update(prepared_test_sha256=runtime.digest(run['prepared_test']),versions=runtime._versions(),
                           aggregation_source=runtime.aggregation_source_hash())
    run['fingerprint']=hashlib.sha256(runtime.canonical(run['identity'])).hexdigest()
    run.update(cache_descriptor='fixture',artifacts={'checkpoint':'fixture'})
    runtime.atomic_json(tmp_path/'run.json',run)
    reader=SimpleNamespace(array=np.zeros((1,6,64,8),dtype=np.float16))
    monkeypatch.setattr(data,'open_validated_cache',lambda path:reader)
    monkeypatch.setattr(data,'tokenizer_policy',lambda tokenizer:{'fixture':True})
    monkeypatch.setattr(adapter,'load_model',lambda *a,**k:(model,SimpleNamespace(pad_token_id=0)))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    runtime.run_worker(tmp_path/'run.json',0)
    runtime.run_worker(tmp_path/'run.json',1)
    mean,metadata=runtime.load_results(tmp_path/'run.json')
    assert mean.shape==(4,384) and metadata['count']==3
    assert not list((tmp_path/'logs').glob('*.pid.json'))
