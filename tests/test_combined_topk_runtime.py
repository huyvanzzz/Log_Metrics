import ast
import json
from pathlib import Path

import numpy as np
import pytest


def test_measurement_sources_exclude_presentation_and_scoring():
    from smolvlm_combined_topk.runtime import measurement_sources
    names = measurement_sources()
    assert 'smolvlm_attention/combined_collector.py' in names
    assert 'smolvlm_combined_topk/projected.py' in names
    assert not any('scoring' in n or 'notebook' in n or 'runtime.py' in n for n in names)


def test_fixed_batch_measurement_commits_each_batch():
    from smolvlm_combined_topk.attention_worker import process_pending
    class Store:
        completed = set()
        def write(self, rows, values):
            assert values.shape == (len(rows), 10, 384)
            self.pending = list(rows)
        def commit(self):
            self.completed.update(self.pending)
    store = Store()
    batches = []
    events = []
    def measure(rows):
        batches.append(list(rows))
        return np.full((len(rows), 10, 384), .001, np.float32)
    process_pending(list(range(130)), store, measure, events.append)
    assert list(map(len, batches)) == [64, 64, 2]
    assert store.completed == set(range(130))
    assert events[-1]['committed'] == 130
    process_pending(list(range(130)), store, measure, events.append)
    assert len(batches) == 3


def test_fixed_batch_does_not_retry_oom():
    from smolvlm_combined_topk.attention_worker import process_pending
    calls = []
    class Store:
        completed = set()
        def commit(self):
            pytest.fail('failed batch committed')
    def measure(rows):
        calls.append(len(rows))
        raise RuntimeError('CUDA out of memory')
    with pytest.raises(RuntimeError, match='[Ff]ixed batch 64'):
        process_pending(list(range(70)), Store(), measure)
    assert calls == [64]


def test_notebook_is_reproducible_and_ten_cells():
    from scripts.build_combined_topk_notebook import make_notebook
    nb = make_notebook()
    saved = json.loads(Path('notebooks/kaggle_smolvlm_combined_attention_topk_benchmark.ipynb').read_text(encoding='utf-8'))
    assert nb == saved
    cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    assert len(cells) == 10
    for cell in cells:
        ast.parse(cell)
    assert 'PROJECTED_CACHE_INPUT' in cells[1]
    assert 'BATCH_SIZE = 64' in cells[1]
    assert 'smolvlm_combined_topk' in cells[0]
    assert all(c['outputs'] == [] for c in nb['cells'] if c['cell_type'] == 'code')


def test_prepare_actual_run_contract_and_incompatible_resume(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import smolvlm_combined_topk.runtime as runtime
    import smolvlm_combined_topk.projected as projected
    import smolvlm_combined_topk.inputs as inputs
    import smolvlm_predictor_io.setup as setup
    import smolvlm_inference.model as native
    import smolvlm_combined_topk.resume as resume
    from smolvlm_combined_topk.store import read_run
    from test_combined_topk_store import rows
    checkpoint=tmp_path/'checkpoint.pt';checkpoint.write_bytes(b'fixture')
    compact=tmp_path/'compact.json';compact.write_text('{}')
    cache=tmp_path/'cache';cache.mkdir()
    model=SimpleNamespace(visual_token_len=384,geometry={'text_dim':576},
        native=SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=30))),
        architecture_signature={'native':'fixture'},processor=None)
    monkeypatch.setattr(projected,'discover_projected_cache',lambda *a,**kw:cache)
    monkeypatch.setattr(projected,'validate_projected_cache',lambda *a,**kw:tmp_path/'receipt.json')
    monkeypatch.setattr(projected,'open_projected_receipt',lambda *a:SimpleNamespace(manifest={'fingerprint':'cache'}))
    monkeypatch.setattr(native,'load_model',lambda *a,**kw:model)
    monkeypatch.setattr(native,'processor_policy',lambda *a:{})
    monkeypatch.setattr(setup,'resolve_artifacts',lambda *a,**kw:{'model':{'checkpoint':str(checkpoint)},'prepared_dir':str(tmp_path)})
    monkeypatch.setattr(inputs,'prepare_inputs',lambda *a,**kw:{'store':str(compact),'rows':rows(5)})
    config=runtime.prepare_run(tmp_path/'output',source_commit='first')
    monkeypatch.setattr(resume,'resume_artifacts',lambda *a,**kw:config)
    run=read_run(config)
    assert run['count']==5 and run['batch_size']==64 and run['shard_count']==2
    assert run['geometry']==[10,384] and run['source_commit']=='first'
    assert runtime.prepare_run(tmp_path/'output')==config
    previous=config.read_bytes()
    monkeypatch.setattr(runtime,'measurement_sources',lambda:{'changed':'different'})
    with pytest.raises(ValueError,match='identity changed'):
        runtime.prepare_run(tmp_path/'output')
    assert config.read_bytes()==previous


def test_wait_final_commit_still_waits_for_gpu_process_exit(tmp_path,monkeypatch):
    import smolvlm_combined_topk.runtime as runtime
    from smolvlm_combined_topk.store import write_run,read_run,load_shard
    from smolvlm_combined_topk.config import LAYER_IDS
    from test_combined_topk_store import rows
    config=write_run(tmp_path,dict(identity={},batch_size=64,layer_ids=list(LAYER_IDS),
        geometry=[10,384],shard_count=2),rows(5))
    run=read_run(config);logs=tmp_path/'logs';logs.mkdir()
    for i in (0,1):
        store=load_shard(run,i)
        store.write(store.global_rows,np.full((len(store.global_rows),10,384),.001,np.float32));store.commit()
        (logs/f'attention_{i}.status.json').write_text(json.dumps(dict(status='complete',stage='complete',
            fingerprint=run['fingerprint'],processed=len(store.completed),committed=len(store.completed))))
    class Process:
        code=None
        def poll(self):return self.code
    processes=[Process(),Process()]
    for i,p in enumerate(processes):monkeypatch.setitem(runtime._PROCESSES,(str(config),i),p)
    monkeypatch.setattr(runtime,'_alive',lambda path:any(p.code is None for p in processes))
    sleeps=[]
    def sleep(seconds):
        sleeps.append(seconds)
        for p in processes:p.code=0
    monkeypatch.setattr(runtime.time,'sleep',sleep)
    runtime.wait_workers(config,poll_seconds=.001)
    assert sleeps==[.001]
