import ast
import json
from pathlib import Path

import numpy as np
import pytest


def fixture_run(tmp_path, count=5, config=None):
    from smolvlm_attention.combined_state import write_run, read_run
    rows = [dict(row_index=i, sample_id=f'q-{i}', eval_id=i, input_ids=[9]*384+[2],
                 image_token_id=9, prompt_length=385, query_position=384, query_token_id=2)
            for i in range(count)]
    config = config or dict(identity={'policy':'fixture'}, artifacts={}, cache_descriptor='fixture',
                           source_commit='measurement-commit', source_sha256='fixture')
    return read_run(write_run(tmp_path, config, rows))


def complete_shards(run):
    from smolvlm_attention.combined_state import load_shard, shard_path, save_shard
    for shard in (0, 1):
        acc = load_shard(run, shard)
        values = np.full((len(acc.sample_ids),15,384), .001*(shard+1), np.float32)
        acc.add(acc.sample_ids, values)
        save_shard(acc, shard_path(run, shard))


def test_modes_finalize_and_plot_without_artifacts(tmp_path, monkeypatch):
    import smolvlm_attention.combined_runtime as runtime
    assert runtime.resolve_run_mode(tmp_path) == 'measure'
    run = fixture_run(tmp_path)
    assert runtime.resolve_run_mode(tmp_path) == 'resume'
    with pytest.raises(ValueError):runtime.resolve_run_mode(tmp_path, 'measure')
    with pytest.raises(ValueError):runtime.resolve_run_mode(tmp_path, 'plot')
    complete_shards(run)
    runtime.finalize_results(run['config_path'])
    assert runtime.resolve_run_mode(tmp_path) == 'plot'
    # Plot mode must not consult runtime CUDA/source compatibility or artifact loaders.
    monkeypatch.setattr(runtime, 'assert_measurement_compatible', lambda *a: pytest.fail('plot checks GPU measurement'))
    assert runtime.prepare_run(tmp_path, mode='plot') == Path(run['config_path'])
    mean, metadata, scales, figures = runtime.load_results(run['config_path'], plot_source_commit='plot-commit')
    np.testing.assert_allclose(mean, .0014, atol=1e-9)
    assert metadata['count'] == 5
    assert figures['measurement_source_commit'] == 'measurement-commit'
    assert figures['plot_source_commit'] == 'plot-commit'
    assert scales['attention_scale'] == 'log'
    assert not (tmp_path/'figures.zip').exists()
    path=tmp_path/'combined_attention.npz'
    with path.open('ab') as stream:stream.write(b'corrupt')
    with pytest.raises(ValueError, match='hash'):runtime.load_results(run['config_path'])


def test_notebook_builder_reproducible_and_parse():
    from scripts.build_combined_attention_notebook import make_notebook
    nb=make_notebook()
    saved=json.loads(Path('notebooks/kaggle_smolvlm_combined_visual_attention.ipynb').read_text(encoding='utf-8'))
    assert nb == saved
    cells=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    assert len(cells)==10
    for cell in cells:ast.parse(cell)
    assert 'BATCH_SIZE = 64' in cells[1]
    assert 'combined_runtime' in cells[0]
    assert 'plot' in cells[0]
    assert all(c['outputs']==[] for c in nb['cells'] if c['cell_type']=='code')


@pytest.mark.parametrize('fail_measurement',[False,True])
def test_worker_status_preflight_and_fixed_batch_failure(tmp_path,monkeypatch,fail_measurement):
    import torch
    from types import SimpleNamespace
    import smolvlm_attention.combined_runtime as runtime
    import smolvlm_attention.combined_collector as collector
    import smolvlm_inference.model as loader
    import smolvlm_attention.data as data
    from smolvlm_attention.combined_worker import run_worker
    from smolvlm_attention.combined_state import load_shard
    from smolvlm_attention.sample_store import digest
    checkpoint=tmp_path/'fixture.pt';checkpoint.write_bytes(b'fixture')
    identity=dict(checkpoint_sha256=digest(checkpoint),data=dict(cache_fingerprint='cache',
        prepared_hashes={},model_id='native',native_revision='revision',image_token_id=9))
    config=dict(identity=identity,artifacts={'model':{'checkpoint':str(checkpoint)}},cache_descriptor='fixture')
    run=fixture_run(tmp_path,config=config)
    monkeypatch.setattr(runtime,'assert_measurement_compatible',lambda run:None)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda device:'fixture')
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda device:0)
    native=SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=30),
        image_token_id=9),eval=lambda:None)
    model=SimpleNamespace(native=native,visual_token_len=384,geometry={},processor=None)
    monkeypatch.setattr(loader,'load_model',lambda *a,**kw:model)
    monkeypatch.setattr(loader,'processor_policy',lambda processor:{})
    reader=SimpleNamespace(manifest=dict(fingerprint='cache',prepared_hashes={},model_id='native',
        revision='revision',geometry={},processor_policy={}))
    monkeypatch.setattr(data,'open_validated_cache',lambda path:reader)
    monkeypatch.setattr(data,'make_batch',lambda reader,rows,*a:{'count':len(rows)})
    def preflight(model,rows,make,progress=None):
        progress('native oracle passed')
        return dict(status='passed')
    monkeypatch.setattr(collector,'numerical_preflight',preflight)
    def collect(model,batch,**kw):
        if fail_measurement:raise torch.cuda.OutOfMemoryError('fixture OOM')
        return torch.zeros((batch['count'],15,384))
    monkeypatch.setattr(collector,'collect_combined_visual_attention',collect)
    if fail_measurement:
        with pytest.raises(RuntimeError,match='Fixed batch 64'):run_worker(run['config_path'],0)
    else:run_worker(run['config_path'],0)
    state=json.loads((tmp_path/'logs/combined_0.status.json').read_text())
    assert state['status']==('failed' if fail_measurement else 'complete')
    assert state['detail']=='native oracle passed'
    assert load_shard(run,0).count==(0 if fail_measurement else 3)
    assert (tmp_path/'logs/combined_0.preflight.json').exists()


def test_launch_duplicate_lock_and_wait_for_process_exit(tmp_path,monkeypatch):
    import torch
    from filelock import FileLock,Timeout
    import smolvlm_attention.combined_runtime as runtime
    from smolvlm_attention.sample_store import atomic_json
    run=fixture_run(tmp_path)
    monkeypatch.setattr(runtime,'assert_measurement_compatible',lambda run:None)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    monkeypatch.setattr(runtime,'_pid_alive',lambda path:False)
    monkeypatch.setattr(runtime,'_PROCESSES',{})
    spawned=[]
    class Process:
        def __init__(self,*a,**kw):self.pid=7000+len(spawned);self.code=None;spawned.append(self)
        def poll(self):return self.code
    monkeypatch.setattr(runtime.subprocess,'Popen',Process)
    logs=tmp_path/'logs';logs.mkdir()
    with FileLock(str(logs/'launch.lock'),timeout=0):
        with pytest.raises(Timeout):runtime.launch_workers(run['config_path'])
    runtime.launch_workers(run['config_path']);runtime.launch_workers(run['config_path'])
    assert len(spawned)==2
    complete_shards(run)
    for shard,count in ((0,3),(1,2)):
        atomic_json(logs/f'combined_{shard}.status.json',dict(status='complete',stage='complete',
            committed=count,processed=count,fingerprint=run['fingerprint']))
    def finish(seconds):
        for p in spawned:p.code=0
    monkeypatch.setattr(runtime.time,'sleep',finish)
    runtime.wait_workers(run['config_path'],poll_seconds=0)
