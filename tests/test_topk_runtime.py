import ast
import json
import sys

import pytest

from test_topk_state import run_fixture, samples, predictions


def test_configuration_finishes_before_next_fixed_batch_and_resume(tmp_path):
    from smolvlm_topk_benchmark.worker import schedule_batches
    rows=samples(67); configs=('layer_00','layer_01','layer_02','layer_09')
    done={name:set() for name in configs}; done[configs[0]].add(0)
    fused=[]; calls=[]; saved=[]
    def build(selected):
        fused.append([r['eval_id'] for r in selected]); return object()
    def generate(name,selected,shared):
        calls.append((name,len(selected))); return predictions(selected)
    schedule_batches(rows,configs,done,build,generate,lambda name,start,result:saved.append((name,start)))
    assert list(map(len,fused))==[3,64,3,64,3,64,3]
    assert calls==[(configs[0],3)]+[(c,size) for c in configs[1:] for size in (64,3)]
    assert len(saved)==7
    schedule_batches(rows,configs,done,build,generate,lambda *a:pytest.fail('already committed'))
    assert len(saved)==7


def test_oom_is_not_retried_or_rebatched(tmp_path):
    import torch
    from smolvlm_topk_benchmark.worker import schedule_batches
    calls=[]
    def fail(name,rows,shared):
        calls.append(len(rows)); raise torch.cuda.OutOfMemoryError('fixture')
    with pytest.raises(RuntimeError,match='batch 64'):
        schedule_batches(samples(),('layer_00',),{'layer_00':set()},lambda r:None,fail,lambda *a:None)
    assert calls==[64]


def test_mode_is_separate_for_each_config_set(tmp_path):
    from smolvlm_topk_benchmark.runtime import resolve_run_mode
    assert resolve_run_mode(tmp_path,'first')=='measure'
    run=run_fixture(tmp_path)
    assert resolve_run_mode(tmp_path,'first')=='resume'
    assert resolve_run_mode(tmp_path,'second')=='resume'
    with pytest.raises(ValueError,match='incomplete'):
        resolve_run_mode(tmp_path,'first','score')


def test_portable_runtime_import_does_not_load_torch():
    import subprocess
    subprocess.run([sys.executable,'-c','import sys; import smolvlm_topk_benchmark.runtime; assert "torch" not in sys.modules'],check=True)


def test_notebook_and_embedded_guard():
    from scripts.build_topk_benchmark_notebook import make_notebook
    nb=make_notebook()
    cells=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    assert len(cells)==8
    for c in cells:ast.parse(c)
    assert "CONFIG_SET = 'first'" in cells[0]+cells[1]
    assert 'smolvlm_topk_benchmark.runtime' in cells[0]
    assert 'BATCH_SIZE = 64' in cells[1]


def test_two_waves_scoring_and_readonly_merge(tmp_path,monkeypatch):
    import numpy as np
    import zipfile
    import smolvlm_topk_benchmark.runtime as runtime
    from smolvlm_topk_benchmark.config import CONFIG_SETS
    from smolvlm_topk_benchmark.scoring import METRICS
    from smolvlm_topk_benchmark.state import write_selection,save_batch,input_rows,digest
    first=run_fixture(tmp_path/'first',2)
    def fill(run,names):
        for name in names:
            if name=='base':write_selection(run,name)
            else:write_selection(run,name,np.tile(np.arange(32,dtype=np.uint16),(2,1)),np.tile(np.arange(32,dtype=np.int32),(2,1)))
            save_batch(run,name,0,predictions(input_rows(run)))
    fill(first,CONFIG_SETS['first'])
    calls=[]
    def scorer(output,*,input_path=None,expected_count=None):
        if input_path is None:
            from smolvlm_topk_benchmark.state import atomic_json
            atomic_json(output,{'preflight':True})
            return {'preflight':True}
        calls.append(str(input_path))
        value=.8 if input_path.parent.name=='base' else .5
        result=dict(metrics=dict.fromkeys(METRICS,value),metrics_display=dict.fromkeys(METRICS,value*100),
                    diagnostics={'empty_prediction_count':0},scoring_identity={})
        from smolvlm_topk_benchmark.state import atomic_json
        atomic_json(output,result)
        return result
    monkeypatch.setattr(runtime,'_scorer_command',scorer)
    monkeypatch.setattr(runtime,'_scoring_fingerprint',lambda value:'fixture-scorer')
    summary=runtime.score_results(first['config_path'],'first')
    assert len(summary['results'])==8 and not summary['baseline_available']
    assert all(r['delta_vs_base'] is None for r in summary['results'])
    runtime.score_results(first['config_path'],'first')
    assert len(calls)==8  # Valid score cache avoids recomputation.
    archive=runtime.bundle_results(first['config_path'])
    previous=tmp_path/'prior'; previous.mkdir()
    with zipfile.ZipFile(archive) as z:
        assert not any('attention.npy' in n or 'batches/' in n for n in z.namelist())
        z.extractall(previous)
    before={p.relative_to(previous):digest(p) for p in previous.rglob('*') if p.is_file()}
    second=run_fixture(tmp_path/'second',2)
    fill(second,CONFIG_SETS['second'])
    combined=runtime.score_results(second['config_path'],'second',previous_results=previous)
    assert len(combined['results'])==16 and combined['baseline_available']
    assert combined['results'][0]['delta_vs_base']['CIDEr']==pytest.approx(-.3)
    assert before=={p.relative_to(previous):digest(p) for p in previous.rglob('*') if p.is_file()}
    assert runtime.resolve_run_mode(tmp_path/'second','second')=='score'


@pytest.mark.parametrize('config_set,worker_count', [('first',2), ('second',2)])
def test_launch_lock_duplicate_workers_and_wait_exit_race(tmp_path,monkeypatch,config_set,worker_count):
    import torch
    from filelock import FileLock,Timeout
    import smolvlm_topk_benchmark.runtime as runtime
    from smolvlm_topk_benchmark.config import worker_configs, active_workers
    from smolvlm_topk_benchmark.state import atomic_json
    run=run_fixture(tmp_path,2); path=run['config_path']
    monkeypatch.setattr(runtime,'assert_compatible',lambda run:None)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:worker_count)
    monkeypatch.setattr(runtime,'_pid_alive',lambda path:False)
    monkeypatch.setattr(runtime,'_PROCESSES',{})
    spawned=[]
    class Process:
        def __init__(self,*a,**kw):
            self.pid=6000+len(spawned);self.code=None;spawned.append(self)
        def poll(self):return self.code
    monkeypatch.setattr(runtime.subprocess,'Popen',Process)
    logs=tmp_path/'logs';logs.mkdir()
    with FileLock(str(logs/'launch.lock'),timeout=0):
        with pytest.raises(Timeout):runtime.launch_workers(path,config_set)
    runtime.launch_workers(path,config_set);runtime.launch_workers(path,config_set)
    assert len(spawned)==worker_count
    for worker in active_workers(config_set):
        status=json.loads((logs/f'bench_{config_set}_{worker}.status.json').read_text())
        assert status['total']==2*len(worker_configs(config_set,worker))
        atomic_json(logs/f'bench_{config_set}_{worker}.status.json',dict(status='complete',stage='complete',
            fingerprint=run['fingerprint'],counts={name:2 for name in worker_configs(config_set,worker)}))
    def finish(seconds):
        for p in spawned:p.code=0
    monkeypatch.setattr(runtime.time,'sleep',finish)
    runtime.wait_workers(path,config_set,poll_seconds=0)
