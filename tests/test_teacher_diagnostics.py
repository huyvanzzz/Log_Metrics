import json
from pathlib import Path
import pytest


def test_diagnostic_launcher_reports_original_failure_and_stopped_peer(tmp_path, monkeypatch):
    from smolvlm_attention.teacher_targets import diagnostics as d, runtime
    run=tmp_path/'run.json'; run.write_text(json.dumps(dict(output_root=str(tmp_path))))
    original=runtime.subprocess
    children=[]
    class Child:
        def __init__(self):
            self.pid=101+len(children); self.returncode=-9 if not children else None
        def poll(self): return self.returncode
        def terminate(self): self.returncode=-15
        def wait(self,timeout=None): return self.returncode
    def create(*a,**kw):
        child=Child(); children.append(child); return child
    def launch(path):
        a=runtime.subprocess.Popen(['worker0']); b=runtime.subprocess.Popen(['worker1'])
        assert a.poll()==-9
        b.terminate(); b.wait()
        raise RuntimeError('Teacher worker failed')
    monkeypatch.setattr(d,'_popen',create)
    monkeypatch.setattr(d,'resource_snapshot',lambda:dict(cgroup_current_bytes=30,cgroup_limit_bytes=32,shm_free_bytes=8,memory_events=dict(oom_kill=2)))
    monkeypatch.setattr(runtime,'launch',launch)
    with pytest.raises(RuntimeError,match='worker failed'): d.launch(run)
    assert runtime.subprocess is original
    report=json.loads((tmp_path/'logs/worker_failure.json').read_text())
    assert report['first_failed_rank']==0
    assert report['workers'][0]['returncode']==-9 and report['workers'][0]['signal']=='SIGKILL'
    assert not report['workers'][0]['terminated_by_parent']
    assert report['workers'][1]['terminated_by_parent'] and report['workers'][1]['returncode']==-15
    assert report['resources_at_failure']['memory_events']['oom_kill']==2
    assert report['oom_kill_delta']==0


def test_success_does_not_write_failure_and_restores_launcher(tmp_path,monkeypatch):
    from smolvlm_attention.teacher_targets import diagnostics as d,runtime
    run=tmp_path/'run.json'; run.write_text(json.dumps(dict(output_root=str(tmp_path))))
    original=runtime.subprocess
    monkeypatch.setattr(runtime,'launch',lambda path:'complete')
    assert d.launch(run)=='complete'
    assert runtime.subprocess is original
    assert not (tmp_path/'logs/worker_failure.json').exists()
    assert (tmp_path/'logs/diagnostics_before.json').is_file()


def test_actual_child_exit_code_is_preserved(tmp_path, monkeypatch):
    import sys
    from smolvlm_attention.teacher_targets import diagnostics as d, runtime
    path = tmp_path/'run.json'
    path.write_text(json.dumps(dict(output_root=str(tmp_path))))
    def launch(run):
        child = runtime.subprocess.Popen([sys.executable, '-c', 'raise SystemExit(7)'])
        assert child.wait(timeout=15) == 7
        assert child.poll() == 7
        raise RuntimeError('Teacher worker failed')
    monkeypatch.setattr(runtime, 'launch', launch)
    with pytest.raises(RuntimeError):
        d.launch(path)
    report = json.loads((tmp_path/'logs/worker_failure.json').read_text())
    assert report['workers'][0]['returncode'] == 7
    assert report['workers'][0]['signal'] is None


def test_notebook_uses_diagnostics_only_for_extraction():
    import ast
    nb = json.loads(Path('notebooks/kaggle_smolvlm_teacher_index22_train_val.ipynb').read_text(encoding='utf-8'))
    sources = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    for source in sources:
        ast.parse(source)
    helper = next(s for s in sources if 'def run(*args):' in s)
    assert 'smolvlm_attention.teacher_targets.diagnostics' in helper
    assert "args[0] == 'run' else BASE" in helper
    from smolvlm_attention.teacher_targets.runtime import _sources
    assert not any(name.endswith('/diagnostics.py') for name in _sources())
