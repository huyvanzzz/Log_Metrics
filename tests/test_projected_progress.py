"""Projected-cache progress without per-image console logging."""
import json
import numpy as np


def test_projected_progress_includes_reused_receipts_and_new_groups(tmp_path):
    from smolvlm_predictor_io.projected import build_cache
    raw=np.ones((3,6,2,2),np.float16);index={str(i):i for i in range(3)}
    opts=dict(batch_size=2,projected_shape=(6,2,2))
    build_cache(raw,index,tmp_path,lambda x:x,{},**opts)
    rows=[]
    build_cache(raw,index,tmp_path,lambda x:(_ for _ in ()).throw(AssertionError()),{},progress=rows.append,**opts)
    assert [r['committed'] for r in rows]==[2,3]
    assert all(r['reused'] for r in rows)


def test_cache_progress_rate_uses_only_new_groups_on_resume(tmp_path,monkeypatch,capsys):
    from smolvlm_predictor_io import progress
    now=[10.];monkeypatch.setattr(progress.time,'perf_counter',lambda:now[0]);monkeypatch.setattr(progress.time,'monotonic',lambda:now[0])
    writer=progress.ProjectionProgress(tmp_path,0)
    writer(dict(committed=32,total=96,reused=True))
    now[0]=12.
    writer(dict(committed=64,total=96,reused=False))
    path=tmp_path/'status/rank_0.json';r=json.loads(path.read_text())
    assert r['committed_new']==32 and r['resumed_groups']==32
    assert r['groups_per_second']==16. and r['estimated_remaining_seconds']==2.
    assert capsys.readouterr().out==''


def test_projected_cache_phase_supersedes_preflight():
    from smolvlm_predictor_io.progress import latest_phase
    a=dict(phase='projected_cache_preflight',rank=0)
    b=dict(phase='projected_cache',rank=1)
    assert latest_phase([a,b])==[b]

def test_notebook_displays_one_aggregate_cache_bar_without_json_output(tmp_path,monkeypatch,capsys):
    from smolvlm_predictor_io import progress
    out=tmp_path/'out'
    script=tmp_path/'fake_cache_build.py'
    script.write_text('import json,time\nfrom pathlib import Path\np=Path('+repr(str(out))+')/"status"\np.mkdir(parents=True)\nfor rank in (0,1):\n value=dict(phase="projected_cache",rank=rank,completed=20,total=20,updated_at=time.time(),elapsed_seconds=2.,committed_new=20,estimated_remaining_seconds=0.)\n (p/f"rank_{rank}.json").write_text(json.dumps(value))\n')
    bars=[]
    class Bar:
        def __init__(self,total,desc,unit):self.total=total;self.n=0;self.unit=unit;self.postfix='';bars.append(self)
        def set_postfix_str(self,text,refresh=False):self.postfix=text
        def refresh(self):pass
        def close(self):pass
    monkeypatch.setattr(progress,'_bar',Bar)
    progress.run_notebook('fake_cache_build','validate-inputs',[],out,tmp_path)
    assert len(bars)==1 and bars[0].unit=='group'
    assert bars[0].n==bars[0].total==40
    assert '20.00 groups/s' in bars[0].postfix and 'ETA 00:00:00' in bars[0].postfix
    assert capsys.readouterr().out==''
