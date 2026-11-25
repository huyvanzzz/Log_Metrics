"""CPU checks for the actual supervisor and stress gate reused by mass tests."""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest


def test_shared_supervisor_stops_live_peer_after_failure(tmp_path,monkeypatch):
    from smolvlm_map_predictor import runtime
    original=runtime.subprocess.Popen
    processes=[]
    def launch(*args,**kwargs):
        process=original(*args,**kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(runtime.subprocess,'Popen',launch)
    with pytest.raises(RuntimeError,match='exited 7'):
        runtime._parallel([
            ([sys.executable,'-c','import sys; sys.exit(7)'],dict(os.environ),tmp_path/'failed.log'),
            ([sys.executable,'-c','import time; time.sleep(20)'],dict(os.environ),tmp_path/'peer.log'),
        ])
    assert len(processes)==2 and all(p.poll() is not None for p in processes)


def test_both_actual_stress_receipts_required(tmp_path,monkeypatch):
    from smolvlm_predictor_mass.runtime import _stress_gate
    monkeypatch.setenv('SMOLVLM_MAP_EVALUATION_SESSION','mass-test-session')
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs=[executor.submit(_stress_gate,{},tmp_path,rank,2,{'method':'mass'},
                              {'ok':True,'samples':64}) for rank in (0,1)]
        for job in jobs: job.result(timeout=10)
    assert len(list((tmp_path/'stress/mass-test-session').glob('rank_*.json')))==2
