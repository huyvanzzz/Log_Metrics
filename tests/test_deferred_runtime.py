import numpy as np
import pytest


def test_layer22_selection_matches_old_ranking_exactly():
    from smolvlm_topk_deferred.runtime import select_scores
    from smolvlm_topk_benchmark.data import topk_indices
    attention = np.zeros((3, 30, 384), dtype=np.float32)
    attention[:, 22] = np.linspace(0, .001, 384)
    attention[:, 21, :32] = .002
    rows = [dict(attention_row=i) for i in (2, 0, 1)]
    result = select_scores(attention, rows)
    np.testing.assert_array_equal(result, topk_indices(attention[[2, 0, 1], 22]))
    attention[:] = 0
    np.testing.assert_array_equal(select_scores(attention, rows), np.tile(np.arange(32), (3, 1)))


def test_fixed64_tail_and_resume_without_retry():
    from smolvlm_topk_deferred.runtime import schedule_batches
    rows = [dict(row_index=i) for i in range(113)]
    done = {0}
    calls = []
    saved = []
    schedule_batches(rows, done, lambda batch: calls.append(len(batch)) or batch,
                     lambda start, records: saved.append(start))
    assert calls == [49] and saved == [64]
    calls.clear()
    def oom(batch):
        calls.append(len(batch))
        raise RuntimeError('out of memory')
    with pytest.raises(RuntimeError, match='out of memory'):
        schedule_batches(rows, set(), oom, lambda *args: None)
    assert calls == [64]


def test_only_paired_protocol_allowed(tmp_path):
    import json
    from smolvlm_topk_deferred.runtime import load_config
    cfg = load_config()
    assert cfg['score_layer_index'] == 22 and cfg['prune_after_layer_index'] == 2
    assert cfg['batch_size'] == 64 and cfg['generation']['num_beams'] == 1
    bad = tmp_path / 'bad.json'
    cfg['score_layer_index'] = 23
    bad.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        load_config(bad)


def test_worker_failure_stops_peer_without_opening_stress_gate(tmp_path, monkeypatch):
    import torch
    from smolvlm_topk_deferred import runtime
    monkeypatch.setattr(runtime, 'read_checked', lambda path: dict(output_root=str(tmp_path)))
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    children = []
    class Child:
        def __init__(self, fail):
            self.returncode = 1 if fail else None
            self.terminated = False
        def poll(self):
            return self.returncode
        def terminate(self):
            self.terminated = True
            self.returncode = -15
        def wait(self, timeout=None):
            return self.returncode
    def spawn(*args, **kwargs):
        child = Child(not children)
        children.append(child)
        return child
    monkeypatch.setattr(runtime.subprocess, 'Popen', spawn)
    with pytest.raises(RuntimeError, match='sibling stopped'):
        runtime.launch(tmp_path/'run.json')
    assert children[1].terminated
    assert not list((tmp_path/'logs').glob('gate_*.json'))


def test_gate_waits_for_both_unique_ready_receipts_and_gpu_mapping(tmp_path, monkeypatch):
    import json
    import torch
    from smolvlm_topk_deferred import runtime, state
    monkeypatch.setattr(runtime, 'read_checked', lambda path: dict(output_root=str(tmp_path)))
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-first,GPU-second')
    checks = []
    monkeypatch.setattr(state, 'validate_result', lambda path, run, expected_config: checks.append(expected_config))
    children, devices = [], []
    class Child:
        returncode = None
        def poll(self):
            if self.gate.exists():
                self.returncode = 0
            return self.returncode
        def wait(self, timeout=None):
            return 0
        def terminate(self):
            raise AssertionError('successful workers must not be terminated')
    def spawn(args, **kwargs):
        from pathlib import Path
        rank = len(children)
        gate = Path(args[args.index('--gate')+1])
        assert not gate.exists()
        child = Child()
        child.gate = gate
        children.append(child)
        devices.append(kwargs['env']['CUDA_VISIBLE_DEVICES'])
        (tmp_path/'logs'/f'ready_{rank}.json').write_text(json.dumps({'gate':str(gate)}))
        return child
    monkeypatch.setattr(runtime.subprocess, 'Popen', spawn)
    runtime.launch(tmp_path/'run.json')
    assert devices == ['GPU-first', 'GPU-second']
    assert checks == list(runtime.CONFIGS)
