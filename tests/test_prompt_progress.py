import ast
import inspect
import json

import pytest


class Bar:
    instances = []

    def __init__(self, **kwargs):
        self.n = 0
        self.total = kwargs['total']
        self.closed = False
        self.instances.append(self)

    def update(self, amount):
        self.n += amount

    def set_postfix_str(self, value, **kwargs):
        self.stage = value

    def close(self):
        self.closed = True


@pytest.mark.parametrize('failed', [False, True])
def test_wait_uses_status_counts_including_resumed_samples(tmp_path, monkeypatch, failed):
    from smolvlm_attention import runtime
    config = tmp_path / 'run.json'
    config.write_text(json.dumps(dict(output_root=str(tmp_path), expected_test_count=7)))
    logs = tmp_path / 'logs'
    logs.mkdir()
    for name in ['base', 'reverse']:
        (logs / f'{name}.status.json').write_text(json.dumps(dict(
            status='failed' if failed and name == 'reverse' else 'complete',
            stage='failed' if failed and name == 'reverse' else 'complete',
            processed=7, total=7, error='test error')))
    class Process:
        def __init__(self, code):
            self.returncode = code
        def poll(self):
            return self.returncode
    monkeypatch.setattr(runtime, '_PROCESSES', {(str(config.resolve()), n): Process(
        1 if failed and n == 'reverse' else 0) for n in ['base', 'reverse']})
    Bar.instances = []
    monkeypatch.setattr(runtime, 'tqdm', Bar, raising=False)
    if failed:
        with pytest.raises(RuntimeError, match='reverse.log'):
            runtime.wait_workers(config)
    else:
        runtime.wait_workers(config)
        assert [bar.n for bar in Bar.instances] == [7, 7, 14]
    assert len(Bar.instances) == 3
    assert all(bar.closed for bar in Bar.instances)


def test_worker_writes_failure_status_before_cuda_loading(tmp_path, monkeypatch):
    from smolvlm_attention import worker
    path = tmp_path / 'run.json'
    path.write_text(json.dumps(dict(output_root=str(tmp_path), models={'base': {}}, expected_test_count=7)))
    monkeypatch.setattr(worker.torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='GPU'):
        worker.run_worker(path, 'base')
    status = json.loads((tmp_path / 'logs/base.status.json').read_text())
    assert status['status'] == status['stage'] == 'failed'
    assert status['processed'] == 0 and status['total'] == 7


def test_wait_polls_before_reading_final_status(tmp_path, monkeypatch):
    from smolvlm_attention import runtime
    config = tmp_path / 'run.json'
    config.write_text(json.dumps(dict(output_root=str(tmp_path), expected_test_count=7)))
    logs = tmp_path / 'logs'
    logs.mkdir()
    class Process:
        def __init__(self, name):
            self.name = name
        def poll(self):
            (logs / f'{self.name}.status.json').write_text(json.dumps(dict(
                status='complete', stage='complete', processed=7)))
            return 0
    for name in ('base','reverse'):
        (logs / f'{name}.status.json').write_text(json.dumps(dict(status='running',stage='measuring',processed=6)))
    monkeypatch.setattr(runtime, '_PROCESSES', {(str(config.resolve()),name):Process(name) for name in ('base','reverse')})
    monkeypatch.setattr(runtime, 'tqdm', Bar)
    Bar.instances = []
    runtime.wait_workers(config)
    assert [bar.n for bar in Bar.instances] == [7,7,14]


def test_prompt_worker_starts_at_batch_16_without_tuning():
    from smolvlm_attention import worker
    tree = ast.parse(inspect.getsource(worker.run_worker))
    batch = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'batch_size' for t in node.targets))
    assert isinstance(batch.value, ast.Constant) and batch.value.value == 16
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == 'tune_batch' for node in ast.walk(tree))


def test_prompt_progress_flag_does_not_change_prepared_inputs(tmp_path):
    from test_smolvlm_cache import fixture_cache
    from test_smolvlm_model import tiny_components
    from smolvlm_attention.data import validate_cache, open_validated_cache, prepare_test
    cache, prepared, identity, _ = fixture_cache(tmp_path)
    reader = open_validated_cache(validate_cache(cache, prepared, tmp_path / 'out'))
    _, processor = tiny_components()
    quiet = prepare_test(reader, processor, identity['camera_order'], tmp_path / 'quiet', 512)
    visible = prepare_test(reader, processor, identity['camera_order'], tmp_path / 'visible', 512, progress=True)
    assert quiet.read_bytes() == visible.read_bytes()


def test_batch16_oom_progress_counts_only_successful_samples():
    import numpy as np
    import torch
    from smolvlm_attention.aggregate import AttentionAccumulator
    from smolvlm_attention.worker import process_pending
    rows = [dict(sample_id=str(i), input_ids=[1]) for i in range(20)]
    accumulator = AttentionAccumulator((1, 1), 'run', [r['sample_id'] for r in rows])
    accumulator.add(['0'], np.ones((1, 1, 1)))
    calls, events = [], []
    def measure(batch):
        calls.append(len(batch))
        if len(batch) > 8:
            raise torch.cuda.OutOfMemoryError('test OOM')
        return np.ones((len(batch), 1, 1))
    final = process_pending(rows, accumulator, measure, batch_size=16, device='cpu', progress=events.append)
    assert calls == [16, 8, 8, 3] and final == 8
    assert events[0] == dict(event='oom_retry', batch_size=8, processed=1)
    assert [e['processed'] for e in events[1:]] == [9, 17, 20]
