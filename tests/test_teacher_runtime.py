import json
import numpy as np
import pytest

from smolvlm_attention.teacher_targets.runtime import (
    load_config, shard_rows, process_pending, required_space, compatible_run,
)


def test_config_is_fixed_and_rejects_method_changes(tmp_path):
    from pathlib import Path
    source = Path(__file__).parents[1] / 'configs/smolvlm_teacher_index22_train_val.json'
    cfg = load_config(source)
    assert (cfg['score_layer_index'], cfg['score_layer_number'], cfg['batch_size'], cfg['gpu_count']) == (22, 23, 64, 2)
    for key, value in [('score_layer_index', 21), ('batch_size', 32), ('precision', 'fp16')]:
        changed = dict(cfg, **{key: value})
        path = tmp_path / 'bad.json'
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            load_config(path)


def test_sharding_covers_odd_split_without_overlap():
    a, b = shard_rows(7, 0), shard_rows(7, 1)
    np.testing.assert_array_equal(np.concatenate([a, b]), np.arange(7))
    assert not set(a) & set(b)
    assert required_space({'train': 341381, 'val': 19785})['numeric_payload_bytes'] == 554750976


def test_fixed_batch_no_retry_and_commits_successful_batches():
    class Store:
        pending_rows = np.arange(67)
        committed = []
        def commit(self, rows, values):
            self.committed.extend(rows.tolist())
    store = Store()
    calls = []
    def measure(rows):
        calls.append(len(rows))
        if len(calls) == 2:
            raise RuntimeError('out of memory')
        return np.zeros((len(rows), 2), dtype=np.float32)
    with pytest.raises(RuntimeError, match='out of memory'):
        process_pending(store, np.arange(67), measure)
    assert calls == [64, 3]
    assert len(store.committed) == 64


def test_resume_only_operational_paths_can_change():
    old = {'identity': {'layer': 22}, 'prompts': {'train': 'old'}, 'output_root': 'old'}
    new = dict(old, output_root='new', prompts={'train': 'new'})
    compatible_run(old, new)
    with pytest.raises(ValueError, match='identity'):
        compatible_run(old, dict(new, identity={'layer': 21}))


def test_worker_failure_stops_sibling_and_preserves_committed_files(tmp_path, monkeypatch):
    import torch
    import smolvlm_attention.teacher_targets.runtime as runtime
    run = dict(output_root=str(tmp_path), prompts={'train': 'unused'}, identity={})
    monkeypatch.setattr(runtime, '_read_run', lambda path: run)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    committed = tmp_path / 'internal' / 'committed.json'
    committed.parent.mkdir()
    committed.write_text('preserved')
    children = []
    class Child:
        def __init__(self, failed):
            self.returncode = 1 if failed else None
            self.terminated = False
        def poll(self):
            return self.returncode
        def terminate(self):
            self.terminated = True
            self.returncode = -15
        def wait(self, timeout=None):
            return self.returncode
    def popen(*args, **kwargs):
        child = Child(failed=not children)
        children.append(child)
        return child
    monkeypatch.setattr(runtime.subprocess, 'Popen', popen)
    with pytest.raises(RuntimeError, match='worker failed'):
        runtime.launch(tmp_path / 'run.json')
    assert children[1].terminated
    assert committed.read_text() == 'preserved'


def test_partial_bundle_whitelist_and_active_run_refusal(tmp_path, monkeypatch):
    import zipfile
    from filelock import FileLock, Timeout
    import smolvlm_attention.teacher_targets.runtime as runtime
    (tmp_path / 'run.json').write_text('{}')
    for relative in ('internal/prompts/train/rows.jsonl', 'internal/shards/train/0/log_probs.npy',
                     'internal/model.pt', 'internal/prepared_data/train.jsonl', 'features.float16.memmap'):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture')
    monkeypatch.setattr(runtime, '_read_run', lambda path: dict(output_root=str(tmp_path)))
    with FileLock(str(tmp_path / 'worker_0.lock')):
        with pytest.raises(Timeout):
            runtime.bundle(tmp_path / 'run.json', partial=True)
    archive = runtime.bundle(tmp_path / 'run.json', partial=True)
    with zipfile.ZipFile(archive) as source:
        assert set(source.namelist()) == {'run.json', 'internal/prompts/train/rows.jsonl',
                                         'internal/shards/train/0/log_probs.npy'}


def test_cache_receipt_binds_actual_feature_content_without_modifying_cache(tmp_path):
    from test_smolvlm_cache import fixture_cache
    from smolvlm_inference.cache import digest
    from smolvlm_attention.teacher_targets.runtime import validate_teacher_cache
    cache, prepared, identity, values = fixture_cache(tmp_path)
    payload = cache / 'features.float16.memmap'
    before = payload.read_bytes()
    receipt = validate_teacher_cache(cache, prepared, tmp_path / 'receipt')
    descriptor = json.loads(receipt.read_text())
    assert descriptor['payload_sha256'] == digest(payload)
    assert payload.read_bytes() == before
    # Finite edits with unchanged manifest are still a new cache identity.
    array = np.memmap(payload, mode='r+', dtype=np.float16)
    array[0] += 1
    array.flush()
    receipt2 = validate_teacher_cache(cache, prepared, tmp_path / 'receipt2')
    assert json.loads(receipt2.read_text())['payload_sha256'] != descriptor['payload_sha256']
    array[0] = np.nan
    array.flush()
    with pytest.raises(ValueError, match='nonfinite'):
        validate_teacher_cache(cache, prepared, tmp_path / 'receipt3')
