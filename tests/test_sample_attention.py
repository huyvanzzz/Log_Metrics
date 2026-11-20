"""Per-question attention durability and portable export contracts."""
import json
import sys
import subprocess
from pathlib import Path

import numpy as np
import pytest

from smolvlm_attention.sample_store import (
    ShardStore, atomic_json, fingerprint, open_prompt_attention, shard_bounds,
    finalize_export, bundle_export, write_run, read_run,
)


@pytest.fixture(autouse=True)
def current_store_api():
    # The source-activation regression deliberately evicts package modules.
    # Keep fault-injection targets and these aliases on the same module object.
    from smolvlm_attention import sample_store
    for name in ('ShardStore', 'atomic_json', 'fingerprint', 'open_prompt_attention',
                 'shard_bounds', 'finalize_export', 'bundle_export', 'write_run', 'read_run'):
        globals()[name] = getattr(sample_store, name)


def rows(count=5):
    return [dict(row_index=i, sample_id=f'test-{i}', eval_id=i,
                 question=f'Question {i}', answer='A red car.', camera_paths={'Front': str(i)},
                 input_ids=[9, 9, 7], image_token_id=9, prompt_length=3,
                 query_position=2, query_token_id=7, feature_index=i, feature_key=f'key-{i}')
            for i in range(count)]


def make_run(tmp_path, count=5, shards=2):
    config = dict(identity={'policy': 'test'}, shard_count=shards, batch_size=32,
                  geometry=[2, 2], source_commit='fixture')
    return write_run(tmp_path, config, rows(count))


def test_shards_and_uncommitted_resume(tmp_path):
    assert shard_bounds(16817, 2, 0) == (0, 8409)
    assert shard_bounds(16817, 2, 1) == (8409, 16817)
    store = ShardStore(tmp_path, 'identity', [0, 1, 2], (2, 2))
    store.write([1], np.full((1, 2, 2), .1, dtype=np.float32))
    # Numeric data alone must not make a row complete.
    reopened = ShardStore(tmp_path, 'identity', [0, 1, 2], (2, 2))
    assert reopened.completed == set()
    store.commit(batch_size=8)
    resumed = ShardStore(tmp_path, 'identity', [0, 1, 2], (2, 2))
    assert resumed.completed == {1} and resumed.batch_size == 8
    with pytest.raises(ValueError, match='already'):
        resumed.write([1], np.full((1, 2, 2), .2, dtype=np.float32))
    with pytest.raises(ValueError, match='identity'):
        ShardStore(tmp_path, 'different', [0, 1, 2], (2, 2))


def test_corruption_and_invalid_attention(tmp_path):
    store = ShardStore(tmp_path, 'id', [0, 1], (2, 2))
    for value in (np.nan, -1, .6):
        with pytest.raises(ValueError):
            store.write([0], np.full((1, 2, 2), value, dtype=np.float32))
    with pytest.raises(ValueError, match='FP32'):
        store.write([0], np.zeros((1, 2, 2), dtype=np.float64))
    store.write([0], np.zeros((1, 2, 2), dtype=np.float32))
    store.commit(batch_size=32)
    store.array[0, 0, 0] = .1
    store.array.flush()
    with pytest.raises(ValueError, match='checksum'):
        ShardStore(tmp_path, 'id', [0, 1], (2, 2))


def test_resume_does_not_overwrite_incompatible_mapping(tmp_path):
    path = make_run(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.glob('*.json*')}
    changed = rows()
    changed[0]['question'] = 'changed'
    with pytest.raises(ValueError, match='incompatible'):
        write_run(tmp_path, dict(identity={'policy': 'test'}, shard_count=2,
                  batch_size=32, geometry=[2, 2]), changed)
    assert before == {p.name: p.read_bytes() for p in tmp_path.glob('*.json*')}
    with pytest.raises(ValueError):
        write_run(tmp_path, read_run(path), rows(), mode='measure')


def test_export_mapping_merge_reader_zip(tmp_path):
    path = make_run(tmp_path)
    run = read_run(path)
    with pytest.raises(ValueError, match='incomplete'):
        finalize_export(path, progress=False)
    for shard in range(2):
        start, end = shard_bounds(5, 2, shard)
        store = ShardStore(tmp_path / 'shards' / str(shard), run['fingerprint'],
                           list(range(start, end)), (2, 2))
        indices = list(reversed(range(start, end)))
        values = np.stack([np.full((2, 2), (i + 1) / 100, np.float32) for i in indices])
        store.write(indices, values)
        store.commit(batch_size=32)
    export = finalize_export(path, progress=False)
    result = open_prompt_attention(export)
    assert isinstance(result.attention, np.memmap)
    assert result.attention.shape == (5, 2, 2) and not result.attention.flags.writeable
    assert result.sample_index['test-4'] == 4
    np.testing.assert_allclose(result.attention[:, 0, 0], [.01, .02, .03, .04, .05])
    assert result.samples[0]['answer'] == 'A red car.'
    assert finalize_export(path, progress=False) == export
    import zipfile
    archive = bundle_export(export)
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == {'attention.npy', 'samples.jsonl', 'metadata.json', 'manifest.json', 'README.md'}
    # Consumer must not silently load an altered mapping.
    with (export / 'samples.jsonl').open('a') as f:
        f.write('\n')
    with pytest.raises(ValueError, match='hash'):
        open_prompt_attention(export)


def test_portable_reader_does_not_import_torch():
    code = 'import sys; import smolvlm_attention.sample_store; assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', code], check=True)


def test_batch_oom_retry_persists_reduced_size(tmp_path):
    import torch
    from smolvlm_attention.sample_worker import process_pending
    selected = rows(5)
    store = ShardStore(tmp_path, 'id', list(range(5)), (2, 2))
    calls = []
    def measure(batch):
        calls.append([r['row_index'] for r in batch])
        if len(batch) > 2:
            raise torch.cuda.OutOfMemoryError('fixture')
        return np.full((len(batch), 2, 2), .1, dtype=np.float32)
    process_pending(selected, store, measure, commit_every=100)
    assert [len(c) for c in calls[:5]] == [5, 5, 5, 4, 2]
    resumed = ShardStore(tmp_path, 'id', list(range(5)), (2, 2))
    assert resumed.batch_size == 2 and resumed.completed == set(range(5))
    calls.clear()
    process_pending(selected, resumed, measure)
    assert calls == []


def test_notebook_parse_and_helper_worker_guard():
    import ast
    from scripts.build_sample_attention_notebook import make_notebook
    book = make_notebook()
    code = [''.join(c['source']) for c in book['cells'] if c['cell_type'] == 'code']
    assert len(code) == 7
    for cell in code:
        ast.parse(cell)
    assert 'BATCH_SIZE = 32' in code[1]
    assert 'smolvlm_attention.sample_runtime' in code[0]
    assert all(c['outputs'] == [] for c in book['cells'] if c['cell_type'] == 'code')


def test_failed_commit_is_not_published(tmp_path, monkeypatch):
    import smolvlm_attention.sample_store as module
    store = ShardStore(tmp_path, 'id', [0, 1], (2, 2))
    store.commit()
    original = store.progress_path.read_bytes()
    store.write([0], np.full((1, 2, 2), .1, np.float32))
    def fail(*args, **kwargs):
        raise OSError('injected fsync failure')
    monkeypatch.setattr(module, 'sync_file', fail)
    with pytest.raises(OSError):
        store.commit()
    assert store.progress_path.read_bytes() == original
    assert ShardStore(tmp_path, 'id', [0, 1], (2, 2)).completed == set()


def test_measurement_error_commits_successful_batches(tmp_path):
    from smolvlm_attention.sample_worker import process_pending
    store = ShardStore(tmp_path, 'id', list(range(5)), (2, 2))
    store.batch_size = 2
    def measure(batch):
        if batch[0]['row_index'] >= 2:
            raise RuntimeError('measurement failed')
        return np.full((len(batch), 2, 2), .1, np.float32)
    with pytest.raises(RuntimeError, match='measurement failed'):
        process_pending(rows(), store, measure)
    assert ShardStore(tmp_path, 'id', list(range(5)), (2, 2)).completed == {0, 1}


def test_interrupted_finalization_retries_without_torch(tmp_path, monkeypatch):
    import smolvlm_attention.sample_store as module
    path = make_run(tmp_path, shards=1)
    run = read_run(path)
    store = ShardStore(tmp_path / 'shards' / '0', run['fingerprint'], list(range(5)), (2, 2))
    store.write(list(range(5)), np.full((5, 2, 2), .1, np.float32))
    store.commit()
    original = module.atomic_json
    def fail_manifest(path, value):
        if Path(path).name == 'manifest.json':
            raise OSError('power loss')
        return original(path, value)
    monkeypatch.setattr(module, 'atomic_json', fail_manifest)
    with pytest.raises(OSError, match='power loss'):
        module.finalize_export(path, progress=False)
    assert not (tmp_path / 'export' / 'manifest.json').exists()
    monkeypatch.setattr(module, 'atomic_json', original)
    assert open_prompt_attention(finalize_export(path, progress=False)).attention.shape == (5, 2, 2)


def test_mode_and_run_fields_cannot_diverge(tmp_path):
    from smolvlm_attention.sample_runtime import resolve_run_mode
    assert resolve_run_mode(tmp_path) == 'measure'
    with pytest.raises(FileNotFoundError):
        resolve_run_mode(tmp_path, 'resume')
    path = make_run(tmp_path)
    assert resolve_run_mode(tmp_path) == 'resume'
    with pytest.raises(ValueError, match='all shards'):
        resolve_run_mode(tmp_path, 'export')
    run = read_run(path)
    run['shard_count'] = 1
    atomic_json(path, run)
    with pytest.raises(ValueError, match='identity'):
        read_run(path)


def test_new_worker_guard_registry_and_pid(tmp_path, monkeypatch):
    from scripts import kaggle_source_setup as helper
    from types import ModuleType
    module = ModuleType('smolvlm_attention.sample_runtime')
    class Live:
        def poll(self):
            return None
    module._PROCESSES = {'live': Live()}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(RuntimeError, match='worker'):
        helper.ensure_no_workers([])
    module._PROCESSES.clear()
    atomic_json(tmp_path / 'logs' / 'samples_0.pid.json', {'pid': __import__('os').getpid()})
    # os.kill(pid, 0) is not a safe liveness probe on Windows.
    monkeypatch.setattr(helper.os, 'kill', lambda pid, signal: None)
    with pytest.raises(RuntimeError, match='PID'):
        helper.ensure_no_workers([tmp_path])


def test_preflight_native_and_batch_parity_on_cpu():
    from test_attention_collector import fixture
    from smolvlm_attention.sample_worker import preflight
    model, batch, items, collator = fixture(questions=('What?', 'What is visible on the road?'))
    selected = [dict(row_index=i, input_ids=list(range(1+i))) for i in range(2)]
    preflight(model, selected, lambda selected: collator([items[row['row_index']] for row in selected]))


def test_runtime_import_export_without_torch():
    code = 'import sys; import smolvlm_attention.sample_runtime; assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', code], check=True)


def test_run_initialization_is_locked(tmp_path):
    from filelock import FileLock, Timeout
    with FileLock(str(tmp_path / 'prepare.lock'), timeout=0):
        with pytest.raises(Timeout):
            make_run(tmp_path)
    assert not (tmp_path / 'sample_run.json').exists()


def test_worker_and_storage_semantics_are_fingerprinted():
    from smolvlm_attention.sample_inputs import measurement_sources
    sources = measurement_sources()
    assert 'smolvlm_attention/sample_worker.py' in sources
    assert 'smolvlm_attention/sample_store.py:ShardStore.write' in sources


def test_launch_duplicate_guard_lock_and_wait_exit_race(tmp_path, monkeypatch):
    import smolvlm_attention.sample_runtime as runtime
    import torch
    from filelock import FileLock, Timeout
    path = make_run(tmp_path)
    run = read_run(path)
    monkeypatch.setattr(runtime, 'assert_measurement_compatible', lambda run: None)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(runtime, '_pid_alive', lambda path: False)
    monkeypatch.setattr(runtime, '_PROCESSES', {})
    spawned = []
    class Process:
        def __init__(self, *args, **kwargs):
            self.pid = 5000 + len(spawned)
            self.code = None
            spawned.append(self)
        def poll(self):
            return self.code
    monkeypatch.setattr(runtime.subprocess, 'Popen', Process)
    logs = tmp_path / 'logs'
    logs.mkdir()
    with FileLock(str(logs / 'launch.lock'), timeout=0):
        with pytest.raises(Timeout):
            runtime.launch_workers(path)
    runtime.launch_workers(path)
    runtime.launch_workers(path)
    assert len(spawned) == 2
    for s in range(2):
        start, end = shard_bounds(5, 2, s)
        atomic_json(logs / f'samples_{s}.status.json', dict(status='complete', stage='complete',
            processed=end-start, committed=end-start, batch_size=32, fingerprint=run['fingerprint']))
    # Complete status can be published before the process exits. Wait must poll again.
    def exit_workers(seconds):
        for p in spawned:
            p.code = 0
    monkeypatch.setattr(runtime.time, 'sleep', exit_workers)
    runtime.wait_workers(path, poll_seconds=0)


def test_preflight_bounds_only_oracle_not_longest_collection(monkeypatch):
    import torch
    from smolvlm_attention.sample_worker import preflight
    import smolvlm_attention.collector as collector
    rows = [dict(row_index=i, input_ids=list(range(n))) for i, n in enumerate((400, 600, 2000))]
    oracle, optimized = [], []
    def values(batch):
        return torch.full((len(batch), 2, 2), .1)
    def collect(model, batch):
        optimized.extend(r['row_index'] for r in batch)
        return values(batch)
    def reference(model, batch):
        assert all(len(r['input_ids']) <= 1024 for r in batch)
        oracle.extend(r['row_index'] for r in batch)
        return values(batch)
    monkeypatch.setattr(collector, 'collect_last_query_attention', collect)
    monkeypatch.setattr(collector, 'reference_last_query_attention', reference)
    preflight(None, rows, lambda selected: selected)
    assert oracle == [0, 1] and 2 in optimized


def test_compatible_resume_refreshes_only_idle_operational_paths(tmp_path, monkeypatch):
    import smolvlm_attention.sample_store as module
    config = dict(identity={'policy': 'test'}, shard_count=1, batch_size=32, geometry=[2, 2],
                  artifacts={'checkpoint': 'old'}, cache_descriptor='old-receipt', source_commit='original')
    path = write_run(tmp_path, config, rows())
    old = read_run(path)
    inputs = (tmp_path / 'samples.jsonl').read_bytes()
    fresh = dict(config, artifacts={'checkpoint': 'new'}, cache_descriptor='new-receipt', source_commit='new')
    atomic_json(tmp_path / 'logs' / 'samples_0.pid.json', {'pid': 123})
    monkeypatch.setattr(module, 'pid_is_alive', lambda pid: True)
    with pytest.raises(RuntimeError, match='running'):
        write_run(tmp_path, fresh, rows())
    assert read_run(path) == old
    monkeypatch.setattr(module, 'pid_is_alive', lambda pid: False)
    write_run(tmp_path, fresh, rows())
    updated = read_run(path)
    assert updated['artifacts'] == fresh['artifacts'] and updated['cache_descriptor'] == 'new-receipt'
    assert updated['source_commit'] == 'original' and updated['fingerprint'] == old['fingerprint']
    assert (tmp_path / 'samples.jsonl').read_bytes() == inputs


def test_completed_shard_recovers_stale_failure_status(tmp_path, monkeypatch):
    import torch
    import smolvlm_attention.sample_runtime as runtime
    path = make_run(tmp_path)
    run = read_run(path)
    store = ShardStore(tmp_path / 'shards' / '0', run['fingerprint'], [0, 1, 2], (2, 2))
    store.write([0, 1, 2], np.full((3, 2, 2), .1, np.float32))
    store.commit(batch_size=8)
    logs = tmp_path / 'logs'
    atomic_json(logs / 'samples_0.status.json', {'status': 'failed'})
    monkeypatch.setattr(runtime, 'assert_measurement_compatible', lambda run: None)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(runtime, '_pid_alive', lambda path: False)
    class Dead:
        def poll(self):
            return 1
    monkeypatch.setattr(runtime, '_PROCESSES', {(str(path), 0): Dead()})
    class Process:
        pid = 42
    monkeypatch.setattr(runtime.subprocess, 'Popen', lambda *a, **kw: Process())
    runtime.launch_workers(path)
    status = json.loads((logs / 'samples_0.status.json').read_text())
    assert status['status'] == 'complete' and status['committed'] == 3 and status['batch_size'] == 8
    assert (str(path), 0) not in runtime._PROCESSES


def test_export_identity_covers_actual_sample_mapping(tmp_path):
    from smolvlm_attention.sample_store import digest
    path = make_run(tmp_path, shards=1)
    run = read_run(path)
    store = ShardStore(tmp_path / 'shards' / '0', run['fingerprint'], list(range(5)), (2, 2))
    store.write(list(range(5)), np.full((5, 2, 2), .1, np.float32))
    store.commit()
    export = finalize_export(path, progress=False)
    sample_path = export / 'samples.jsonl'
    sample_path.write_bytes(sample_path.read_bytes().replace(b'Question 0', b'Question X'))
    manifest = json.loads((export / 'manifest.json').read_text())
    manifest['sha256']['samples.jsonl'] = digest(sample_path)
    atomic_json(export / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='identity'):
        open_prompt_attention(export)


def test_prepare_inputs_preserves_source_and_checks_receipt(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from test_smolvlm_cache import fixture_cache
    from test_smolvlm_model import tiny_components
    import smolvlm_attention.data as data
    import smolvlm_attention.config as cfg
    import smolvlm_inference.model as model_module
    from smolvlm_inference.cache import records, record_key, canonical, digest
    from smolvlm_attention.sample_inputs import prepare_inputs
    from transformers import AutoConfig, AutoProcessor
    cache, prepared, identity, _ = fixture_cache(tmp_path)
    # Keep real cache validation and prompt tokenization, replace only pinned model download.
    _, processor = tiny_components()
    manifest_path = cache / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    # Tokenize real 384-image-token prompts while keeping the unused payload tiny.
    manifest['geometry']['visual_tokens_per_view'] = 64
    manifest['model_id'] = model_module.MODEL_ID
    manifest['revision'] = model_module.MODEL_REVISION
    for split in ('train', 'val', 'test'):
        source_path = prepared / f'{split}.jsonl'
        row = json.loads(source_path.read_text())
        row['split'] = split
        source_path.write_text(json.dumps(row) + '\n')
    manifest['prepared_hashes'] = {p.name: digest(p) for p in prepared.iterdir()}
    manifest.pop('fingerprint')
    manifest['fingerprint'] = __import__('hashlib').sha256(canonical(manifest)).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(cfg, 'TEST_COUNT', 1)
    monkeypatch.setattr(cfg, 'CACHE_FINGERPRINT', manifest['fingerprint'])
    monkeypatch.setattr(AutoConfig, 'from_pretrained', lambda *a, **kw:
        SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=30, max_position_embeddings=512)))
    monkeypatch.setattr(AutoProcessor, 'from_pretrained', lambda *a, **kw: processor)
    monkeypatch.setattr(model_module, '_configure_processor', lambda *a: None)
    monkeypatch.setattr(model_module, '_geometry', lambda *a: manifest['geometry'])
    monkeypatch.setattr(model_module, 'processor_policy', lambda *a: identity['processor_policy'])
    output = tmp_path / 'output'
    output.mkdir()
    artifacts = dict(cache_dir=str(cache), prepared_dir=str(prepared))
    prepared_rows, receipt, _ = prepare_inputs(artifacts, output)
    source = list(records(prepared / 'test.jsonl'))[0]
    row = prepared_rows[0]
    for key, value in source.items():
        assert row[key] == value
    assert row['feature_key'] == record_key(source)
    assert row['query_position'] == len(row['input_ids']) - 1
    assert row['input_ids'].count(row['image_token_id']) == 384
    original = receipt.read_bytes()
    monkeypatch.setattr(data, 'validate_cache', lambda *a, **k: pytest.fail('must reuse receipt'))
    assert prepare_inputs(artifacts, output)[0] == prepared_rows
    assert receipt.read_bytes() == original
