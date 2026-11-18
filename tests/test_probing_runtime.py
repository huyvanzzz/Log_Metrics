import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


def test_atomic_state_detects_tampering_and_preserves_incompatible_result(tmp_path):
    from smolvlm_probing.state import save_record, load_record
    record = {'sample_id': 'a', 'status': 'complete', 'score': -1.2}
    path = save_record(tmp_path, 'baseline', record, 'fingerprint', ranking=np.ones((5, 384), np.float32))
    loaded, arrays = load_record(path, 'fingerprint', sample_id='a')
    assert loaded == record
    np.testing.assert_array_equal(arrays['ranking'], 1)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='identity|fingerprint'):
        save_record(tmp_path, 'baseline', record, 'other', ranking=np.zeros((5, 384)))
    assert path.read_bytes() == before
    with np.load(path, allow_pickle=False) as data:
        changed = {key: data[key].copy() for key in data.files}
    changed['ranking'][0, 0] = 2
    np.savez(path, **changed)
    with pytest.raises(ValueError, match='checksum'):
        load_record(path, 'fingerprint')


def test_source_hash_ignores_plotting_but_includes_measurement(tmp_path):
    from smolvlm_probing.state import source_hashes
    for name in ('config.py', 'measurement.py', 'preflight.py', 'localization.py', 'data.py', 'state.py', 'runtime.py', 'worker.py', 'plotting.py'):
        (tmp_path / name).write_text(name)
    first = source_hashes(tmp_path)
    (tmp_path / 'plotting.py').write_text('new plot')
    assert source_hashes(tmp_path) == first
    (tmp_path / 'measurement.py').write_text('new measurement')
    assert source_hashes(tmp_path) != first


def test_batch_oom_retry_and_completed_skip():
    import torch
    from smolvlm_probing.runtime import process_pending
    rows = [{'sample_id': str(i)} for i in range(6)]
    saved, events, calls = [], [], []
    def measure(batch):
        calls.append(len(batch))
        if len(batch) > 2:
            raise torch.cuda.OutOfMemoryError('fixture')
        return [dict(row, status='complete') for row in batch]
    size = process_pending(rows, {'0'}, measure, saved.append, batch_size=16, progress=events.append)
    assert size == 2 and [r['sample_id'] for r in saved] == ['1', '2', '3', '4', '5']
    assert calls[:3] == [5, 5, 4]
    assert events[-1]['processed'] == 6


def test_frozen_dataset_all_ids_and_targets_validate():
    from smolvlm_probing.data import load_dataset
    dataset = load_dataset()
    assert len(dataset['rows']) == 2581
    assert sum(len(r.get('localization_targets', {}).get('with_coordinates', [])) for r in dataset['rows']) == 749
    assert sum(len(r.get('localization_targets', {}).get('without_coordinates', [])) for r in dataset['rows']) == 652
    assert len({r['sample_id'] for r in dataset['rows']}) == 2581


def test_plot_mode_never_resolves_artifacts(tmp_path, monkeypatch):
    from smolvlm_probing import runtime
    def forbidden(*args, **kwargs):
        raise AssertionError('plot must not resolve model or cache')
    monkeypatch.setattr(runtime, 'resolve_artifacts', forbidden)
    with pytest.raises(FileNotFoundError):
        runtime.prepare_run(tmp_path, mode='plot')


def test_existing_incompatible_config_is_not_overwritten(tmp_path):
    from smolvlm_probing.runtime import write_run
    config = {'version': 1, 'identity': {'generation': 'one'}, 'output_root': str(tmp_path)}
    path = write_run(tmp_path, config, [{'sample_id': 'a'}], mode='measure')
    before = path.read_bytes()
    with pytest.raises(ValueError, match='new output|incompatible|exists'):
        write_run(tmp_path, dict(config, identity={'generation': 'two'}), [{'sample_id': 'b'}], mode='resume')
    assert path.read_bytes() == before
    assert json.loads((tmp_path / 'inputs.json').read_text()) == [{'sample_id': 'a'}]


def test_trim_keeps_middle_special_and_classifies_reserved_image():
    from types import SimpleNamespace
    from smolvlm_probing.worker import trim_response
    tok = SimpleNamespace(eos_token_id=2, all_special_ids=[0, 1, 2, 3],
        decode=lambda ids, **kwargs: ''.join('word' for token in ids if token not in [0,1,2,3]))
    result = trim_response(tok, [7, 3, 8, 2, 0], image_token_id=1)
    assert result['generated_ids'] == [7, 3, 8] and result['generation_status'] == 'valid'
    assert trim_response(tok, [7, 1, 2], 1)['generation_status'] == 'invalid_multimodal_generation'
    assert trim_response(tok, [2, 0], 1)['generation_status'] == 'empty'
    native = trim_response(tok, [7, 9, 0], 1, eos_token_id=[9, 10])
    assert native['generated_ids'] == [7] and native['stop_reason'] == 'eos'
    assert result['special_token_mask'] == [False, True, False]


def test_probing_worker_guard_blocks_pull(monkeypatch):
    import importlib.util
    from types import ModuleType
    import sys
    spec = importlib.util.spec_from_file_location('source_helper', 'scripts/kaggle_source_setup.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    module = ModuleType('smolvlm_probing.runtime')
    module._PROCESSES = {0: type('Live', (), {'poll': lambda self: None})()}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(RuntimeError, match='worker'):
        helper.ensure_no_workers([])


def test_launch_serializes_multiple_notebook_kernels(tmp_path, monkeypatch):
    from filelock import FileLock, Timeout
    from smolvlm_probing import runtime
    monkeypatch.setattr(runtime, 'read_run', lambda path: {'output_root': str(tmp_path)})
    (tmp_path / 'logs').mkdir()
    # A second kernel must not overwrite another launcher's status/PID files.
    with FileLock(str(tmp_path / 'logs' / 'launch.lock')):
        with pytest.raises(Timeout):
            runtime.launch_stage(tmp_path / 'run.json', 'baseline')


def test_plot_imports_do_not_load_torch_in_a_fresh_process():
    import subprocess
    import sys
    completed = subprocess.run([sys.executable, '-c',
        'import sys; import smolvlm_probing.runtime; import smolvlm_probing.plotting; '
        'assert "torch" not in sys.modules'], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_frozen_hash_bytes_are_protected_from_git_crlf():
    import subprocess
    result = subprocess.check_output(['git', 'check-attr', 'text', '--',
        'data/drivelm_attention_v1/causal.jsonl', 'scripts/filter_drivelm_attention_data.py'], text=True)
    assert all(line.endswith(': text: unset') for line in result.strip().splitlines())


def test_rejected_resume_cannot_overwrite_the_original_validation_receipt(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from smolvlm_probing import data, runtime
    from smolvlm_probing.state import atomic_json
    from smolvlm_attention import data as cache_data, generated_runtime
    from smolvlm_inference import model
    from transformers import AutoConfig, AutoProcessor
    geometry = {'visual_tokens_per_view': 64}
    monkeypatch.setattr(AutoConfig, 'from_pretrained', lambda *a, **k:
        SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=30)))
    monkeypatch.setattr(AutoProcessor, 'from_pretrained', lambda *a, **k: object())
    monkeypatch.setattr(model, '_configure_processor', lambda *a: None)
    monkeypatch.setattr(model, '_geometry', lambda *a: geometry)
    monkeypatch.setattr(model, 'processor_policy', lambda *a: {'fixed': True})
    monkeypatch.setattr(data, 'load_dataset', lambda *a: {})
    monkeypatch.setattr(data, 'prepare_inputs', lambda *a: [{'sample_id': 'fixture', 'input_ids': [1]}])
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'fixture weights')
    def artifacts(label):
        folder = tmp_path / label
        folder.mkdir()
        (folder/'manifest.json').write_text(json.dumps({'fingerprint': label}))
        (folder/'index.json').write_text('{}')
        (folder/'features.float16.memmap').write_bytes(b'\0\0')
        return dict(model={'repo': 'base', 'revision': 'pinned', 'checkpoint': str(checkpoint)},
            cache_dir=str(folder), prepared_dir=str(tmp_path/'prepared'))
    def receipt(artifacts, old_receipt):
        return atomic_json(Path(artifacts['output_root'])/'validated_cache.json',
            {'cache_dir': artifacts['cache_dir'], 'payload_stat': {'size': 2}})
    def reader(path):
        folder = Path(json.loads(Path(path).read_text())['cache_dir'])
        manifest = json.loads((folder/'manifest.json').read_text())
        return SimpleNamespace(manifest=dict(manifest, processor_policy={'fixed': True},
            geometry=geometry, prepared_hashes={'test': 'same'}))
    monkeypatch.setattr(generated_runtime, '_cache_receipt', receipt)
    monkeypatch.setattr(cache_data, 'open_validated_cache', reader)
    output = tmp_path/'output'
    path = runtime.prepare_run(output, mode='measure', artifacts=artifacts('cache-a'), gpu_count=1)
    original = path.read_bytes()
    run = runtime.read_run(path)
    old_receipt = Path(run['cache_descriptor'])
    original_receipt = old_receipt.read_bytes()
    with pytest.raises(ValueError, match='incompatible'):
        runtime.prepare_run(output, mode='resume', artifacts=artifacts('cache-b'), gpu_count=1)
    assert path.read_bytes() == original
    assert old_receipt.read_bytes() == original_receipt
