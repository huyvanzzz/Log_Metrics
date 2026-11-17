import json
from pathlib import Path

import pytest
import torch
import numpy as np


def test_process_batches_resume_and_oom(tmp_path):
    from smolvlm_attention.generated_runtime import process_batches
    rows = [dict(sample_id=str(i), input_ids=[1] * (i + 1)) for i in range(5)]
    saved, events, calls = {}, [], []
    def measure(batch):
        calls.append(len(batch))
        if len(batch) > 2:
            raise torch.cuda.OutOfMemoryError()
        return [dict(sample_id=r['sample_id']) for r in batch]
    def save(record):
        assert record['sample_id'] not in saved
        saved[record['sample_id']] = record
    process_batches(rows, set(), measure, save, batch_size=4, progress=events.append)
    assert len(saved) == 5 and calls == [4, 2, 2, 1]
    assert events[-1]['processed'] == 5
    process_batches(rows, set(saved), measure, save, batch_size=2)
    assert calls == [4, 2, 2, 1]


def test_non_oom_is_not_swallowed():
    from smolvlm_attention.generated_runtime import process_batches
    def fail(batch):
        raise ValueError('bad data')
    with pytest.raises(ValueError, match='bad data'):
        process_batches([dict(sample_id='a', input_ids=[1])], set(), fail, lambda r: None, batch_size=1)


def test_generation_config_is_greedy_without_inherited_penalties():
    from transformers import GenerationConfig
    from smolvlm_attention.generated_runtime import generation_config
    native = type('Model', (), {'generation_config': GenerationConfig(
        eos_token_id=[2, 3], bos_token_id=1, pad_token_id=2, repetition_penalty=1.8,
        num_beams=5, do_sample=True, min_new_tokens=15)})()
    config = generation_config(native, 32)
    assert config.max_new_tokens == 32 and config.num_beams == 1
    assert not config.do_sample and config.repetition_penalty == 1
    assert config.min_new_tokens is None and config.use_cache
    assert config.eos_token_id == [2, 3]


def test_measure_moves_inputs_to_model_device(monkeypatch):
    from types import SimpleNamespace
    from transformers import GenerationConfig
    from smolvlm_attention import generated_collector
    from smolvlm_attention.generated_runtime import measure_generated_batch
    # A meta parameter tests device routing without requiring a local CUDA GPU.
    native = torch.nn.Linear(1, 1, device='meta')
    native.generation_config = GenerationConfig(eos_token_id=2, bos_token_id=1, pad_token_id=2)
    def generate(**inputs):
        for key in ('input_ids', 'attention_mask', 'visual_features'):
            assert inputs[key].device.type == 'meta'
        return torch.tensor([[2]])
    def collect(model, batch, content):
        assert batch['input_ids'].device.type == 'meta'
        return dict(visual=torch.zeros(1,30,384), groups=torch.zeros(1,30,3), token_counts=[0])
    monkeypatch.setattr(generated_collector, 'collect_generated_attention', collect)
    model = SimpleNamespace(native=native, generate_from_features=generate,
        processor=SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda *a,**k: '')))
    batch = dict(input_ids=torch.ones(1,1,dtype=torch.long),attention_mask=torch.ones(1,1,dtype=torch.long),
                 visual_features=torch.zeros(1,6,1,1))
    assert measure_generated_batch(model,batch,[dict(sample_id='a')])[0]['token_count'] == 0


def test_cache_validation_progress_and_receipt_reuse(tmp_path):
    from test_smolvlm_cache import fixture_cache
    from smolvlm_attention.data import validate_cache, open_validated_cache
    cache, prepared, _, _ = fixture_cache(tmp_path)
    events = []
    receipt = validate_cache(cache, prepared, tmp_path / 'out', progress=events.append)
    assert open_validated_cache(receipt).counts['test'] == 1
    assert any(e['stage'] == 'features' and e['completed'] == e['total'] for e in events)


def test_real_generation_to_replay_with_unequal_prompts():
    from test_attention_collector import fixture
    from smolvlm_attention.generated_runtime import measure_generated_batch
    from smolvlm_attention.generated_collector import reference_generated_attention
    model, batch, _, _ = fixture(questions=('What?', 'What is visible?'))
    # Suppress reserved image generation via the model's logits, not sampling settings.
    with torch.no_grad():
        model.native.lm_head.weight.zero_()
    rows = [dict(sample_id='a'), dict(sample_id='b')]
    result = measure_generated_batch(model, batch, rows, max_new_tokens=3)
    assert len(result) == 2
    for row in result:
        assert len(row['generated_ids']) <= 3
        assert row['token_count'] == len(row['content_ids'])
    reference = reference_generated_attention(model, batch, [r['content_ids'] for r in result])
    for i, row in enumerate(result):
        torch.testing.assert_close(torch.as_tensor(row['visual']), reference['visual'][i])
        torch.testing.assert_close(torch.as_tensor(row['groups']), reference['groups'][i])


def test_cache_receipt_reuse_skips_feature_scan(tmp_path, monkeypatch):
    from test_smolvlm_cache import fixture_cache
    from smolvlm_attention import generated_runtime as runtime
    from smolvlm_attention.data import validate_cache
    cache, prepared, _, _ = fixture_cache(tmp_path)
    receipt = validate_cache(cache, prepared, tmp_path / 'out')
    fingerprint = json.loads((cache / 'manifest.json').read_text())['fingerprint']
    monkeypatch.setattr(runtime, 'CACHE_FINGERPRINT', fingerprint)
    def forbidden(*args, **kwargs):
        raise AssertionError('full scan must not run')
    monkeypatch.setattr(runtime, 'validate_cache', forbidden)
    assert runtime._cache_receipt(dict(output_root=str(tmp_path / 'out'), cache_dir=str(cache),
        prepared_dir=str(prepared)), receipt) == receipt.resolve()


def _run_fixture(tmp_path):
    import hashlib
    from smolvlm_inference.cache import canonical, digest
    selected = tmp_path / 'selected_samples.json'
    selected.write_text(json.dumps(dict(records=[dict(sample_id='a', question='Q1', eval_id=1),
                                                  dict(sample_id='b', question='Q2', eval_id=2)])))
    identity = dict(selected_sha256=digest(selected), sample_count=2)
    fingerprint = hashlib.sha256(canonical(identity)).hexdigest()
    run = dict(output_root=str(tmp_path), identity=identity, fingerprint=fingerprint,
               selected_samples=str(selected), shard_count=2)
    config = tmp_path / 'generated_run.json'
    config.write_text(json.dumps(run))
    return config, run


def test_merge_shards_preserves_samples_and_rejects_incomplete(tmp_path):
    from smolvlm_attention.generated_runtime import load_generated_results
    from smolvlm_attention.generated_state import save_sample
    config, run = _run_fixture(tmp_path)
    logs = tmp_path / 'logs'
    logs.mkdir()
    for i, sample_id in enumerate(('a', 'b')):
        groups = np.tile([.2 + .2*i, .6 - .2*i, .2], (30, 1))
        record = dict(sample_id=sample_id, visual=np.full((30,384), groups[0,0]/384),
            groups=groups, token_count=i+1, generated_ids=[5]*(i+1), content_ids=[5]*(i+1),
            response='answer', stop_reason='length')
        save_sample(tmp_path / 'data' / f'shard_{i}', record, run['fingerprint'])
        (logs / f'shard_{i}.json').write_text(json.dumps(dict(status='complete', fingerprint=run['fingerprint'])))
    result, metadata = load_generated_results(config)
    np.testing.assert_allclose(result['mean_groups'][0], [.3,.5,.2])
    assert metadata['valid_count'] == 2 and (tmp_path / 'responses.jsonl').is_file()
    (logs / 'shard_1.json').write_text(json.dumps(dict(status='running',fingerprint=run['fingerprint'])))
    with pytest.raises(ValueError, match='incomplete'):
        load_generated_results(config)


def test_launch_is_nonblocking_and_reuses_processes(tmp_path, monkeypatch):
    from smolvlm_attention import generated_runtime as runtime
    config, _ = _run_fixture(tmp_path)
    calls = []
    class Process:
        pid = 999999999
        def poll(self):
            return None
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return Process()
    monkeypatch.setattr(runtime.subprocess, 'Popen', popen)
    monkeypatch.setattr(runtime.torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(runtime, '_PROCESSES', {})
    first = runtime.launch_generated_workers(config)
    assert runtime.launch_generated_workers(config) == first
    assert len(calls) == 2
    assert [call[1]['env']['CUDA_VISIBLE_DEVICES'] for call in calls] == ['0','1']


def test_generated_notebook_is_standalone_and_each_plot_has_own_cell():
    import ast
    import base64
    import hashlib
    import io
    import zipfile
    nb = json.loads(Path('notebooks/kaggle_smolvlm_generated_attention.ipynb').read_text(encoding='utf-8'))
    cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    assert len([c for c in cells if 'fraction_files = plot_attention_fractions(' in c]) == 1
    assert len([c for c in cells if 'histogram_files = plot_generated_histogram(' in c]) == 1
    for cell in cells:
        ast.parse(cell)
    values = {s.targets[0].id:s.value.value for s in ast.parse(cells[0]).body
              if isinstance(s,ast.Assign) and isinstance(s.targets[0],ast.Name) and isinstance(s.value,ast.Constant)}
    payload = base64.b64decode(values['SOURCE_B64'])
    assert hashlib.sha256(payload).hexdigest() == values['SOURCE_ARCHIVE_SHA256']
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        expected = {p.as_posix() for package in ('smolvlm_inference','smolvlm_attention')
                    for p in Path(package).iterdir() if p.suffix in ('.py','.txt')}
        assert set(archive.namelist()) == expected
        for name in expected:
            assert archive.read(name) == Path(name).read_bytes()
    assert all(c['execution_count'] is None and not c['outputs'] for c in nb['cells'] if c['cell_type']=='code')
