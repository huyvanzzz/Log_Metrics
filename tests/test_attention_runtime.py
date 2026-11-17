import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest


def test_notebook_bundles_exact_current_source():
    nb = json.loads(Path('notebooks/kaggle_smolvlm_attention_distribution.ipynb').read_text(encoding='utf-8'))
    code = next(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    assignments = {}
    for statement in ast.parse(code).body:
        if (isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name)
                and isinstance(statement.value, ast.Constant)):
            assignments[statement.targets[0].id] = statement.value.value
    payload = base64.b64decode(assignments['SOURCE_B64'])
    assert hashlib.sha256(payload).hexdigest() == assignments['SOURCE_ARCHIVE_SHA256']
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        expected = {p.as_posix() for package in ['smolvlm_inference', 'smolvlm_attention']
                    for p in Path(package).iterdir() if p.suffix in ('.py', '.txt')}
        assert set(archive.namelist()) == expected
        for name in archive.namelist():
            assert archive.read(name) == Path(name).read_bytes(), f'Rebuild notebook bundle after editing {name}'


def test_launch_is_nonblocking_and_duplicate_launch_reuses_worker(tmp_path, monkeypatch):
    from smolvlm_attention import runtime
    config = tmp_path / 'run.json'
    config.write_text(json.dumps(dict(output_root=str(tmp_path), models={'base': {'gpu': 0}})))
    calls = []
    class Process:
        pid = 123456789
        def poll(self):
            return None
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return Process()
    monkeypatch.setattr(runtime.subprocess, 'Popen', popen)
    monkeypatch.setattr(runtime, '_PROCESSES', {})
    first = runtime.launch_worker(config, 'base')
    assert runtime.launch_worker(config, 'base') is first
    assert len(calls) == 1
    assert calls[0][1]['env']['CUDA_VISIBLE_DEVICES'] == '0'
    assert calls[0][0][-1] == 'base'
    assert 'smolvlm_attention.worker' in calls[0][0]


def test_results_require_matching_source_and_full_coverage(tmp_path):
    from smolvlm_attention.aggregate import AttentionAccumulator
    from smolvlm_attention.runtime import load_results
    from smolvlm_attention.config import QUERY_POLICY
    from smolvlm_inference.cache import canonical, digest
    models = {}
    for name in ['base', 'reverse']:
        folder = tmp_path / 'data' / name
        folder.mkdir(parents=True)
        prepared = folder / 'prepared_test.json'
        prepared.write_text(json.dumps(dict(version=1, manifest_fingerprint='cache',
            records=[dict(sample_id='test-0'), dict(sample_id='test-1')])) )
        spec = dict(repo=name, revision='pinned', camera_order=list(range(6)), prepared_test=str(prepared), gpu=0)
        models[name] = spec
        identity = dict(algorithm=QUERY_POLICY, model=name, revision='pinned',
            checkpoint_sha256='checkpoint', order=spec['camera_order'], cache='cache',
            prepared_sha256=digest(prepared), precision='float32', transformers='4.46.3',
            source_archive_sha256='source')
        fingerprint = hashlib.sha256(canonical(identity)).hexdigest()
        accumulator = AttentionAccumulator((30, 384), fingerprint, ['test-0', 'test-1'])
        accumulator.add(['test-0', 'test-1'], np.full((2, 30, 384), .001))
        accumulator.save(folder / 'attention.npz')
        (folder / 'metadata.json').write_text(json.dumps(dict(identity, status='complete', count=2, fingerprint=fingerprint)))
    config = tmp_path / 'run.json'
    run = dict(output_root=str(tmp_path), models=models, expected_test_count=2, source_archive_sha256='source')
    config.write_text(json.dumps(run))
    results, metadata, scales, _ = load_results(config)
    assert set(results) == {'base', 'reverse'}
    np.testing.assert_allclose(results['base'], .001)
    run['source_archive_sha256'] = 'different_source'
    config.write_text(json.dumps(run))
    with pytest.raises(ValueError, match='does not match'):
        load_results(config)
