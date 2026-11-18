import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest


@pytest.mark.parametrize('filename', ['kaggle_smolvlm_attention_distribution.ipynb',
                                    'kaggle_smolvlm_generated_attention.ipynb'])
def test_notebook_pulls_entire_source_before_importing_models(filename):
    nb = json.loads((Path('notebooks') / filename).read_text(encoding='utf-8'))
    code = next(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    ast.parse(code)
    assert Path('scripts/kaggle_source_setup.py').read_text(encoding='utf-8').strip() in code
    assert 'SOURCE_B64' not in code
    assert 'SOURCE_BRANCH = "smolvlm-attention"' in code
    assert "'pull', '--ff-only'" in code and "'clone'" in code
    assert code.index('identity = sync_source(') < code.index("'-m', 'pip'")
    assert code.index('activate_source(SOURCE)') < code.index('import torch, transformers')


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
