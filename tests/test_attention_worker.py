import json

import numpy as np
import pytest


def test_oom_retry_does_not_commit_failed_samples():
    import torch
    from smolvlm_attention.worker import process_pending
    from smolvlm_attention.aggregate import AttentionAccumulator
    rows = [{'sample_id': str(i), 'input_ids': [1] * (i + 1)} for i in range(5)]
    accumulator = AttentionAccumulator((2, 3), 'run', [r['sample_id'] for r in rows])
    calls = []
    def measure(batch):
        calls.append(len(batch))
        if len(batch) > 2:
            raise torch.cuda.OutOfMemoryError('test OOM')
        return np.stack([np.full((2, 3), int(row['sample_id']) / 10) for row in batch])
    process_pending(rows, accumulator, measure, batch_size=4, device='cpu')
    assert calls[0] == 4
    assert accumulator.count == 5
    np.testing.assert_allclose(accumulator.mean, .2)
    process_pending(rows, accumulator, measure, batch_size=2, device='cpu')
    assert accumulator.count == 5


def test_non_oom_errors_are_not_silently_skipped():
    from smolvlm_attention.worker import process_pending
    from smolvlm_attention.aggregate import AttentionAccumulator
    accumulator = AttentionAccumulator((1, 1), 'run', ['a'])
    def measure(_):
        raise ValueError('bad sample')
    with pytest.raises(ValueError, match='bad sample'):
        process_pending([{'sample_id': 'a', 'input_ids': [1]}], accumulator,
                        measure, batch_size=1, device='cpu')
    assert accumulator.count == 0


def test_notebook_has_separate_model_and_plot_cells():
    import ast
    from pathlib import Path
    path = Path('notebooks/kaggle_smolvlm_attention_distribution.ipynb')
    nb = json.loads(path.read_text(encoding='utf-8'))
    code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    assert len(code) == 14
    for cell in code:
        ast.parse(cell)
    assert 'launch_worker' in code[3] and '"base"' in code[3]
    assert 'launch_worker' in code[4] and '"reverse"' in code[4]
    for offset, model in [(7, 'base'), (10, 'reverse')]:
        for i, function in enumerate(['plot_token_atlas', 'plot_spatial_atlas', 'plot_layer_histogram']):
            assert function in code[offset + i] and f'"{model}"' in code[offset + i]
    assert 'make_figures_zip' in code[-1]
