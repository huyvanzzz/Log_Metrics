import numpy as np
import pytest

from smolvlm_attention.aggregate import AttentionAccumulator


def test_weighted_resume_and_atomic_rejection(tmp_path):
    acc = AttentionAccumulator((2, 3), 'fp', ['a', 'b', 'c'])
    acc.add(['a', 'b'], np.stack([np.ones((2, 3)), np.full((2, 3), 3)]))
    with pytest.raises(ValueError):
        acc.add(['c', 'a'], np.zeros((2, 2, 3)))
    assert acc.count == 2
    assert acc.pending_ids == ['c']
    path = tmp_path / 'state.npz'
    acc.save(path)
    acc = AttentionAccumulator.load(path, (2, 3), 'fp', ['a', 'b', 'c'])
    acc.add(['c'], np.full((1, 2, 3), 8))
    np.testing.assert_allclose(acc.mean, 4)
    assert acc.sum.dtype == np.float64
    with pytest.raises(ValueError):
        AttentionAccumulator.load(path, (2, 3), 'other', ['a', 'b', 'c'])


def test_nonfinite_and_unknown_are_atomic():
    acc = AttentionAccumulator((1, 2), 'fp', ['a'])
    for ids, values in [(['x'], [[[1, 2]]]), (['a'], [[[1, np.nan]]])]:
        with pytest.raises(ValueError):
            acc.add(ids, values)
        assert acc.count == 0


def test_damaged_checkpoint_rejected(tmp_path):
    path = tmp_path / 'bad.npz'
    acc = AttentionAccumulator((1, 2), 'fp', ['a', 'b'])
    acc.add(['a'], [[[1, 2]]])
    acc.save(path)
    with np.load(path, allow_pickle=False) as state:
        saved = {key: state[key] for key in state.files}
    saved['processed_ids'] = np.asarray(['a', 'a'])
    np.savez(path, **saved)
    with pytest.raises(ValueError):
        AttentionAccumulator.load(path, (1, 2), 'fp', ['a', 'b'])


def test_cpu_bfloat16_attention_supported():
    import torch
    acc = AttentionAccumulator((1, 2), 'fp', ['a'])
    acc.add(['a'], torch.tensor([[[1, 2]]], dtype=torch.bfloat16))
    np.testing.assert_array_equal(acc.mean, [[1, 2]])
