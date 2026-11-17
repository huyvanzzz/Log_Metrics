import numpy as np
import pytest

from smolvlm_attention.generated_state import (
    select_samples, trim_generation, save_sample, load_samples, aggregate_samples,
)


def record(sample_id, mass=.3, count=1):
    return dict(sample_id=sample_id, visual=np.full((2, 4), mass / 4) if count else np.zeros((2, 4)),
                groups=np.tile([mass, .5, .5-mass], (2, 1)) if count else np.zeros((2, 3)),
                generated_ids=[7] * count + [2], content_ids=[7] * count,
                response='answer' if count else '', stop_reason='eos', token_count=count)


def test_selection_is_order_independent_unique_and_seeded():
    rows = [dict(sample_id=str(i)) for i in range(12)]
    chosen = select_samples(rows, count=5, seed=42)
    assert chosen == select_samples(rows[::-1], count=5, seed=42)
    assert len({row['sample_id'] for row in chosen}) == 5
    with pytest.raises(ValueError):
        select_samples(rows + [rows[0]], count=5)


def test_trim_eos_excludes_eos_and_retains_pad_before_eos():
    content, reasons = trim_generation([[0, 8, 2, 0], [4, 5, 6, 7], [2, 0, 0, 0]], 2)
    assert content == [[0, 8], [4, 5, 6, 7], []]
    assert reasons == ['eos', 'length', 'eos']


def test_samples_resume_and_equal_weight_aggregation(tmp_path):
    records = [record('a', .1, 1), record('b', .5, 10), record('empty', count=0)]
    for item in records:
        save_sample(tmp_path, item, 'fp', shape=(2, 4))
    loaded = load_samples(tmp_path, ['a', 'b', 'empty'], 'fp', shape=(2, 4))
    result = aggregate_samples(loaded, ['a', 'b', 'empty'])
    np.testing.assert_allclose(result['mean_visual'], .3 / 4)
    np.testing.assert_allclose(result['mean_groups'][0], [.3, .5, .2])
    np.testing.assert_allclose(result['std_groups'][:, 0], np.std([.1, .5], ddof=1))
    assert (result['valid_count'], result['empty_count'], result['processed_count']) == (2, 1, 3)
    with pytest.raises(ValueError, match='fingerprint'):
        load_samples(tmp_path, ['a', 'b', 'empty'], 'wrong', shape=(2, 4))
    with pytest.raises(ValueError):
        load_samples(tmp_path, ['a'], 'fp', shape=(2, 4))


def test_reject_bad_probabilities_and_incomplete_or_empty_aggregation(tmp_path):
    bad = record('a')
    bad['visual'][0, 0] = np.nan
    with pytest.raises(ValueError):
        save_sample(tmp_path, bad, 'fp', shape=(2, 4))
    bad = record('a')
    bad['groups'][0, 0] += .1
    with pytest.raises(ValueError):
        save_sample(tmp_path, bad, 'fp', shape=(2, 4))
    with pytest.raises(ValueError):
        aggregate_samples({'a': record('a')}, ['a', 'missing'])
    with pytest.raises(ValueError):
        aggregate_samples({'a': record('a', count=0)}, ['a'])


def test_resume_detects_array_tampering(tmp_path):
    path = save_sample(tmp_path, record('a'), 'fp', shape=(2, 4))
    with np.load(path, allow_pickle=False) as data:
        saved = {key: data[key] for key in data.files}
    saved['visual'][0, 0] = -.1
    np.savez(path, **saved)
    with pytest.raises(ValueError):
        load_samples(tmp_path, ['a'], 'fp', shape=(2, 4))
