import json

import numpy as np
import pytest

from test_smolvlm_cache import fixture_cache
from smolvlm_attention.data import validate_cache, open_validated_cache, make_batch


def test_receipt_readonly_reorder_and_changed_payload(tmp_path):
    cache, prepared, identity, values = fixture_cache(tmp_path)
    receipt = validate_cache(cache, prepared, tmp_path / 'out')
    reader = open_validated_cache(receipt)
    assert not reader.array.flags.writeable
    class Processor:
        class tokenizer:
            pad_token_id = 0
    order = list(reversed(identity['camera_order']))
    batch = make_batch(reader, [{'sample_id': 'x', 'feature_index': 0, 'input_ids': [3, 4]},
                                {'sample_id': 'y', 'feature_index': 0, 'input_ids': [4]}], Processor(), order)
    np.testing.assert_array_equal(batch['input_ids'], [[3, 4], [0, 4]])
    np.testing.assert_array_equal(batch['visual_features'][0], values[::-1].copy())
    assert str(batch['visual_features'].dtype) == 'torch.float32'
    with (cache / 'features.float16.memmap').open('ab') as stream:
        stream.write(b'xx')
    with pytest.raises(ValueError):
        open_validated_cache(receipt)


def test_receipt_rejects_prepared_changes_and_wrong_fingerprint(tmp_path):
    cache, prepared, _, _ = fixture_cache(tmp_path)
    with pytest.raises(ValueError):
        validate_cache(cache, prepared, tmp_path / 'out', expected_fingerprint='wrong')
    receipt = validate_cache(cache, prepared, tmp_path / 'out')
    (prepared / 'train.jsonl').write_text('{}')
    with pytest.raises(ValueError):
        open_validated_cache(receipt)


def test_prepared_test_matches_native_collator_and_has_no_ground_truth(tmp_path):
    from test_smolvlm_model import tiny_components
    from smolvlm_inference.text import CachedBatchCollator
    from smolvlm_attention.data import prepare_test, load_prepared_test
    import torch
    cache, prepared, identity, _ = fixture_cache(tmp_path)
    reader = open_validated_cache(validate_cache(cache, prepared, tmp_path / 'out'))
    _, processor = tiny_components()
    order = identity['camera_order']
    result = load_prepared_test(prepare_test(reader, processor, order, tmp_path / 'out', 512))
    row = result['records'][0]
    assert set(row) == {'sample_id', 'eval_id', 'feature_index', 'input_ids'}
    source = json.loads((prepared / 'test.jsonl').read_text())
    source['visual_features'] = torch.zeros((6, 4, 3))
    reference = CachedBatchCollator(processor, image_seq_len=1, max_length=512,
                                  max_new_tokens=0, prompt_camera_order=order)([source])
    assert row['input_ids'] == reference['input_ids'][0].tolist()
    with pytest.raises(ValueError, match='context'):
        prepare_test(reader, processor, order, tmp_path / 'out', 2)


def test_reopen_skips_full_validation_and_record_parsing(tmp_path, monkeypatch):
    import smolvlm_attention.data as data
    cache, prepared, _, _ = fixture_cache(tmp_path)
    receipt = validate_cache(cache, prepared, tmp_path / 'out')
    def forbidden(*args, **kwargs):
        raise AssertionError('full validation must not run')
    monkeypatch.setattr(data, 'FeatureCache', forbidden)
    monkeypatch.setattr(data, 'records', forbidden)
    monkeypatch.setattr(np, 'isfinite', forbidden)
    assert open_validated_cache(receipt).counts['test'] == 1
