"""CPU integration of cache identity, prompt order, collection, and resumed means."""
import hashlib
import json

import numpy as np
import pytest
import torch

from test_smolvlm_model import tiny_components
from smolvlm_inference.cache import CAMERA_ORDER, canonical, digest, record_key
from smolvlm_inference.model import CachedSmolVLM, processor_policy
from smolvlm_inference.text import CachedBatchCollator
from smolvlm_attention.aggregate import AttentionAccumulator
from smolvlm_attention.collector import collect_last_query_attention, reference_last_query_attention
from smolvlm_attention.data import (validate_cache, open_validated_cache, prepare_test,
                                   load_prepared_test, make_batch)
from smolvlm_attention.worker import process_pending


@pytest.mark.parametrize('reverse', [False, True], ids=['canonical', 'reverse'])
def test_real_cache_to_resumed_attention_matches_native_reference(tmp_path, reverse):
    native, processor = tiny_components()
    order = list(reversed(CAMERA_ORDER)) if reverse else CAMERA_ORDER
    model = CachedSmolVLM(native, processor, model_id='tiny', model_revision='local',
                          prompt_camera_order=order)
    prepared, cache = tmp_path / 'prepared', tmp_path / 'cache'
    prepared.mkdir()
    cache.mkdir()
    rows = [dict(sample_id=f'test-{i}', eval_id=i, question=question, answer='SECRET GROUND TRUTH',
                 camera_paths={camera: f'images/{camera}.jpg' for camera in CAMERA_ORDER})
            for i, question in enumerate(('What?', 'What is visible? A road. A car.'))]
    counts = dict(train=0, val=0, test=2)
    (prepared / 'manifest.json').write_bytes(canonical(dict(counts=counts)))
    for split in ('train', 'val', 'test'):
        (prepared / f'{split}.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in rows) if split == 'test' else '', encoding='utf-8')
    # Both questions intentionally share one cached six-camera observation.
    index = {record_key(rows[0]): 0}
    (cache / 'index.json').write_bytes(canonical(index))
    features = (np.arange(6 * 16 * 8).reshape(6, 16, 8) / 128).astype(np.float16)
    features.tofile(cache / 'features.float16.memmap')
    manifest = dict(backend='smolvlm_no_pruner', version=1, model_id=model.model_id,
                    revision=model.model_revision, processor_policy=processor_policy(processor),
                    geometry=model.geometry, storage_dtype='float16', compute_dtype='float32',
                    camera_order=CAMERA_ORDER, shape=[1, 6, 16, 8], num_samples=1,
                    record_counts=counts, index_sha256=digest(cache / 'index.json'),
                    prepared_hashes={path.name: digest(path) for path in prepared.iterdir()})
    manifest['fingerprint'] = hashlib.sha256(canonical(manifest)).hexdigest()
    (cache / 'manifest.json').write_bytes(canonical(manifest))

    reader = open_validated_cache(validate_cache(cache, prepared, tmp_path / 'out', manifest['fingerprint']))
    inputs = load_prepared_test(prepare_test(reader, processor, order, tmp_path / 'out', 512))
    selected = inputs['records']
    assert inputs['camera_order'] == order
    assert inputs['manifest_fingerprint'] == manifest['fingerprint']
    assert [row['feature_index'] for row in selected] == [0, 0]
    assert all('answer' not in row for row in selected)
    batch = make_batch(reader, selected, processor, model.prompt_camera_order)
    reference_batch = CachedBatchCollator(processor, image_seq_len=4, max_length=512,
        max_new_tokens=0, prompt_camera_order=order)([
            dict(row, visual_features=torch.from_numpy(features.copy()).float()) for row in rows])
    for key in ('input_ids', 'attention_mask', 'visual_features'):
        torch.testing.assert_close(batch[key], reference_batch[key], rtol=0, atol=0)
    assert (batch['attention_mask'][0] == 0).any()  # Unequal lengths exercise left padding.
    expected = reference_last_query_attention(model, reference_batch).numpy().mean(axis=0, dtype=np.float64)
    torch.testing.assert_close(collect_last_query_attention(model, batch),
                               reference_last_query_attention(model, reference_batch), atol=1e-6, rtol=1e-4)

    ids = [row['sample_id'] for row in selected]
    fingerprint = hashlib.sha256(canonical(dict(cache=manifest['fingerprint'], order=order))).hexdigest()
    accumulator = AttentionAccumulator((2, 24), fingerprint, ids)
    measured = []
    def measure(subset):
        measured.extend(row['sample_id'] for row in subset)
        return collect_last_query_attention(model, make_batch(reader, subset, processor, order))
    checkpoint = tmp_path / 'attention.npz'
    process_pending(selected[:1], accumulator, measure, batch_size=1, device='cpu', checkpoint=checkpoint)
    assert accumulator.count == 1
    resumed = AttentionAccumulator.load(checkpoint, (2, 24), fingerprint, ids)
    process_pending(selected, resumed, measure, batch_size=2, device='cpu', checkpoint=checkpoint)
    completed = AttentionAccumulator.load(checkpoint, (2, 24), fingerprint, ids)
    assert completed.count == 2 and not completed.pending_ids
    assert measured == ids
    np.testing.assert_allclose(completed.mean, expected, atol=1e-6, rtol=1e-4)
    process_pending(selected, completed, measure, batch_size=2, device='cpu', checkpoint=checkpoint)
    assert measured == ids
