import hashlib
import json

import numpy as np
import pytest

from smolvlm_inference.cache import FeatureCache, canonical, digest, record_key


def fixture_cache(tmp_path):
    prepared = tmp_path / 'prepared'
    cache = tmp_path / 'cache'
    prepared.mkdir()
    cache.mkdir()
    order = ['Front', 'Front-Left', 'Front-Right', 'Back', 'Back-Left', 'Back-Right']
    row = dict(question='Where?', answer='Ahead.', sample_id='test-0', eval_id=0,
               camera_paths={c: f'images/{c}.jpg' for c in order})
    (prepared / 'manifest.json').write_text(json.dumps({'counts': {'train': 1, 'val': 1, 'test': 1}}))
    for split in ('train', 'val', 'test'):
        (prepared / f'{split}.jsonl').write_text(json.dumps(row) + '\n')
    geometry = dict(num_views=6, vision_tokens=4, vision_dim=3, visual_tokens_per_view=1, text_dim=8)
    index = {record_key(row): 0}
    (cache / 'index.json').write_bytes(canonical(index))
    values = np.broadcast_to(np.arange(6, dtype=np.float16)[:, None, None], (6, 4, 3)).copy()
    values.tofile(cache / 'features.float16.memmap')
    identity = dict(backend='smolvlm_no_pruner', version=1, model_id='fixture', revision='fixture',
                    processor_policy={'fixture': True}, geometry=geometry, storage_dtype='float16',
                    compute_dtype='float32', camera_order=order)
    manifest = dict(identity, prepared_hashes={p.name: digest(p) for p in prepared.iterdir()},
                    shape=[1, 6, 4, 3], num_samples=1, record_counts={'train': 1, 'val': 1, 'test': 1},
                    index_sha256=digest(cache / 'index.json'))
    manifest['fingerprint'] = hashlib.sha256(canonical(manifest)).hexdigest()
    (cache / 'manifest.json').write_text(json.dumps(manifest))
    return cache, prepared, identity, values


def test_cache_key_camera_order_and_owned_single_sample(tmp_path):
    cache, prepared, identity, values = fixture_cache(tmp_path)
    reader = FeatureCache(cache, prepared, identity)
    sample = reader.sample('test', 0)
    np.testing.assert_array_equal(sample['visual_features'], values)
    assert sample['sample_id'] == 'test-0'
    assert sample['visual_features'].flags.owndata
    with pytest.raises(IndexError):
        reader.sample('test', -1)


@pytest.mark.parametrize('damage', ['size', 'identity', 'prepared', 'index', 'nan'])
def test_invalid_cache_rejected(tmp_path, damage):
    cache, prepared, identity, _ = fixture_cache(tmp_path)
    if damage == 'size':
        (cache / 'features.float16.memmap').write_bytes(b'0')
    elif damage == 'identity':
        identity['revision'] = 'wrong'
    elif damage == 'prepared':
        (prepared / 'test.jsonl').write_text('{}\n')
    elif damage == 'index':
        (cache / 'index.json').write_text('{}')
    else:
        array = np.memmap(cache / 'features.float16.memmap', dtype=np.float16, mode='r+')
        array[0] = np.nan
        array.flush()
        del array
    with pytest.raises(ValueError):
        FeatureCache(cache, prepared, identity)


def test_record_key_rejects_unsafe_paths():
    with pytest.raises(ValueError):
        record_key({'camera_paths': {'Front': '../oops'}})
