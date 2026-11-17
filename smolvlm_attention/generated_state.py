"""Atomic per-question generation results and equal-question statistics."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def select_samples(rows, count=1000, seed=42):
    rows = sorted(list(rows), key=lambda row: row['sample_id'])
    ids = [row['sample_id'] for row in rows]
    if (any(not isinstance(value, str) or not value for value in ids)
            or len(set(ids)) != len(ids) or not 0 < count <= len(rows)):
        raise ValueError('need unique sample IDs and a valid selection count')
    indices = np.random.Generator(np.random.PCG64(seed)).choice(len(rows), count, replace=False)
    return [rows[int(index)] for index in indices]


def trim_generation(sequences, eos_token_id):
    eos = {int(eos_token_id)} if np.isscalar(eos_token_id) else set(map(int, eos_token_id))
    contents, reasons = [], []
    for row in sequences:
        values = [int(value) for value in row]
        end = next((i for i, value in enumerate(values) if value in eos), len(values))
        contents.append(values[:end])
        reasons.append('eos' if end < len(values) else 'length')
    return contents, reasons


def _validate(record, shape):
    if not isinstance(record.get('sample_id'), str) or not record['sample_id']:
        raise ValueError('invalid sample ID')
    visual = np.asarray(record['visual'], dtype=np.float64)
    groups = np.asarray(record['groups'], dtype=np.float64)
    if visual.shape != tuple(shape) or groups.shape != (shape[0], 3):
        raise ValueError('attention shape mismatch')
    for values in (visual, groups):
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1 + 1e-6):
            raise ValueError('invalid attention probabilities')
    count = record.get('token_count')
    if type(count) is not int or count < 0:
        raise ValueError('invalid token count')
    for field in ('generated_ids', 'content_ids'):
        if not isinstance(record.get(field), list) or any(type(value) is not int or value < 0 for value in record[field]):
            raise ValueError('invalid token IDs')
    if (len(record['content_ids']) != count or
            record['generated_ids'][:count] != record['content_ids'] or
            record.get('stop_reason') not in ('eos', 'length') or
            not isinstance(record.get('response'), str)):
        raise ValueError('inconsistent generation metadata')
    if count:
        if not np.allclose(groups.sum(axis=1), 1, atol=2e-5, rtol=0):
            raise ValueError('attention groups must sum to one')
    elif np.any(visual) or np.any(groups):
        raise ValueError('empty generation must have zero attention')
    if not np.allclose(visual.sum(axis=1), groups[:, 0], atol=2e-5, rtol=0):
        raise ValueError('visual sum disagrees with image attention')
    return visual, groups


def _filename(sample_id):
    return hashlib.sha256(sample_id.encode('utf-8')).hexdigest() + '.npz'


def save_sample(root, record, fingerprint, shape=(30, 384)):
    """Atomically save one result; return its hashed-ID NPZ path."""
    visual, groups = _validate(record, shape)
    directory = Path(root) / 'results'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _filename(record['sample_id'])
    metadata = {key: value for key, value in record.items() if key not in ('visual', 'groups')}
    metadata['fingerprint'] = fingerprint
    fd, temporary = tempfile.mkstemp(dir=directory, suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez(stream, visual=visual, groups=groups,
                     metadata=json.dumps(metadata, ensure_ascii=False, sort_keys=True))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def load_samples(root, expected_ids, fingerprint, shape=(30, 384)):
    """Load completed records, refusing mixed runs, unknown IDs or bad values."""
    ids = list(expected_ids)
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate expected sample IDs')
    expected = set(ids)
    result = {}
    for path in sorted((Path(root) / 'results').glob('*.npz')):
        with np.load(path, allow_pickle=False) as saved:
            record = json.loads(saved['metadata'].item())
            record['visual'] = saved['visual'].copy()
            record['groups'] = saved['groups'].copy()
        if record.pop('fingerprint', None) != fingerprint:
            raise ValueError('sample fingerprint mismatch')
        _validate(record, shape)
        sample_id = record['sample_id']
        if sample_id not in expected or sample_id in result or path.name != _filename(sample_id):
            raise ValueError('unknown, duplicate or incorrectly named sample result')
        result[sample_id] = record
    return result


def aggregate_samples(records, expected_ids):
    """Mean samples equally; SD is across sample means (ddof=1)."""
    ids = list(expected_ids)
    if len(set(ids)) != len(ids) or set(records) != set(ids):
        raise ValueError('exact selected sample coverage is required')
    valid = []
    for sample_id in ids:
        record = records[sample_id]
        if record['sample_id'] != sample_id:
            raise ValueError('sample ID mismatch')
        _validate(record, np.asarray(record['visual']).shape)
        if record['token_count']:
            valid.append(record)
    if not valid:
        raise ValueError('no nonempty generated samples')
    visual = np.stack([record['visual'] for record in valid]).astype(np.float64)
    groups = np.stack([record['groups'] for record in valid]).astype(np.float64)
    return dict(mean_visual=visual.mean(axis=0), mean_groups=groups.mean(axis=0),
                std_groups=groups.std(axis=0, ddof=1) if len(valid) > 1 else np.zeros_like(groups[0]),
                valid_count=len(valid), empty_count=len(ids)-len(valid), processed_count=len(ids))
