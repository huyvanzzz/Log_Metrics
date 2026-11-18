"""Checked, atomic per-sample state. Contains no model or CUDA imports."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    payload = canonical(value) + b'\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def source_hashes(directory=None):
    folder = Path(directory) if directory else Path(__file__).parent
    return {name: digest(folder / name) for name in (
        'config.py', 'measurement.py', 'preflight.py', 'localization.py', 'data.py', 'state.py', 'runtime.py', 'worker.py')}


def record_path(root, stage, sample_id):
    if stage not in ('generation', 'baseline', 'causal', 'localization'):
        raise ValueError('Unknown measurement stage')
    return Path(root) / 'state' / stage / (hashlib.sha256(sample_id.encode('utf-8')).hexdigest() + '.npz')


def _checksums(record, arrays):
    hashes = {'record': fingerprint(record)}
    for name, value in arrays.items():
        if name == 'metadata' or not name.isidentifier():
            raise ValueError('Invalid array name')
        value = np.asarray(value)
        if value.dtype.kind not in 'biuf' or not np.isfinite(value).all():
            raise ValueError('State arrays must be finite numeric arrays')
        hashes[name] = fingerprint({'shape': value.shape, 'dtype': value.dtype.str,
            'bytes': hashlib.sha256(value.tobytes(order='C')).hexdigest()})
    return hashes


def load_record(path, identity, *, sample_id=None):
    with np.load(path, allow_pickle=False) as data:
        header = json.loads(data['metadata'].item())
        arrays = {key: data[key].copy() for key in data.files if key != 'metadata'}
    if header['identity'] != identity:
        raise ValueError('Result fingerprint/identity differs')
    record = header['record']
    if sample_id is not None and record['sample_id'] != sample_id:
        raise ValueError('Result sample ID differs')
    if header['checksums'] != _checksums(record, arrays):
        raise ValueError('Result checksum mismatch')
    return record, arrays


def save_record(root, stage, record, identity, **arrays):
    if not isinstance(record.get('sample_id'), str) or not record['sample_id']:
        raise ValueError('Nonempty sample ID required')
    checksums = _checksums(record, arrays)
    path = record_path(root, stage, record['sample_id'])
    if path.exists():
        previous, values = load_record(path, identity, sample_id=record['sample_id'])
        if _checksums(previous, values) != checksums:
            raise ValueError('Existing result differs; use a new output directory')
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, metadata=json.dumps({'identity': identity,
                'record': record, 'checksums': checksums}, ensure_ascii=False, allow_nan=False), **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def load_stage(root, stage, ids, identity):
    expected = {record_path(root, stage, sid).name: sid for sid in ids}
    if len(expected) != len(ids):
        raise ValueError('Duplicate expected sample IDs')
    loaded = {}
    for path in sorted((Path(root) / 'state' / stage).glob('*.npz')):
        if path.name not in expected:
            raise ValueError(f'Unknown result in {stage}: {path.name}')
        sid = expected[path.name]
        loaded[sid] = load_record(path, identity, sample_id=sid)
    return loaded
