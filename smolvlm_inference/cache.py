"""Read-only validation of the original pre-connector FP16 cache format."""
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np

CAMERA_ORDER = ['Front', 'Front-Left', 'Front-Right', 'Back', 'Back-Left', 'Back-Right']
FEATURE_FILE = 'features.float16.memmap'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def record_key(record):
    paths = []
    for camera in CAMERA_ORDER:
        raw = record['camera_paths'][camera].replace('\\', '/')
        path = PurePosixPath(raw)
        if path.is_absolute() or '..' in path.parts or ':' in raw:
            raise ValueError('Camera paths must be relative dataset paths')
        paths.append(path.as_posix())
    return hashlib.sha256(canonical(paths)).hexdigest()


def records(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


class FeatureCache:
    def __init__(self, directory, prepared_directory, identity):
        self.directory = Path(directory)
        self.prepared = Path(prepared_directory)
        manifest = json.loads((self.directory / 'manifest.json').read_text(encoding='utf-8'))
        self.index = json.loads((self.directory / 'index.json').read_text(encoding='utf-8'))
        if any(manifest.get(key) != value for key, value in identity.items()):
            raise ValueError('Cache identity/processor/geometry mismatch')
        fingerprint = hashlib.sha256(canonical({k: v for k, v in manifest.items() if k != 'fingerprint'})).hexdigest()
        if manifest.get('fingerprint') != fingerprint:
            raise ValueError('Cache manifest fingerprint mismatch')
        names = ['manifest.json', 'train.jsonl', 'val.jsonl', 'test.jsonl']
        if manifest.get('prepared_hashes') != {name: digest(self.prepared / name) for name in names}:
            raise ValueError('Prepared-data hashes mismatch')
        if manifest.get('index_sha256') != digest(self.directory / 'index.json'):
            raise ValueError('Cache index hash mismatch')
        counts, required = {}, set()
        for split in ('train', 'val', 'test'):
            count = 0
            for row in records(self.prepared / f'{split}.jsonl'):
                required.add(record_key(row))
                count += 1
            counts[split] = count
        prepared_manifest = json.loads((self.prepared / 'manifest.json').read_text(encoding='utf-8'))
        if counts != prepared_manifest.get('counts') or counts != manifest.get('record_counts'):
            raise ValueError('Prepared record counts mismatch')
        if (not isinstance(self.index, dict) or set(self.index) != required
                or any(type(v) is not int for v in self.index.values())
                or sorted(self.index.values()) != list(range(len(self.index)))):
            raise ValueError('Cache index must densely cover all prepared records')
        geometry = identity['geometry']
        shape = [len(self.index), geometry['num_views'], geometry['vision_tokens'], geometry['vision_dim']]
        if not self.index or manifest.get('shape') != shape or manifest.get('num_samples') != len(self.index):
            raise ValueError('Cache shape mismatch')
        payload = self.directory / FEATURE_FILE
        if payload.stat().st_size != int(np.prod(shape)) * 2:
            raise ValueError('Cache feature file size mismatch')
        self.array = np.memmap(payload, dtype=np.float16, mode='r', shape=tuple(shape))
        flat = self.array.reshape(-1)
        for offset in range(0, flat.size, 1024 * 1024):
            if not np.isfinite(flat[offset:offset + 1024 * 1024]).all():
                raise ValueError('Cache contains nonfinite features')
        self.manifest = manifest
        self.counts = counts

    def sample(self, split='test', index=0):
        if split not in self.counts:
            raise ValueError('split must be train, val or test')
        if type(index) is not int or not 0 <= index < self.counts[split]:
            raise IndexError('Sample index outside split')
        row = next(row for i, row in enumerate(records(self.prepared / f'{split}.jsonl')) if i == index)
        features = np.array(self.array[self.index[record_key(row)]], copy=True)
        return {**row, 'visual_features': features}
