"""Sample-weighted, resumable CPU attention statistics."""
import os
from pathlib import Path
import tempfile

import numpy as np


class AttentionAccumulator:
    def __init__(self, shape, fingerprint, sample_ids):
        self.shape = tuple(shape)
        if len(self.shape) != 2 or any(type(n) is not int or n <= 0 for n in self.shape):
            raise ValueError('attention shape must be positive [layers, visual tokens]')
        self.fingerprint = str(fingerprint)
        self.sample_ids = list(sample_ids)
        if not self.sample_ids or any(not isinstance(s, str) for s in self.sample_ids) or len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError('expected sample IDs must be unique strings')
        self.sum = np.zeros(self.shape, dtype=np.float64)
        self._processed = []
        self._expected = set(self.sample_ids)
        self._seen = set()

    @property
    def count(self):
        return len(self._processed)

    @property
    def processed_ids(self):
        return list(self._processed)

    @property
    def pending_ids(self):
        return [s for s in self.sample_ids if s not in self._seen]

    @property
    def mean(self):
        return self.sum / self.count if self.count else np.zeros(self.shape, dtype=np.float64)

    def add(self, sample_ids, attention):
        ids = list(sample_ids)
        if not ids or any(not isinstance(s, str) for s in ids) or len(set(ids)) != len(ids) or not set(ids) <= self._expected or set(ids) & self._seen:
            raise ValueError('duplicate, unknown, or empty attention sample IDs')
        if hasattr(attention, 'detach'):
            if attention.device.type != 'cpu':
                raise ValueError('attention must be on CPU')
            attention = attention.detach().double().numpy()
        values = np.asarray(attention, dtype=np.float64)
        if values.shape != (len(ids), *self.shape) or not np.isfinite(values).all():
            raise ValueError('attention shape mismatch or nonfinite attention')
        updated = self.sum + values.sum(axis=0, dtype=np.float64)
        if not np.isfinite(updated).all():
            raise ValueError('attention sum overflow')
        self.sum = updated
        self._processed.extend(ids)
        self._seen.update(ids)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.npz')
        try:
            with os.fdopen(fd, 'wb') as stream:
                np.savez(stream, sum=self.sum, mean=self.mean, count=self.count,
                         processed_ids=np.asarray(self.processed_ids, dtype=str),
                         sample_ids=np.asarray(self.sample_ids, dtype=str),
                         fingerprint=self.fingerprint, shape=self.shape)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path, shape, fingerprint, sample_ids):
        result = cls(shape, fingerprint, sample_ids)
        with np.load(path, allow_pickle=False) as saved:
            ids = saved['processed_ids'].tolist()
            total = saved['sum']
            count = saved['count'].item()
            if (saved['fingerprint'].item() != result.fingerprint or
                    saved['sample_ids'].tolist() != result.sample_ids or
                    tuple(saved['shape'].tolist()) != result.shape or
                    not isinstance(ids, list) or any(not isinstance(s, str) for s in ids) or
                    len(set(ids)) != len(ids) or not set(ids) <= result._expected or
                    type(count) is not int or count != len(ids) or
                    total.shape != result.shape or total.dtype != np.float64 or
                    not np.isfinite(total).all() or
                    saved['mean'].shape != result.shape or
                    not np.array_equal(saved['mean'], total / count if count else np.zeros(result.shape)) or
                    (count == 0 and np.any(total))):
                raise ValueError('incompatible or damaged attention checkpoint')
            result.sum = total.copy()
            result._processed = ids
            result._seen = set(ids)
        return result
