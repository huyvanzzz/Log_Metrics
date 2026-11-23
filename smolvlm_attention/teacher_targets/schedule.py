"""Immutable, original-row schedules; shuffling never consumes model RNG."""
from pathlib import Path
import os
import numpy as np
import torch
from smolvlm_attention.sample_store import atomic_json, digest


def split_order(count, split, seed=42):
    if split not in ('train', 'val') or count < 0:
        raise ValueError('Invalid schedule split/count')
    if split == 'val':
        return np.arange(count, dtype=np.int64)
    return torch.randperm(count, generator=torch.Generator(device='cpu').manual_seed(seed)).numpy()


def rank_rows(count, rank, split='val', seed=42):
    if rank not in (0, 1):
        raise ValueError('Two shards required')
    return split_order(count, split, seed)[rank::2]


def prepare_schedule(directory, counts, batch_size=64, seed=42):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    description = dict(version=1, seed=seed, batch_size=batch_size,
                       policy='train_randperm_val_sequential_rank_stride_v1', splits={})
    for split, count in counts.items():
        path = directory / f'{split}.npy'
        expected = split_order(count, split, seed)
        if path.exists():
            actual = np.load(path, mmap_mode='r', allow_pickle=False)
            if actual.dtype != np.dtype('int64') or not np.array_equal(actual, expected):
                raise ValueError('Stored schedule changed; refusing resume')
        else:
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.save(stream, expected, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        description['splits'][split] = dict(count=count, sha256=digest(path))
    manifest = directory / 'manifest.json'
    if manifest.exists():
        import json
        if json.loads(manifest.read_text()) != description:
            raise ValueError('Schedule identity changed')
    atomic_json(manifest, description)
    return description


def load_rows(directory, description, split, rank):
    path = Path(directory) / f'{split}.npy'
    if rank not in (0, 1) or digest(path) != description['splits'][split]['sha256']:
        raise ValueError('Schedule checksum/rank mismatch')
    rows = np.load(path, mmap_mode='r', allow_pickle=False)
    if rows.dtype != np.dtype('int64') or rows.shape != (description['splits'][split]['count'],):
        raise ValueError('Schedule geometry mismatch')
    return rows[rank::2]


def pending_batches(store, rows, batch_size=64):
    """Never re-pack pending rows: a receipt must cover a complete scheduled batch."""
    pending = set(map(int, store.pending_rows))
    result = []
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        flags = [int(row) in pending for row in batch]
        if any(flags) and not all(flags):
            raise ValueError('Partial scheduled batch committed; refusing resume')
        if all(flags):
            result.append(batch.tolist())
    # A fully committed batch must also have been recorded as one receipt.
    receipts = getattr(store, 'directory', None)
    if receipts is not None:
        import json
        allowed = {tuple(map(int, rows[i:i + batch_size])) for i in range(0, len(rows), batch_size)}
        for receipt in (Path(receipts) / 'commits').glob('*.json'):
            if tuple(json.loads(receipt.read_text())['rows']) not in allowed:
                raise ValueError('Receipt does not match fixed batch schedule')
    return result
