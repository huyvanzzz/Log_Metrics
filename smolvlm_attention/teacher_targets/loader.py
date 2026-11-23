"""CPU-only spawned loading for teacher extraction. No model is held by workers."""
from contextlib import contextmanager
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from smolvlm_attention.sample_store import atomic_json, fingerprint
from .prompts import PromptStore, FILES

SETTINGS = dict(num_workers=2, persistent_workers=True, pin_memory=True,
                prefetch_factor=1, multiprocessing_context='spawn')


def file_stat(path):
    value = Path(path).stat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns)


def prepare_loader_receipt(path, descriptor, prompts):
    """Called only after full parent cache/prompt validation has succeeded."""
    descriptor = json.loads(Path(descriptor).read_text())
    cache = Path(descriptor['cache_dir'])
    files = [cache / name for name in ('manifest.json', 'index.json', 'features.float16.memmap')]
    files += [Path(directory) / name for directory in prompts.values() for name in (*FILES, 'manifest.json')]
    value = dict(version=1, cache_dir=str(cache), shape=descriptor['shape'],
                 payload_sha256=descriptor['payload_sha256'], prompts=prompts,
                 files={str(p.resolve()): file_stat(p) for p in files})
    atomic_json(path, value)
    return path


class ReceiptInputs:
    def __init__(self, receipt_path):
        self.receipt = json.loads(Path(receipt_path).read_text())
        if self.receipt.get('version') != 1:
            raise ValueError('Unsupported loader receipt')
        if any(file_stat(path) != expected for path, expected in self.receipt['files'].items()):
            raise ValueError('Validated loader input file changed')
        self.array = np.memmap(Path(self.receipt['cache_dir']) / 'features.float16.memmap',
                               dtype='<f2', mode='r', shape=tuple(self.receipt['shape']))
        self.prompts = {}

    def prompt(self, split):
        if split not in self.prompts:
            self.prompts[split] = ReceiptPromptStore(self.receipt['prompts'][split])
        return self.prompts[split]


class ReceiptPromptStore(PromptStore):
    """Parent-validated files; workers check current stat receipt instead of hashing GBs."""
    def __init__(self, directory):
        self.directory = Path(directory)
        manifest = json.loads((self.directory / 'manifest.json').read_text())
        if manifest.get('complete') is not True or manifest.get('version') != 1:
            raise ValueError('Invalid prompt store manifest')
        self.identity = manifest['identity']
        if fingerprint(self.identity) != manifest['identity_sha256']:
            raise ValueError('Prompt identity checksum mismatch')
        self.offsets = np.load(self.directory / 'offsets.npy', mmap_mode='r', allow_pickle=False)
        self.lengths = np.load(self.directory / 'lengths.npy', mmap_mode='r', allow_pickle=False)
        self.row_offsets = np.load(self.directory / 'row_offsets.npy', mmap_mode='r', allow_pickle=False)
        self.ids = np.memmap(self.directory / 'input_ids.bin', mode='r', dtype='<u4')


class TeacherDataset(Dataset):
    def __init__(self, receipt):
        self.receipt = str(receipt)
        self._inputs = None

    def __getstate__(self):
        return dict(receipt=self.receipt, _inputs=None)

    def __getitem__(self, item):
        phase, split, row = item
        if self._inputs is None:
            self._inputs = ReceiptInputs(self.receipt)
        return phase, split, self._inputs.prompt(split).row(int(row))


class TeacherCollator:
    def __init__(self, receipt, pad_token_id, camera_order):
        self.receipt = str(receipt)
        self.pad_token_id = int(pad_token_id)
        self.camera_order = list(camera_order)
        self._inputs = None

    def __getstate__(self):
        return dict(receipt=self.receipt, pad_token_id=self.pad_token_id,
                    camera_order=self.camera_order, _inputs=None)

    def __call__(self, items):
        from smolvlm_inference.cache import CAMERA_ORDER
        if self.camera_order != list(CAMERA_ORDER):
            raise ValueError('Teacher requires canonical camera order')
        if self._inputs is None:
            self._inputs = ReceiptInputs(self.receipt)
        phase, split = items[0][:2]
        if any(item[:2] != (phase, split) for item in items):
            raise ValueError('Mixed split/phase batch')
        rows = [item[2] for item in items]
        lengths = [len(row['input_ids']) for row in rows]
        width = max(lengths)
        ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        # One FP32 allocation, copying FP16 sources directly into their final batch slot.
        features = torch.empty((len(rows), *self._inputs.array.shape[1:]), dtype=torch.float32)
        for i, row in enumerate(rows):
            if type(row['feature_index']) is not int or not 0 <= row['feature_index'] < len(self._inputs.array):
                raise ValueError('Feature index outside validated cache')
            ids[i, -lengths[i]:] = torch.tensor(row['input_ids'], dtype=torch.long)
            mask[i, -lengths[i]:] = 1
            np.copyto(features[i].numpy(), self._inputs.array[row['feature_index']])
        if not torch.isfinite(features).all():
            raise ValueError('Nonfinite feature batch')
        return dict(phase=phase, split=split, rows=torch.tensor([r['row'] for r in rows]),
                    input_ids=ids, attention_mask=mask, visual_features=features,
                    sample_ids=[r['sample_id'] for r in rows])


@contextmanager
def teacher_loader(receipt, batches, pad_token_id, camera_order, rank=0):
    """One pool handles stress, train and val, and is joined on every exit path."""
    generator = torch.Generator(device='cpu').manual_seed(1042 + rank)
    loader = DataLoader(TeacherDataset(receipt), batch_sampler=batches,
        collate_fn=TeacherCollator(receipt, pad_token_id, camera_order), generator=generator, **SETTINGS)
    iterator = None
    try:
        iterator = iter(loader)
        yield iterator
    finally:
        # PyTorch offers no public close API for persistent workers.
        if iterator is not None:
            iterator._shutdown_workers()
        loader._iterator = None


def move_batch(batch, device):
    return {key: value.to(device=device, non_blocking=True) if key in
            ('input_ids', 'attention_mask', 'visual_features') else value for key, value in batch.items()}


def memory_snapshot():
    result = dict(shm_free_bytes=None, cgroup_limit_bytes=None, cgroup_current_bytes=None)
    if Path('/dev/shm').exists():
        result['shm_free_bytes'] = shutil.disk_usage('/dev/shm').free
    for key, paths in (('cgroup_limit_bytes', ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes')),
                       ('cgroup_current_bytes', ('/sys/fs/cgroup/memory.current', '/sys/fs/cgroup/memory/memory.usage_in_bytes'))):
        try:
            path = next((Path(p) for p in paths if Path(p).is_file()), None)
            value = path.read_text().strip() if path else 'max'
            result[key] = int(value) if value != 'max' else None
        except (FileNotFoundError, ValueError):
            pass
    if Path('/proc/meminfo').exists():
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith(('MemAvailable:', 'MemTotal:')):
                key, value, _ = line.split()
                result[key.rstrip(':') + '_bytes'] = int(value) * 1024
    return result
