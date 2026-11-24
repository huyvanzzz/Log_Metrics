"""Durable, content-bound FP32 native-connector cache (no decoder execution)."""
import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from filelock import FileLock
from smolvlm_attention.sample_store import atomic_json, digest, fingerprint

FORMAT = 'smolvlm_projected_visual_fp32_v1'
PAYLOAD = 'projected.npy'


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _bytes_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype='<f4').tobytes()).hexdigest()


def _index(index):
    if (not index or any(type(v) is not int for v in index.values())
            or sorted(index.values()) != list(range(len(index)))):
        raise ValueError('Projected index must densely map unique six-camera keys')


class ProjectedCache:
    """Read complete caches without requiring the original raw cache to exist."""
    def __init__(self, directory, expected_identity=None, *, full=True):
        self.directory = Path(directory)
        self.manifest = manifest = _read(self.directory / 'manifest.json')
        if manifest.get('format') != FORMAT or manifest.get('complete') is not True:
            raise ValueError('Incomplete or unsupported projected cache')
        if manifest.get('fingerprint') != fingerprint({k: v for k, v in manifest.items() if k != 'fingerprint'}):
            raise ValueError('Projected manifest fingerprint mismatch')
        if expected_identity is not None and manifest['identity'] != expected_identity:
            raise ValueError('Projected cache identity mismatch')
        self.index = _read(self.directory / 'index.json')
        _index(self.index)
        if digest(self.directory / 'index.json') != manifest['index_sha256']:
            raise ValueError('Projected index checksum mismatch')
        self.array = np.load(self.directory / PAYLOAD, mmap_mode='r', allow_pickle=False)
        if (manifest.get('dtype') != '<f4' or self.array.dtype.str != '<f4' or list(self.array.shape) != manifest['shape']
                or self.array.shape[0] != len(self.index)
                or (self.directory / PAYLOAD).stat().st_size != manifest['file_bytes']):
            raise ValueError('Projected geometry/dtype/file size mismatch')
        if full:
            if digest(self.directory / PAYLOAD) != manifest['payload_sha256']:
                raise ValueError('Projected payload checksum mismatch')
            for start in range(0, len(self.array), 32):
                if not np.isfinite(self.array[start:start + 32]).all():
                    raise ValueError('Nonfinite projected features')


def build_cache(raw_array, index, output, project, identity, *, batch_size=32,
                rank=0, world_size=1, projected_shape=(6, 64, 576), progress=None):
    """Build one disjoint rank shard; callbacks receive raw FP32 NumPy batches.

    Flush payload before publishing immutable batch receipts. A crash before receipt
    publication simply causes that batch to be recomputed. Caller owns GPU launch.
    """
    _index(index)
    if batch_size < 1 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError('Invalid projected builder rank/batch configuration')
    if len(raw_array) != len(index):
        raise ValueError('Raw cache and projected index coverage mismatch')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = dict(format=FORMAT, identity=identity, shape=[len(index), *projected_shape],
                index=index, batch_size=batch_size, world_size=world_size)
    with FileLock(str(output / '.initialize.lock')):
        if (output / 'manifest.json').exists():
            complete = ProjectedCache(output, identity)
            if complete.index != index or list(complete.array.shape) != spec['shape']:
                raise ValueError('Completed projected cache index/geometry mismatch')
            return complete
        run = output / 'build.json'
        if run.exists():
            if _read(run) != spec:
                raise ValueError('Projected build identity/schedule mismatch')
            if not (output / PAYLOAD).exists():
                raise ValueError('Projected partial payload missing')
        else:
            unknown = [p.name for p in output.iterdir() if not p.name.endswith('.lock')]
            if unknown:
                raise ValueError('Refusing to overwrite unknown projected-cache contents')
            array = np.lib.format.open_memmap(output / PAYLOAD, mode='w+', dtype='<f4', shape=tuple(spec['shape']))
            array.flush()
            del array
            atomic_json(output / 'index.json', index)
            atomic_json(run, spec)
    with FileLock(str(output / f'.rank-{rank}.lock'), timeout=0):
        array = np.load(output / PAYLOAD, mmap_mode='r+', allow_pickle=False)
        if list(array.shape) != spec['shape'] or array.dtype.str != '<f4':
            raise ValueError('Projected partial geometry/dtype mismatch')
        rows = np.arange(rank, len(index), world_size, dtype=np.int64)
        receipt_dir = output / 'receipts' / str(rank)
        receipt_dir.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(rows), batch_size):
            selected = rows[start:start + batch_size]
            receipt = receipt_dir / f'{start:012d}.json'
            if receipt.exists():
                saved = _read(receipt)
                if saved['rows'] != selected.tolist() or saved['sha256'] != _bytes_hash(array[selected]):
                    raise ValueError('Projected committed batch checksum/schedule mismatch')
                continue
            raw = np.asarray(raw_array[selected], dtype=np.float32)
            if not np.isfinite(raw).all():
                raise ValueError('Nonfinite raw features')
            values = np.asarray(project(raw), dtype='<f4')
            if values.shape != (len(selected), *projected_shape) or not np.isfinite(values).all():
                raise ValueError('Invalid projected connector output')
            array[selected] = values
            array.flush()
            # Explicit durable ordering; receipts must never outlive their payload.
            with (output / PAYLOAD).open('r+b') as stream:
                os.fsync(stream.fileno())
            atomic_json(receipt, dict(rows=selected.tolist(), sha256=_bytes_hash(values)))
            if progress:
                progress(dict(rank=rank, committed=min(start + batch_size, len(rows)), total=len(rows)))


def finalize_cache(output):
    """Validate all rank receipts and publish completion atomically."""
    output = Path(output)
    with FileLock(str(output / '.initialize.lock')):
        spec = _read(output / 'build.json')
        if (output / 'manifest.json').exists():
            return ProjectedCache(output, spec['identity'])
        with ExitStack() as stack:
            for rank in range(spec['world_size']):
                stack.enter_context(FileLock(str(output / f'.rank-{rank}.lock'), timeout=0))
            array = np.load(output / PAYLOAD, mmap_mode='r', allow_pickle=False)
            for rank in range(spec['world_size']):
                rows = list(range(rank, len(spec['index']), spec['world_size']))
                expected = set()
                for start in range(0, len(rows), spec['batch_size']):
                    path = output / 'receipts' / str(rank) / f'{start:012d}.json'
                    expected.add(path.name)
                    if not path.is_file():
                        raise ValueError('Projected cache coverage incomplete')
                    receipt = _read(path)
                    selected = rows[start:start + spec['batch_size']]
                    if receipt['rows'] != selected or receipt['sha256'] != _bytes_hash(array[selected]):
                        raise ValueError('Projected receipt checksum/schedule mismatch')
                if {p.name for p in (output / 'receipts' / str(rank)).glob('*.json')} != expected:
                    raise ValueError('Unexpected projected completion receipts')
            manifest = dict(format=FORMAT, complete=True, identity=spec['identity'], shape=spec['shape'],
                            dtype='<f4', index_sha256=digest(output / 'index.json'),
                            payload_sha256=digest(output / PAYLOAD), file_bytes=(output / PAYLOAD).stat().st_size)
            manifest['fingerprint'] = fingerprint(manifest)
            atomic_json(output / 'manifest.json', manifest)
    return ProjectedCache(output, spec['identity'])


def project_native(backbone, raw_batch):
    """Execute only the pretrained/fine-tuned frozen native connector in FP32."""
    import torch
    connector = backbone.native.model.connector
    if any(p.requires_grad for p in connector.parameters()):
        raise ValueError('Projected builder requires frozen connector')
    connector.eval()
    weight = next(connector.parameters())
    if weight.dtype != torch.float32:
        raise ValueError('Projected builder requires FP32 connector')
    with torch.inference_mode(), torch.autocast(device_type=weight.device.type, enabled=False):
        result = backbone._project(torch.as_tensor(raw_batch, device=weight.device, dtype=torch.float32), len(raw_batch))
        return result.reshape(len(raw_batch), 6, *result.shape[-2:]).cpu().numpy()


def verify_native_parity(backbone, raw_batch, *, rtol=1e-5, atol=1e-6):
    """Measure production-vs-single-group GEMM batching before cache publication.

    This is a numerical guard, not a promise of bitwise CUDA equivalence. Caller
    must stop on failure rather than modifying tolerance to accept its hardware.
    """
    batch = project_native(backbone, raw_batch)
    reference = np.concatenate([project_native(backbone, raw_batch[i:i + 1])
                                for i in range(len(raw_batch))], axis=0)
    error = batch.astype(np.float64) - reference.astype(np.float64)
    report = dict(max_abs_error=float(np.abs(error).max()),
                  relative_l2=float(np.linalg.norm(error.ravel()) /
                                    max(np.linalg.norm(reference.astype(np.float64).ravel()), 1e-30)),
                  rtol=rtol, atol=atol)
    if not np.allclose(batch, reference, rtol=rtol, atol=atol):
        raise ValueError(f'Projected FP32 connector batching parity failed: {report}')
    return report


validate_cache = ProjectedCache
