"""Durable per-question FP32 shards and portable NumPy-only attention exports."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import zipfile

import numpy as np

from .config import BATCH_SIZE, LAYER_IDS, FORMAT, FORMULA, K
EXPORT_FILES = ('attention.npy', 'samples.jsonl', 'metadata.json', 'README.md')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def atomic_json(path, value):
    return atomic_bytes(path, canonical(value) + b'\n')


def sync_file(path):
    with Path(path).open('r+b') as stream:
        os.fsync(stream.fileno())


def pid_is_alive(pid):
    """Non-destructive process probe, including on Windows (kill(pid,0) is unsafe there)."""
    if not isinstance(pid, int) or pid < 1:
        return False
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # Access denied: conservatively alive.
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        stat = Path(f'/proc/{pid}/stat')
        if stat.exists() and stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z':
            return False
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except (OSError, ProcessLookupError):
        return False


def shard_bounds(count, shards, shard):
    if shards not in (1, 2) or not 0 <= shard < shards or count < shards:
        raise ValueError('Invalid shard topology')
    size, extra = divmod(count, shards)
    start = shard * size + min(shard, extra)
    return start, start + size + (shard < extra)


def validate_samples(rows, visual_tokens):
    ids, evals = set(), set()
    for i, row in enumerate(rows):
        sequence = row['input_ids']
        if (row['row_index'] != i or not isinstance(row['sample_id'], str)
                or row['sample_id'] in ids or row['eval_id'] in evals or not sequence
                or row['prompt_length'] != len(sequence) or row['query_position'] != len(sequence) - 1
                or row['query_token_id'] != sequence[-1]
                or sequence.count(row['image_token_id']) != visual_tokens):
            raise ValueError('Invalid sample/prompt mapping')
        positions = [j for j, token in enumerate(sequence) if token == row['image_token_id']]
        if row.get('image_positions', positions) != positions:
            raise ValueError('Image positions differ from native prompt')
        ids.add(row['sample_id'])
        evals.add(row['eval_id'])


def write_run(output_root, config, rows, *, mode='auto'):
    from filelock import FileLock
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / 'prepare.lock'), timeout=0):
        return _write_run(output, config, rows, mode=mode)


def _write_run(output_root, config, rows, *, mode='auto'):
    if mode not in ('auto', 'measure', 'resume'):
        raise ValueError('Invalid measurement mode')
    output = Path(output_root).resolve()
    path = output / 'combined_run.json'
    if config['batch_size'] != BATCH_SIZE or tuple(config['layer_ids']) != LAYER_IDS or list(config['geometry']) != [len(LAYER_IDS), 384]:
        raise ValueError('Invalid fixed experiment configuration')
    validate_samples(rows, config['geometry'][1])
    payload = b''.join(canonical(row) + b'\n' for row in rows)
    identity = dict(config['identity'], inputs_sha256=hashlib.sha256(payload).hexdigest(),
                    geometry=list(config['geometry']), shard_count=config['shard_count'],
                    layer_ids=list(LAYER_IDS), format=FORMAT, batch_size=BATCH_SIZE)
    candidate = dict(config, identity=identity, fingerprint=fingerprint(identity),
                     output_root=str(output), count=len(rows), inputs=str(output / 'samples.jsonl'),
                     inputs_sha256=identity['inputs_sha256'])
    if path.exists():
        existing = read_run(path)
        if mode == 'measure':
            raise ValueError('Run already exists; use resume or a new OUTPUT')
        if existing['fingerprint'] != candidate['fingerprint']:
            raise ValueError('Run incompatible; choose a new OUTPUT')
        updates = {key: candidate[key] for key in ('artifacts', 'cache_descriptor')
                   if key in candidate and candidate[key] != existing.get(key)}
        if updates:
            from filelock import FileLock
            logs = output / 'logs'
            logs.mkdir(parents=True, exist_ok=True)
            with FileLock(str(logs / 'launch.lock'), timeout=0):
                if any(pid_is_alive(json.loads(p.read_text())['pid']) for p in logs.glob('*.pid.json')):
                    raise RuntimeError('Worker still running; finish it before updating operational paths')
                # The identity/mapping and original measurement provenance remain immutable.
                atomic_json(path, dict(existing, **updates))
        return path
    if mode == 'resume':
        raise FileNotFoundError('No run exists to resume')
    if (output / 'shards').exists() or (output / 'attention_export').exists():
        raise ValueError('Orphan state exists; choose a new OUTPUT')
    inputs = output / 'samples.jsonl'
    if inputs.exists() and inputs.read_bytes() != payload:
        raise ValueError('Orphan inputs incompatible; choose a new OUTPUT')
    atomic_bytes(inputs, payload)
    atomic_json(path, candidate)
    return path


def read_run(path):
    path = Path(path).resolve()
    run = json.loads(path.read_text(encoding='utf-8'))
    run['config_path'] = str(path)
    run['output_root'] = str(path.parent)
    run['inputs'] = str(path.parent / 'samples.jsonl')
    if (run['fingerprint'] != fingerprint(run['identity'])
            or run['geometry'] != run['identity']['geometry']
            or run['shard_count'] != run['identity']['shard_count']
            or run['batch_size'] != BATCH_SIZE
            or tuple(run['layer_ids']) != LAYER_IDS
            or run['identity'].get('layer_ids') != list(LAYER_IDS)
            or run['identity'].get('format') != FORMAT
            or run.get('camera_order') != run['identity'].get('camera_order')):
        raise ValueError('Run identity mismatch')
    if digest(run['inputs']) != run['inputs_sha256'] or run['inputs_sha256'] != run['identity']['inputs_sha256']:
        raise ValueError('Input mapping hash mismatch')
    input_rows(run)
    return run


def input_rows(run):
    rows = [json.loads(line) for line in Path(run['inputs']).read_text(encoding='utf-8').splitlines()]
    validate_samples(rows, run['geometry'][1])
    if len(rows) != run['count']:
        raise ValueError('Input count mismatch')
    return rows


def validate_attention(values, shape):
    values = np.asarray(values)
    if values.dtype != np.dtype('float32') or values.shape != shape:
        raise ValueError(f'Expected FP32 attention shape {shape}, got {values.dtype} {values.shape}')
    if not np.isfinite(values).all() or (values < 0).any() or (values.sum(-1, dtype=np.float64) > 1 + 1e-6).any():
        raise ValueError('Attention must be finite nonnegative probabilities without image renormalization')
    return values


class ShardStore:
    """Caller holds a shard lock. Only progress.json makes flushed rows committed."""

    def __init__(self, directory, identity, global_rows, geometry=(10, 384)):
        self.directory = Path(directory)
        self.path = self.directory / 'attention.npy'
        self.progress_path = self.directory / 'progress.json'
        self.global_rows = list(global_rows)
        self.local = {row: i for i, row in enumerate(self.global_rows)}
        self.geometry = tuple(geometry)
        self.identity = identity
        self.completed = set()
        self.pending = set()
        self.checksums = {}
        self.batch_size = BATCH_SIZE
        if not self.global_rows or len(self.local) != len(self.global_rows):
            raise ValueError('Empty or duplicate shard rows')
        shape = (len(self.global_rows), *self.geometry)
        if self.progress_path.exists():
            progress = json.loads(self.progress_path.read_text(encoding='utf-8'))
            if (progress['identity'] != identity or progress['global_rows'] != self.global_rows
                    or progress['shape'] != list(shape)):
                raise ValueError('Shard identity/topology differs')
            self.batch_size = progress['batch_size']
            if self.batch_size != BATCH_SIZE:
                raise ValueError('Invalid persisted batch size')
            self.checksums = progress['checksums']
            self.completed = {int(i) for i in self.checksums}
            if not self.completed <= set(self.global_rows) or len(self.completed) != len(self.checksums):
                raise ValueError('Invalid committed rows')
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.array = np.load(self.path, mmap_mode='r+', allow_pickle=False)
            if self.array.shape != shape or self.array.dtype != np.dtype('<f4') or not self.array.flags.c_contiguous:
                raise ValueError('Shard shape/dtype differs')
        else:
            if self.progress_path.exists():
                raise ValueError('Missing committed shard file')
            temporary = self.directory / 'initializing.npy'
            array = np.lib.format.open_memmap(temporary, mode='w+', dtype='<f4', shape=shape)
            array[:] = np.nan
            array.flush()
            del array
            sync_file(temporary)
            os.replace(temporary, self.path)
            self.array = np.load(self.path, mmap_mode='r+', allow_pickle=False)
        for row in self.completed:
            value = self.array[self.local[row]]
            validate_attention(value, self.geometry)
            if hashlib.sha256(value.tobytes(order='C')).hexdigest() != self.checksums[str(row)]:
                raise ValueError(f'Committed row checksum mismatch: {row}')

    @property
    def processed(self):
        return len(self.completed | self.pending)

    def write(self, indices, values):
        indices = list(indices)
        values = validate_attention(values, (len(indices), *self.geometry))
        if len(set(indices)) != len(indices) or not set(indices) <= set(self.global_rows):
            raise ValueError('Unknown or duplicate rows')
        if set(indices) & (self.completed | self.pending):
            raise ValueError('Row already measured')
        for i, row in enumerate(indices):
            self.array[self.local[row]] = values[i]
        self.pending.update(indices)

    def commit(self, *, batch_size=None):
        if batch_size is not None and batch_size != BATCH_SIZE:
            raise ValueError('Batch size must remain 64')
        self.array.flush()
        sync_file(self.path)
        hashes = dict(self.checksums)
        for row in self.pending:
            hashes[str(row)] = hashlib.sha256(self.array[self.local[row]].tobytes(order='C')).hexdigest()
        atomic_json(self.progress_path, dict(version=1, identity=self.identity, global_rows=self.global_rows,
            shape=list(self.array.shape), batch_size=self.batch_size, checksums=hashes,
            complete=len(hashes) == len(self.global_rows)))
        self.checksums = hashes
        self.completed.update(self.pending)
        self.pending.clear()


def shards_complete(run):
    for shard in range(run['shard_count']):
        start, end = shard_bounds(run['count'], run['shard_count'], shard)
        path = Path(run['output_root']) / 'shards' / str(shard) / 'progress.json'
        if not path.exists():
            return False
        value = json.loads(path.read_text(encoding='utf-8'))
        if value['identity'] != run['fingerprint'] or value['global_rows'] != list(range(start, end)):
            raise ValueError('Shard identity mismatch')
        if not value['complete'] or set(value['checksums']) != {str(i) for i in range(start, end)}:
            return False
    return True


def load_shard(run, shard):
    if not isinstance(run, dict):
        run = read_run(run)
    start, end = shard_bounds(run['count'], run['shard_count'], shard)
    return ShardStore(Path(run['output_root']) / 'shards' / str(shard),
                      run['fingerprint'], range(start, end), tuple(run['geometry']))


def finalize_export(config_path, *, progress=True):
    """No Torch/model/cache access; reentrant after an interrupted merge."""
    from filelock import FileLock
    from tqdm.auto import tqdm
    run = read_run(config_path)
    output = Path(run['output_root'])
    export = output / 'attention_export'
    with FileLock(str(output / 'finalize.lock'), timeout=0):
        if (export / 'manifest.json').exists():
            result = open_attention(export)
            if result.metadata['fingerprint'] != run['fingerprint']:
                raise ValueError('Export identity mismatch')
            return export
        if not shards_complete(run):
            raise ValueError('Measurement incomplete; finish all shards first')
        rows = input_rows(run)
        export.mkdir(parents=True, exist_ok=True)
        temporary = export / 'attention.partial.npy'
        target = np.lib.format.open_memmap(temporary, mode='w+', dtype='<f4',
                                         shape=(run['count'], *run['geometry']))
        with tqdm(total=run['count'], desc='Verify and merge shards', unit='question', disable=not progress) as bar:
            for shard in range(run['shard_count']):
                start, end = shard_bounds(run['count'], run['shard_count'], shard)
                store = ShardStore(output / 'shards' / str(shard), run['fingerprint'],
                                   list(range(start, end)), run['geometry'])
                if len(store.completed) != end - start:
                    raise ValueError('Measurement incomplete')
                for offset in range(0, end - start, 128):
                    block = store.array[offset:offset + 128]
                    target[start + offset:start + offset + len(block)] = block
                    bar.update(len(block))
                del store
        target.flush()
        del target
        sync_file(temporary)
        os.replace(temporary, export / 'attention.npy')
        atomic_bytes(export / 'samples.jsonl', Path(run['inputs']).read_bytes())
        metadata = dict(version=1, status='complete', fingerprint=run['fingerprint'], identity=run['identity'],
            count=run['count'], shape=[run['count'], *run['geometry']], dtype='<f4',
            axes=['sample', 'layer', 'visual_token'], layer_ids=list(LAYER_IDS), format=FORMAT, formula=FORMULA,
            camera_order=run.get('camera_order'),
            grid=[8, 8], tokens_per_camera=64, query='last valid native prompt token including assistant suffix',
            normalization='softmax over all valid prompt keys including self, then select image keys',
            reduction='mean heads before products; retain every sample and selected layer',
            source_commit=run.get('source_commit'), source_sha256=run.get('source_sha256'))
        atomic_json(export / 'metadata.json', metadata)
        atomic_bytes(export / 'README.md', (
            '# Per-question same-layer combined attention score\n\n'
            f'Formula: {FORMULA}.\nLayers: {list(LAYER_IDS)} (axis is not layer index).\n'
            'FP32 [sample, layer, visual_token]. No sample/layer averaging or image-only normalization.\n'
            'Query is the last valid native prompt token predicting the first answer token.\n'
            'Rows follow samples.jsonl, not lexical sample ID order. Answers are metadata only.\n'
            'Visual index j: camera=j//64; row,column=divmod(j%64,8).\n'
            'Absolute prompt positions are indices whose input_ids equal image_token_id.\n'
            'Use open_attention(export_dir) for a verified read-only NumPy memmap.\n'
            'For layer l: scores=attention[row, metadata["layer_ids"].index(l)].\n'
            'indices=np.argsort(-scores, kind="stable")[:k] (ties follow visual index).\n'
            'This export retains per-sample scores for the subsequent Top-32 benchmark.\n'
            'Extract ZIP before memory mapping.\n'
        ).encode('utf-8'))
        atomic_json(export / 'manifest.json', dict(version=1, format=FORMAT, status='complete', fingerprint=run['fingerprint'],
            count=len(rows), shape=metadata['shape'], sha256={name: digest(export / name) for name in EXPORT_FILES}))
    return export


def open_attention(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('format') != FORMAT or manifest.get('status') != 'complete'
            or set(manifest['sha256']) != set(EXPORT_FILES)):
        raise ValueError('Export incomplete')
    for name in EXPORT_FILES:
        if digest(directory / name) != manifest['sha256'][name]:
            raise ValueError(f'Export hash mismatch: {name}')
    metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    array = np.load(directory / 'attention.npy', mmap_mode='r', allow_pickle=False)
    rows = [json.loads(line) for line in (directory / 'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    if (metadata.get('format') != FORMAT or metadata.get('layer_ids') != list(LAYER_IDS)
            or metadata['fingerprint'] != manifest['fingerprint'] or fingerprint(metadata['identity']) != manifest['fingerprint']
            or digest(directory / 'samples.jsonl') != metadata['identity']['inputs_sha256']
            or list(array.shape) != manifest['shape'] or list(array.shape) != metadata['shape']
            or len(rows) != array.shape[0] or len(rows) != manifest['count']
            or array.shape[1:] != (len(LAYER_IDS), 384)
            or array.dtype != np.dtype('<f4') or not array.flags.c_contiguous):
        raise ValueError('Export geometry/identity mismatch')
    validate_samples(rows, array.shape[2])
    return SimpleNamespace(attention=array, samples=rows, metadata=metadata,
                           sample_index={row['sample_id']: i for i, row in enumerate(rows)})


def bundle_export(directory, destination=None):
    from filelock import FileLock
    directory = Path(directory)
    open_attention(directory)
    destination = Path(destination) if destination else directory.parent / 'prompt_attention_export.zip'
    with FileLock(str(destination) + '.lock', timeout=0):
        temporary = destination.with_suffix('.partial.zip')
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for name in (*EXPORT_FILES, 'manifest.json'):
                archive.write(directory / name, arcname=name)
        os.replace(temporary, destination)
    return destination


def prepare_selections(directory, destination=None):
    """Stable per-sample Top-32, sorted back into original prompt order."""
    from filelock import FileLock
    result = open_attention(directory)
    directory = Path(directory)
    destination = Path(destination) if destination else directory.parent / 'selections'
    destination.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination / 'prepare.lock'), timeout=0):
        manifest_path = destination / 'manifest.json'
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding='utf-8'))
            if existing.get('attention_sha256') != digest(directory / 'attention.npy'):
                raise ValueError('Selection attention payload mismatch')
            for layer in LAYER_IDS:
                load_selections(destination, layer, fingerprint_value=result.metadata['fingerprint'])
            return destination
        files = {}
        for axis, layer in enumerate(LAYER_IDS):
            indices = np.empty((len(result.samples), K), dtype=np.uint16)
            positions = np.empty((len(result.samples), K), dtype=np.int32)
            for start in range(0, len(indices), 128):
                block = result.attention[start:start + 128, axis]
                chosen = np.sort(np.argsort(-block, axis=-1, kind='stable')[:, :K], axis=-1)
                indices[start:start + len(block)] = chosen
                for local, selection in enumerate(chosen):
                    row = result.samples[start + local]
                    image_positions = row.get('image_positions')
                    if image_positions is None:
                        image_positions = [i for i, token in enumerate(row['input_ids'])
                                           if token == row['image_token_id']]
                    positions[start + local] = np.asarray(image_positions)[selection]
            name = f'layer_{layer:02d}.npz'
            temporary = destination / f'{name}.partial'
            with temporary.open('wb') as stream:
                np.savez(stream, indices=indices, absolute_positions=positions)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination / name)
            files[name] = digest(destination / name)
        atomic_json(manifest_path, dict(format=FORMAT + '_top32', status='complete',
            attention_fingerprint=result.metadata['fingerprint'],
            attention_sha256=digest(directory / 'attention.npy'),
            layer_ids=list(LAYER_IDS), count=len(result.samples), k=K,
            tie_policy='lower visual index first; selected indices restored to prompt order', sha256=files))
    return destination


def load_selections(directory, layer, *, fingerprint_value=None):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('format') != FORMAT + '_top32' or manifest.get('status') != 'complete'
            or manifest.get('layer_ids') != list(LAYER_IDS) or layer not in LAYER_IDS
            or manifest.get('k') != K or (fingerprint_value is not None
                and manifest['attention_fingerprint'] != fingerprint_value)):
        raise ValueError('Selection identity mismatch')
    name = f'layer_{layer:02d}.npz'
    if digest(directory / name) != manifest['sha256'][name]:
        raise ValueError('Selection hash mismatch')
    with np.load(directory / name, allow_pickle=False) as data:
        indices = data['indices']
        positions = data['absolute_positions']
    if (indices.shape != (manifest['count'], K) or positions.shape != indices.shape
            or indices.dtype != np.uint16 or positions.dtype != np.int32
            or (indices >= 384).any() or (np.diff(indices.astype(np.int32), axis=1) <= 0).any()
            or (positions < 0).any() or (np.diff(positions, axis=1) <= 0).any()):
        raise ValueError('Invalid selection geometry/order')
    return dict(indices=indices, absolute_positions=positions, manifest=manifest)
