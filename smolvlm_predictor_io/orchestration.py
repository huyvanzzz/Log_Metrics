"""Two independent CUDA connector builders; parent validates raw storage once."""
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import numpy as np

from smolvlm_attention.sample_store import atomic_json, digest
from .projected import ProjectedCache, build_cache, finalize_cache, project_native, verify_native_parity


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _check_origin(cache, model, checkpoint, prepared):
    identity = cache.manifest['identity']
    if identity.get('checkpoint_sha256') != digest(checkpoint) or identity.get('model') != model.architecture_signature:
        raise ValueError('Projected checkpoint/connector identity mismatch')
    for name, sha in identity['raw_cache_descriptor']['prepared_hashes'].items():
        if digest(Path(prepared) / name) != sha:
            raise ValueError('Projected prepared identity mismatch')


def resolve_projected(cfg, artifacts, model, output_dir, projected_input='', projected_dir=''):
    """Return validated projected path; Input reuse never opens raw-cache files."""
    output = Path(output_dir).resolve()
    checkpoint, prepared = artifacts['model']['checkpoint'], artifacts['prepared_dir']
    if projected_input:
        cache = ProjectedCache(projected_input)
        _check_origin(cache, model, checkpoint, prepared)
        return str(cache.directory.resolve())
    directory = Path(projected_dir or output / 'projected_visual_cache').resolve()
    if (directory / 'manifest.json').is_file():
        cache = ProjectedCache(directory)
        _check_origin(cache, model, checkpoint, prepared)
        return str(directory)
    import torch
    if torch.cuda.device_count() < 2:
        raise RuntimeError('Projected builder requires two CUDA GPUs')
    from smolvlm_attention.teacher_targets.runtime import validate_teacher_cache
    logs = output / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    internal = output / 'internal' / 'projected-builder'
    internal.mkdir(parents=True, exist_ok=True)
    # Source validator prints detailed progress; keep it entirely in files.
    with (logs / 'projected_validation.log').open('a', encoding='utf-8') as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            receipt = validate_teacher_cache(artifacts['cache_dir'], prepared, internal)
    descriptor = _read(receipt)
    raw_dir = Path(artifacts['cache_dir']).resolve()
    identity = dict(raw_cache_manifest=_read(raw_dir / 'manifest.json'), raw_cache_descriptor=descriptor,
                    checkpoint_sha256=digest(checkpoint), model=model.architecture_signature)
    session = uuid.uuid4().hex
    spec = dict(identity=identity, raw_dir=str(raw_dir), checkpoint=str(Path(checkpoint).resolve()),
                output=str(directory), gate=str(internal / session), world_size=2, batch_size=32,
                projected_shape=[6, model.geometry['visual_tokens_per_view'], model.geometry['text_dim']])
    spec_path = internal / f'{session}.json'
    atomic_json(spec_path, spec)
    processes = []
    with contextlib.ExitStack() as stack:
        try:
            for rank in range(2):
                stream = stack.enter_context((logs / f'projected_gpu_{rank}.log').open('a', encoding='utf-8'))
                processes.append(subprocess.Popen([sys.executable, '-m', 'smolvlm_predictor_io',
                    'build-worker', '--spec', str(spec_path), '--rank', str(rank)], stdout=stream,
                    stderr=subprocess.STDOUT, env=dict(os.environ, USE_TF='0', USE_FLAX='0',
                        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')))
            while any(p.poll() is None for p in processes):
                failed = [(r, p.returncode) for r, p in enumerate(processes) if p.poll() not in (None, 0)]
                if failed:
                    raise RuntimeError(f'Projected worker failed {failed}; see {logs}/projected_gpu_*.log')
                time.sleep(.25)
            if any(p.returncode != 0 for p in processes):
                raise RuntimeError(f'Projected worker failed; see {logs}/projected_gpu_*.log')
        finally:
            for p in processes:
                if p.poll() is None:
                    p.terminate()
            for p in processes:
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
    finalize_cache(directory)
    return str(directory)


def build_worker(spec_path, rank):
    import torch
    from smolvlm_inference.model import load_model, CAMERA_ORDER
    spec = _read(spec_path)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    raw_dir = Path(spec['raw_dir'])
    descriptor = spec['identity']['raw_cache_descriptor']
    payload = raw_dir / 'features.float16.memmap'
    stat = payload.stat()
    if (dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) != descriptor['payload_stat']
            or digest(raw_dir / 'manifest.json') != descriptor['manifest_sha256']
            or digest(raw_dir / 'index.json') != descriptor['index_sha256']
            or digest(spec['checkpoint']) != spec['identity']['checkpoint_sha256']):
        raise ValueError('Projected builder validated inputs changed')
    model = load_model(spec['checkpoint'], device=f'cuda:{rank}', expected_prompt_camera_order=CAMERA_ORDER)
    if model.architecture_signature != spec['identity']['model']:
        raise ValueError('Projected builder native model identity mismatch')
    raw = np.memmap(payload, dtype=np.float16, mode='r', shape=tuple(descriptor['shape']))
    rows = list(range(rank, len(raw), spec['world_size']))[:spec['batch_size']]
    if not rows:
        raise ValueError('Projected builder empty rank shard')
    parity = verify_native_parity(model, np.asarray(raw[rows], dtype=np.float32))
    gate = Path(spec['gate'])
    gate.mkdir(parents=True, exist_ok=True)
    atomic_json(gate / f'{rank}.json', parity)
    while not all((gate / f'{i}.json').exists() for i in range(spec['world_size'])):
        time.sleep(.25)
    build_cache(raw, _read(raw_dir / 'index.json'), spec['output'], lambda batch: project_native(model, batch),
                spec['identity'], batch_size=spec['batch_size'], rank=rank, world_size=spec['world_size'],
                projected_shape=spec['projected_shape'], progress=lambda value: print(json.dumps(value), flush=True))
