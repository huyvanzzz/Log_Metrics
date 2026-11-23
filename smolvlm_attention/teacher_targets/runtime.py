"""Two fixed-batch GPU workers, safe resume, and portable teacher exports.

This orchestration never runs generation, teacher forcing or training.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np

from smolvlm_attention.sample_store import atomic_json, digest, fingerprint

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'configs/smolvlm_teacher_index22_train_val.json'
BATCH_SIZE = 64


def load_config(path=DEFAULT_CONFIG):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    # Method settings are deliberately not CLI knobs. A new method needs its own format.
    reference = json.loads(DEFAULT_CONFIG.read_text(encoding='utf-8'))
    if value != reference:
        raise ValueError('Teacher configuration differs from the pinned index-22 train/val protocol')
    return value


def shard_rows(count, rank):
    if rank not in (0, 1) or count < 0:
        raise ValueError('Two shards and a nonnegative count required')
    boundary = (count + 1) // 2
    return np.arange(0 if rank == 0 else boundary, boundary if rank == 0 else count, dtype=np.int64)


def required_space(counts, visual_tokens=384):
    payload = sum(counts.values()) * visual_tokens * 4
    # Shards + final temporary payload, prompt IDs/mapping, ZIP and safety reserve.
    return dict(numeric_payload_bytes=payload, recommended_free_bytes=4 * payload + 3 * 1024**3)


def compatible_run(previous, current):
    if previous['identity'] != current['identity']:
        raise ValueError('Resume identity changed; use a new output directory')


def process_pending(store, lengths, measure, *, progress=None):
    """Fixed 64, except final partial batch; no OOM catch/retry or dropped samples."""
    pending = store.pending_rows
    ordered = pending[np.lexsort((pending, -np.asarray(lengths)[pending].astype(np.int64)))]
    for offset in range(0, len(ordered), BATCH_SIZE):
        rows = ordered[offset:offset + BATCH_SIZE]
        values = measure(rows)
        store.commit(rows, values)
        if progress:
            progress(len(rows))


def _sources():
    names = ['attention.py', 'prompts.py', 'storage.py', 'runtime.py']
    paths = [Path(__file__).parent / name for name in names]
    paths += [ROOT / name for name in ('smolvlm_attention/data.py', 'smolvlm_attention/collector.py',
              'smolvlm_inference/model.py', 'smolvlm_inference/text.py', 'smolvlm_inference/cache.py',
              'smolvlm_inference/decoder.py', 'smolvlm_attention/sample_runtime.py',
              'smolvlm_attention/config.py', 'smolvlm_attention/sample_inputs.py')]
    return {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in paths}


def validate_teacher_cache(cache_dir, prepared_dir, receipt_dir, expected_fingerprint=None):
    """Single parent payload pass: finiteness AND content SHA, without cache writes.

    A manifest fingerprint alone cannot detect a finite edit to the memmap.
    Workers reuse this receipt and only verify file metadata plus each batch.
    """
    from smolvlm_attention.data import IDENTITY_KEYS
    from smolvlm_inference.cache import FeatureCache, FEATURE_FILE, CAMERA_ORDER
    directory, prepared = Path(cache_dir).resolve(), Path(prepared_dir).resolve()
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    identity = {k: manifest[k] for k in IDENTITY_KEYS}
    if ((expected_fingerprint is not None and manifest['fingerprint'] != expected_fingerprint) or
            identity['storage_dtype'] != 'float16' or identity['compute_dtype'] != 'float32' or
            identity['camera_order'] != CAMERA_ORDER):
        raise ValueError('Teacher cache fingerprint/dtype/camera order mismatch')
    print(json.dumps(dict(phase='validate_cache_identity_and_coverage', cache_dir=str(directory))), flush=True)
    reader = FeatureCache(directory, prepared, identity, full=False)
    payload = directory / FEATURE_FILE
    before = payload.stat()
    hasher = hashlib.sha256()
    flat = reader.array.reshape(-1)
    for offset in range(0, flat.size, 2 * 1024 * 1024):
        block = flat[offset:offset + 2 * 1024 * 1024]
        if not np.isfinite(block).all():
            raise ValueError('Teacher cache contains nonfinite features')
        hasher.update(memoryview(block).cast('B'))
        _validation_progress(dict(stage='features_finite_and_sha256',
            completed=min(offset + len(block), flat.size) * 2, total=flat.nbytes))
    after = payload.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Cache payload changed during validation')
    descriptor = dict(version=1, cache_dir=str(directory), prepared_dir=str(prepared),
        fingerprint=manifest['fingerprint'], manifest_sha256=digest(directory / 'manifest.json'),
        index_sha256=digest(directory / 'index.json'), prepared_hashes=manifest['prepared_hashes'],
        payload_stat=dict(size=after.st_size, mtime_ns=after.st_mtime_ns),
        payload_sha256=hasher.hexdigest(), shape=list(reader.array.shape), counts=reader.counts, identity=identity)
    path = Path(receipt_dir) / 'validated_cache.json'
    atomic_json(path, descriptor)
    return path


def prepare(output_root, *, config=DEFAULT_CONFIG, mode='new', cache_input='', prepared_dir=None):
    """Parent-only validation and compact prompt preparation. No CUDA/model allocation."""
    from filelock import FileLock
    from transformers import AutoConfig, AutoProcessor
    from smolvlm_attention.sample_runtime import resolve_artifacts
    from smolvlm_attention.config import (MODELS, PREPARED_REPO, PREPARED_REVISION, CACHE_DATASET, CACHE_FINGERPRINT)
    from smolvlm_attention.sample_inputs import library_versions
    from smolvlm_attention.data import open_validated_cache
    from smolvlm_inference.model import MODEL_ID, MODEL_REVISION, _configure_processor, _geometry, processor_policy
    from .prompts import prepare_prompt_store
    cfg = load_config(config)
    if (MODELS['base']['repo'] != cfg['checkpoint_repo'] or MODELS['base']['revision'] != cfg['checkpoint_revision']
            or PREPARED_REPO != cfg['prepared_repo'] or PREPARED_REVISION != cfg['prepared_revision']
            or CACHE_DATASET != cfg['cache_dataset'] or CACHE_FINGERPRINT != cfg['cache_fingerprint']):
        raise ValueError('Artifact resolver pins differ from teacher protocol')
    if mode not in ('new', 'resume'):
        raise ValueError('Mode must be new or resume')
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as locks:
        locks.enter_context(FileLock(str(output / 'parent.lock'), timeout=0))
        for rank in (0, 1):
            locks.enter_context(FileLock(str(output / f'worker_{rank}.lock'), timeout=0))
        run_path = output / 'run.json'
        if run_path.exists() and mode == 'new':
            raise ValueError('Run already exists; select resume or a fresh output directory')
        if not run_path.exists() and mode == 'resume':
            raise FileNotFoundError('Resume needs run.json and partial artifacts')
        artifacts = resolve_artifacts(output / 'internal', cache_input, prepared_dir)
        receipt = validate_teacher_cache(artifacts['cache_dir'], artifacts['prepared_dir'],
                                 output / 'internal' / 'validation', cfg['cache_fingerprint'])
        reader = open_validated_cache(receipt)
        native = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=False)
        processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=False)
        _configure_processor(processor, native)
        expected = dict(model_id=MODEL_ID, revision=MODEL_REVISION, geometry=_geometry(native),
                        processor_policy=processor_policy(processor), camera_order=cfg['camera_order'])
        if (any(reader.manifest.get(k) != v for k, v in expected.items()) or
                any(reader.counts[s] != n for s, n in cfg['splits'].items()) or
                native.text_config.num_hidden_layers != 30 or
                native.text_config.num_attention_heads != 9 or native.text_config.num_key_value_heads != 3):
            raise ValueError('Pinned teacher model/cache/prepared geometry or counts differ')
        space = required_space(cfg['splits'])
        print(json.dumps(dict(phase='storage', **space, free_bytes=shutil.disk_usage(output).free)), flush=True)
        if shutil.disk_usage(output).free < space['recommended_free_bytes']:
            raise OSError('Insufficient space for targets, partial shards, prompt store and archive')
        stores = {split: prepare_prompt_store(reader, processor, split,
                    output / 'internal' / 'prompts' / split,
                    max_length=native.text_config.max_position_embeddings,
                    progress=lambda event: print(json.dumps(event), flush=True))
                  for split in cfg['splits']}
        for split, store in stores.items():
            row = store.portable_row(0)
            print(json.dumps(dict(phase='query_probe', split=split,
                **{k: row[k] for k in ('query_token_id', 'query_token_decoded', 'query_position', 'prompt_length')})), flush=True)
        prompt_bytes = sum(p.stat().st_size for p in (output / 'internal' / 'prompts').rglob('*') if p.is_file())
        print(json.dumps(dict(phase='prompt_storage', actual_bytes=prompt_bytes,
            free_bytes=shutil.disk_usage(output).free)), flush=True)
        if shutil.disk_usage(output).free < space['recommended_free_bytes']:
            raise OSError('Insufficient remaining space after prompt preparation; prompt artifacts are preserved')
        identity = dict(format='smolvlm_teacher_log_scores_v1', protocol=cfg,
                        checkpoint_sha256=digest(artifacts['model']['checkpoint']),
                        native_model=MODEL_ID, native_revision=MODEL_REVISION,
                        model=expected, cache_fingerprint=reader.manifest['fingerprint'],
                        cache_manifest_sha256=digest(Path(artifacts['cache_dir']) / 'manifest.json'),
                        cache_index_sha256=digest(Path(artifacts['cache_dir']) / 'index.json'),
                        cache_payload_sha256=json.loads(receipt.read_text())['payload_sha256'],
                        prepared_hashes=reader.manifest['prepared_hashes'],
                        prompts={s: store.identity for s, store in stores.items()},
                        sources=_sources(), versions=library_versions())
        run = dict(version=1, identity=identity, output_root=str(output),
                   checkpoint=artifacts['model']['checkpoint'], cache_descriptor=str(Path(receipt).resolve()),
                   prompts={s: str((output / 'internal' / 'prompts' / s).resolve()) for s in stores})
        if run_path.exists():
            compatible_run(json.loads(run_path.read_text()), run)
        atomic_json(run_path, run)
        print(json.dumps(dict(phase='prepared', counts=cfg['splits'], fingerprint=fingerprint(identity))), flush=True)
        return run_path


_LAST_VALIDATION = [0.0, None]


def _validation_progress(event):
    now = time.monotonic()
    if event['stage'] != _LAST_VALIDATION[1] or now - _LAST_VALIDATION[0] >= 10 or event['completed'] == event['total']:
        print(json.dumps(dict(phase='validate_cache', **event)), flush=True)
        _LAST_VALIDATION[:] = [now, event['stage']]


def _read_run(path):
    run = json.loads(Path(path).read_text(encoding='utf-8'))
    if run.get('version') != 1 or run['identity']['sources'] != _sources():
        raise ValueError('Teacher code/format changed; refusing incompatible resume')
    cfg = run['identity']['protocol']
    if cfg != load_config():
        raise ValueError('Teacher protocol identity differs')
    return run


def _split_identity(run, split):
    return dict(run['identity'], split=split, score_layer_index=22, score_layer_number=23,
                axes=['sample', 'visual_token'], dtype='<f4',
                normalization=run['identity']['protocol']['normalization'])


def _store(run, split, rank):
    from .storage import TargetShard
    count = run['identity']['protocol']['splits'][split]
    return TargetShard(Path(run['output_root']) / 'internal' / 'shards' / split / str(rank),
                       _split_identity(run, split), shard_rows(count, rank), visual_tokens=384)


def run_worker(run_path, rank, gate):
    from filelock import FileLock
    run = _read_run(run_path)
    with FileLock(str(Path(run['output_root']) / f'worker_{rank}.lock'), timeout=0):
        return _run_worker_impl(run_path, rank, gate)


def _run_worker_impl(run_path, rank, gate):
    """One fresh process/model per GPU; both stress tests finish before either commits."""
    import torch
    from smolvlm_inference.model import load_model, processor_policy
    from smolvlm_attention.data import open_validated_cache, make_batch
    from smolvlm_attention.sample_inputs import library_versions
    from .prompts import PromptStore
    from .attention import collect_teacher_log_scores
    run = _read_run(run_path)
    output = Path(run['output_root'])
    if library_versions() != run['identity']['versions']:
        raise ValueError('Library versions changed')
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    reader = open_validated_cache(run['cache_descriptor'])
    if json.loads(Path(run['cache_descriptor']).read_text())['payload_sha256'] != run['identity']['cache_payload_sha256']:
        raise ValueError('Cache validation receipt content hash differs from run')
    if digest(run['checkpoint']) != run['identity']['checkpoint_sha256']:
        raise ValueError('Checkpoint hash changed')
    model = load_model(run['checkpoint'], device=f'cuda:{rank}',
                       expected_prompt_camera_order=run['identity']['protocol']['camera_order'])
    if model.geometry != run['identity']['model']['geometry'] or processor_policy(model.processor) != run['identity']['model']['processor_policy']:
        raise ValueError('Worker model/processor differs from validated identity')
    model.requires_grad_(False)
    print(json.dumps(dict(phase='worker_model', gpu=rank, dtype=str(next(model.parameters()).dtype),
        training=model.training, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        layer_index=22, layer_number=23, query_heads=9, kv_heads=3, visual_tokens=384,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
        autocast_enabled=torch.is_autocast_enabled(), attention_backend='native_sdpa_prefix_qk_target')), flush=True)
    stores = {s: PromptStore(p) for s, p in run['prompts'].items()}
    if any(stores[s].identity != run['identity']['prompts'][s] for s in stores):
        raise ValueError('Prompt store identity changed')
    def measure(split, rows):
        selected = [stores[split].row(int(i)) for i in rows]
        batch = make_batch(reader, selected, model.processor, run['identity']['protocol']['camera_order'])
        if not torch.isfinite(batch['visual_features']).all():
            raise ValueError('Nonfinite feature batch')
        with torch.inference_mode(), torch.autocast('cuda', enabled=False):
            values = collect_teacher_log_scores(model, batch, layer_index=22)
        return values.detach().cpu().numpy().astype('<f4', copy=False)
    with ExitStack() as stack:
        shards = {s: _store(run, s, rank) for s in stores}
        for shard in shards.values():
            stack.enter_context(shard.lock())
        for split, prompts in stores.items():
            rows = shard_rows(len(prompts), rank)
            longest = rows[np.lexsort((rows, -prompts.lengths[rows].astype(np.int64)))][:BATCH_SIZE]
            torch.cuda.reset_peak_memory_stats(rank)
            started = time.perf_counter()
            measure(split, longest)
            torch.cuda.synchronize(rank)
            print(json.dumps(dict(phase='stress_passed', gpu=rank, split=split, batch=len(longest),
                longest_prompt=int(prompts.lengths[longest].max()), elapsed_seconds=time.perf_counter()-started,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(rank),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(rank))), flush=True)
        atomic_json(output / 'logs' / f'ready_{rank}.json', dict(gate=str(gate), rank=rank))
        while not Path(gate).exists():
            time.sleep(.2)
        for split, store in shards.items():
            start = time.perf_counter()
            initial = len(store.pending_rows)
            processed = [0]
            last = [0.0]
            def report(batch):
                processed[0] += batch
                elapsed = time.perf_counter()-start
                if elapsed-last[0] >= 10 or processed[0] == initial:
                    speed = processed[0]/max(elapsed, 1e-9)
                    print(json.dumps(dict(phase='extract', gpu=rank, split=split, batch=batch,
                        processed=processed[0], committed=len(store.rows)-len(store.pending_rows), total=len(store.rows),
                        remaining_at_session_start=initial,
                        elapsed_seconds=elapsed, samples_per_second=speed,
                        eta_seconds=(initial-processed[0])/max(speed, 1e-9),
                        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(rank),
                        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(rank))), flush=True)
                    last[0] = elapsed
            process_pending(store, stores[split].lengths, lambda rows: measure(split, rows), progress=report)


def launch(run_path):
    from filelock import FileLock
    import torch
    from .prompts import PromptStore
    from .storage import finalize_split, validate_export
    run = _read_run(run_path)
    output = Path(run['output_root'])
    with FileLock(str(output / 'parent.lock'), timeout=0):
        complete = all((output / 'export' / s / 'manifest.json').exists() for s in run['prompts'])
        if complete:
            for split in run['prompts']:
                validate_export(output / 'export' / split, _split_identity(run, split))
            print('Complete portable export validated; reusing without GPU extraction.', flush=True)
            return
        if torch.cuda.device_count() < 2:
            raise RuntimeError('Exactly two available CUDA GPUs are required; no CPU fallback')
        logs = output / 'logs'
        logs.mkdir(exist_ok=True)
        # Unique gate avoids accidentally reusing an earlier session's ready files.
        gate = logs / f'gate_{time.time_ns()}.json'
        processes = []
        try:
            with ExitStack() as files:
                readers = []
                for rank in (0, 1):
                    stream = files.enter_context((logs / f'gpu_{rank}.log').open('a', encoding='utf-8'))
                    reader = files.enter_context((logs / f'gpu_{rank}.log').open('r', encoding='utf-8'))
                    reader.seek(0, os.SEEK_END)
                    readers.append(reader)
                    env = dict(os.environ, PYTHONUNBUFFERED='1', USE_TF='0', USE_FLAX='0')
                    processes.append(subprocess.Popen([sys.executable, '-m',
                        'smolvlm_attention.teacher_targets', 'worker', '--run', str(Path(run_path).resolve()),
                        '--rank', str(rank), '--gate', str(gate)], stdout=stream, stderr=subprocess.STDOUT, env=env))
                while True:
                    for rank, reader in enumerate(readers):
                        for line in reader:
                            print(f'[GPU {rank}] {line.rstrip()}', flush=True)
                    failed = [p for p in processes if p.poll() is not None and p.returncode != 0]
                    if failed:
                        raise RuntimeError('Teacher worker failed; see logs/gpu_*.log. Committed shards are retained.')
                    if all(p.poll() == 0 for p in processes):
                        break
                    if not gate.exists():
                        ready = []
                        for rank in (0, 1):
                            path = logs / f'ready_{rank}.json'
                            ready.append(path.exists() and json.loads(path.read_text()).get('gate') == str(gate))
                        if all(ready):
                            atomic_json(gate, dict(stress_passed=True))
                            print('Both GPUs passed longest-64 stress; extraction started.', flush=True)
                    time.sleep(.5)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for split, path in run['prompts'].items():
            prompts = PromptStore(path)
            stores = [_store(run, split, rank) for rank in (0, 1)]
            finalize_split(output / 'export' / split, _split_identity(run, split), stores,
                           (prompts.portable_row(i) for i in range(len(prompts))), len(prompts), visual_tokens=384)
        print('Train and validation exports complete.', flush=True)


def bundle(run_path, *, partial=False):
    from filelock import FileLock
    run = _read_run(run_path)
    with ExitStack() as locks:
        output = Path(run['output_root'])
        locks.enter_context(FileLock(str(output / 'parent.lock'), timeout=0))
        for rank in (0, 1):
            locks.enter_context(FileLock(str(output / f'worker_{rank}.lock'), timeout=0))
        return _bundle_stopped_run(run_path, partial=partial)


def _bundle_stopped_run(run_path, *, partial=False):
    """Package only outputs: never vision cache, checkpoint, prepared data or images."""
    from .storage import validate_export
    run = _read_run(run_path)
    output = Path(run['output_root'])
    if not partial:
        for split in run['prompts']:
            validate_export(output / 'export' / split, _split_identity(run, split))
        files = [output / 'export' / split / name for split in run['prompts']
                 for name in ('log_probs.npy', 'samples.jsonl', 'manifest.json')]
        target = output / 'teacher_index22_train_val_complete.zip'
    else:
        files = [output / 'run.json']
        files += list((output / 'internal' / 'prompts').rglob('*'))
        files += list((output / 'internal' / 'shards').rglob('*'))
        target = output / 'teacher_index22_train_val_partial_resume.zip'
    with zipfile.ZipFile(str(target) + '.tmp', 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in sorted(files):
            if path.is_file() and path.suffix != '.lock':
                archive.write(path, path.relative_to(output))
    os.replace(str(target) + '.tmp', target)
    atomic_json(target.with_suffix('.sha256.json'), dict(filename=target.name, sha256=digest(target)))
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'validate', 'bundle', 'worker'])
    parser.add_argument('--output', default='/kaggle/working/smolvlm_teacher_index22_train_val')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--mode', choices=['new', 'resume'], default='new')
    parser.add_argument('--cache-input', default='')
    parser.add_argument('--prepared-dir')
    parser.add_argument('--run')
    parser.add_argument('--rank', type=int, choices=[0, 1])
    parser.add_argument('--gate')
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--export', help='Validate a relocated portable export root without run.json or inputs')
    args = parser.parse_args(argv)
    run_path = Path(args.run) if args.run else Path(args.output) / 'run.json'
    if args.command == 'prepare':
        prepare(args.output, config=args.config, mode=args.mode, cache_input=args.cache_input, prepared_dir=args.prepared_dir)
    elif args.command == 'worker':
        if args.rank is None or not args.gate:
            parser.error('worker requires --rank and --gate')
        run_worker(run_path, args.rank, args.gate)
    elif args.command == 'run':
        launch(run_path)
    elif args.command == 'bundle':
        print(bundle(run_path, partial=args.partial))
    else:
        from .storage import validate_export
        run = None if args.export else _read_run(run_path)
        for split in ('train', 'val') if args.export else run['prompts']:
            validate_export(Path(args.export) / split if args.export else Path(run['output_root']) / 'export' / split,
                            None if args.export else _split_identity(run, split))
            print(json.dumps(dict(split=split, valid=True)), flush=True)


if __name__ == '__main__':
    main()
