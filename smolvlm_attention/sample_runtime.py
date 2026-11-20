"""Kaggle orchestration for full-test per-question prompt attention exports."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .sample_store import (BATCH_SIZE, atomic_json, digest, fingerprint, input_rows, read_run,
                           shards_complete, shard_bounds, write_run, open_prompt_attention,
                           ShardStore, pid_is_alive)

_PROCESSES = {}


def resolve_run_mode(output_root, mode='auto'):
    if mode not in ('auto', 'measure', 'resume', 'export'):
        raise ValueError('Mode must be auto, measure, resume or export')
    output = Path(output_root)
    path = output / 'sample_run.json'
    if not path.exists():
        if mode in ('resume', 'export'):
            raise FileNotFoundError('No existing sample attention run')
        return 'measure'
    run = read_run(path)
    if mode == 'measure':
        raise ValueError('Run exists; choose a new OUTPUT or resume')
    if (output / 'export' / 'manifest.json').exists():
        result = open_prompt_attention(output / 'export')
        if result.metadata['fingerprint'] != run['fingerprint']:
            raise ValueError('Export differs from run')
        return 'export'
    if shards_complete(run):
        return 'export'
    if mode == 'export':
        raise ValueError('Export requires all shards complete')
    return 'resume'


def resolve_artifacts(output_root, cache_input_dir='', prepared_dir=None):
    """Base only. Prefer existing prepared files and matching mounted cache."""
    from huggingface_hub import hf_hub_download
    from .config import MODELS, PREPARED_REPO, PREPARED_REVISION, CACHE_FINGERPRINT, CACHE_DATASET
    output = Path(output_root).resolve()
    prepared = Path(prepared_dir).resolve() if prepared_dir else output / 'prepared_data'
    for name in ('manifest.json', 'train.jsonl', 'val.jsonl', 'test.jsonl'):
        if not (prepared / name).is_file():
            hf_hub_download(repo_id=PREPARED_REPO, repo_type='dataset', revision=PREPARED_REVISION,
                            filename=name, local_dir=prepared, token=False)
    model = dict(MODELS['base'])
    model['checkpoint'] = str(Path(hf_hub_download(repo_id=model['repo'], revision=model['revision'],
        filename='finetune_best.pt', token=False)).resolve())
    if cache_input_dir:
        root = Path(cache_input_dir)
    else:
        mounted = [p.parent for p in Path('/kaggle/input').rglob('features.float16.memmap')
                   if (p.parent / 'index.json').is_file() and (p.parent / 'manifest.json').is_file()
                   and json.loads((p.parent / 'manifest.json').read_text())['fingerprint'] == CACHE_FINGERPRINT]
        if len(mounted) > 1:
            raise ValueError('Several matching caches; set CACHE_INPUT explicitly')
        if mounted:
            root = mounted[0]
        else:
            import kagglehub
            root = Path(kagglehub.dataset_download(CACHE_DATASET))
    candidates = [p.parent for p in root.rglob('features.float16.memmap')
                  if (p.parent / 'manifest.json').is_file() and (p.parent / 'index.json').is_file()
                  and json.loads((p.parent / 'manifest.json').read_text())['fingerprint'] == CACHE_FINGERPRINT]
    if len(candidates) != 1:
        raise ValueError(f'Expected one matching cache in {root}, found {len(candidates)}')
    return dict(model=model, cache_dir=str(candidates[0].resolve()), prepared_dir=str(prepared), output_root=str(output))


def assert_measurement_compatible(run):
    from .sample_inputs import library_versions, measurement_sources
    if run['identity']['sources'] != measurement_sources() or run['identity']['versions'] != library_versions():
        raise ValueError('Measurement code/library versions changed; choose a new OUTPUT')


def prepare_run(output_root, *, mode='auto', cache_input_dir='', prepared_dir=None,
                gpu_count=None, batch_size=BATCH_SIZE, source_commit=None, source_sha256=None):
    output = Path(output_root).resolve()
    resolved = resolve_run_mode(output, mode)
    path = output / 'sample_run.json'
    if resolved == 'export':
        return path
    if batch_size != BATCH_SIZE:
        raise ValueError('Initial batch must be 32 per GPU')
    previous = read_run(path) if path.exists() else None
    if previous:
        assert_measurement_compatible(previous)
    import torch
    from smolvlm_inference.cache import CAMERA_ORDER
    from .config import MODELS, QUERY_POLICY, PREPARED_REPO, PREPARED_REVISION
    from .sample_inputs import prepare_inputs, library_versions, measurement_sources
    count = min(2, torch.cuda.device_count()) if gpu_count is None else gpu_count
    if count not in (1, 2) or count > torch.cuda.device_count():
        raise ValueError('Select one or two available CUDA GPUs')
    if previous and count != previous['shard_count']:
        raise ValueError('Resume must retain original GPU shard count; set GPU_COUNT explicitly')
    output.mkdir(parents=True, exist_ok=True)
    artifacts = resolve_artifacts(output, cache_input_dir, prepared_dir)
    rows, receipt, data_identity = prepare_inputs(artifacts, output)
    identity = dict(policy=QUERY_POLICY, reduction='mean_heads_only_per_sample_v1',
        model=MODELS['base']['repo'], revision=MODELS['base']['revision'],
        checkpoint_sha256=digest(artifacts['model']['checkpoint']),
        data=data_identity, precision='float32_no_tf32', sources=measurement_sources(),
        versions=library_versions(), camera_order=list(CAMERA_ORDER),
        prepared_repo=PREPARED_REPO, prepared_revision=PREPARED_REVISION)
    config = dict(version=1, identity=identity, artifacts=artifacts, cache_descriptor=str(receipt.resolve()),
        shard_count=count, batch_size=BATCH_SIZE, geometry=[30, 384], camera_order=list(CAMERA_ORDER),
        source_commit=source_commit, source_sha256=source_sha256)
    path = write_run(output, config, rows, mode=resolved)
    print(f'Ready: {len(rows)} questions, {count} GPU(s), initial batch=32/GPU. {path}', flush=True)
    return path


def _pid_path(run, shard):
    return Path(run['output_root']) / 'logs' / f'samples_{shard}.pid.json'


def _pid_alive(path):
    try:
        return pid_is_alive(json.loads(Path(path).read_text())['pid'])
    except (FileNotFoundError, OSError, ProcessLookupError):
        return False


def launch_workers(config_path):
    from filelock import FileLock
    path = Path(config_path).resolve()
    run = read_run(path)
    assert_measurement_compatible(run)
    import torch
    if torch.cuda.device_count() < run['shard_count']:
        raise RuntimeError('Not enough GPUs for saved shard topology')
    logs = Path(run['output_root']) / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    with FileLock(str(logs / 'launch.lock'), timeout=0):
        for shard in range(run['shard_count']):
            key = (str(path), shard)
            current = _PROCESSES.get(key)
            if (current is not None and current.poll() is None) or _pid_alive(_pid_path(run, shard)):
                print(f'Shard {shard}: already running; reconnect with wait cell.', flush=True)
                continue
            progress_path = Path(run['output_root']) / 'shards' / str(shard) / 'progress.json'
            if progress_path.exists() and json.loads(progress_path.read_text()).get('complete'):
                start, end = shard_bounds(run['count'], run['shard_count'], shard)
                store = ShardStore(progress_path.parent, run['fingerprint'], list(range(start, end)), run['geometry'])
                if len(store.completed) != end-start:
                    raise ValueError('Completed shard coverage differs')
                atomic_json(logs / f'samples_{shard}.status.json', dict(status='complete', stage='complete',
                    processed=end-start, committed=end-start, total=end-start,
                    fingerprint=run['fingerprint'], batch_size=store.batch_size))
                _PROCESSES.pop(key, None)
                print(f'Shard {shard}: already measured.', flush=True)
                continue
            start, end = shard_bounds(run['count'], run['shard_count'], shard)
            atomic_json(logs / f'samples_{shard}.status.json', dict(status='starting', stage='starting',
                processed=0, committed=0, total=end-start, fingerprint=run['fingerprint'], batch_size=32))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), TOKENIZERS_PARALLELISM='false',
                       PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1')
            source = str(Path(__file__).resolve().parents[1])
            env['PYTHONPATH'] = source + os.pathsep + env.get('PYTHONPATH', '')
            kwargs = {'start_new_session': True} if os.name != 'nt' else {'creationflags': subprocess.CREATE_NO_WINDOW}
            with (logs / f'samples_{shard}.log').open('a', encoding='utf-8') as stream:
                process = subprocess.Popen([sys.executable, '-m', 'smolvlm_attention.sample_worker',
                    '--config', str(path), '--shard', str(shard)], stdout=stream, stderr=subprocess.STDOUT,
                    env=env, cwd=source, **kwargs)
            _PROCESSES[key] = process
            atomic_json(_pid_path(run, shard), dict(pid=process.pid, config=str(path)))
            print(f'Shard {shard}: GPU {shard}, PID {process.pid}, log {logs / (f"samples_{shard}.log")}', flush=True)
    return config_path


def wait_workers(config_path, *, poll_seconds=1):
    from tqdm.auto import tqdm
    path = Path(config_path).resolve()
    run = read_run(path)
    logs = Path(run['output_root']) / 'logs'
    bars = [tqdm(total=shard_bounds(run['count'], run['shard_count'], s)[1] -
                 shard_bounds(run['count'], run['shard_count'], s)[0],
                 desc=f'GPU {s}', unit='question', position=s) for s in range(run['shard_count'])]
    total = tqdm(total=run['count'], desc='Total', unit='question', position=len(bars))
    try:
        while True:
            finished, processed = True, 0
            for shard, bar in enumerate(bars):
                status_path = logs / f'samples_{shard}.status.json'
                if not status_path.exists():
                    raise RuntimeError('Launch workers before waiting')
                status = json.loads(status_path.read_text(encoding='utf-8'))
                if status['fingerprint'] != run['fingerprint']:
                    raise ValueError('Worker progress fingerprint differs')
                process = _PROCESSES.get((str(path), shard))
                code = process.poll() if process else None
                alive = code is None if process else _pid_alive(_pid_path(run, shard))
                if status['status'] == 'failed' or (code is not None and code != 0):
                    raise RuntimeError(f'GPU {shard}: {status.get("error", code)}; log: {logs / (f"samples_{shard}.log")}')
                if not alive and status['status'] != 'complete':
                    raise RuntimeError(f'Worker exited without completion; read {logs / (f"samples_{shard}.log")}')
                finished = finished and not alive and status['status'] == 'complete'
                bar.update(status['processed'] - bar.n)
                bar.set_postfix_str(f"{status['stage']}, batch={status.get('batch_size', '-')}, committed={status.get('committed', 0)}", refresh=False)
                processed += status['processed']
            total.update(processed-total.n)
            if finished:
                return
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print('Workers keep running. Rerun the wait cell to reconnect.', flush=True)
        raise
    finally:
        for bar in bars + [total]:
            bar.close()
