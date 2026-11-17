"""Small notebook helpers for pinned artifacts, independent workers and results."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoProcessor

from smolvlm_inference.model import MODEL_ID, MODEL_REVISION, _configure_processor, _geometry, processor_policy
from .config import (MODELS, PREPARED_REPO, PREPARED_REVISION, CACHE_DATASET,
                     CACHE_FINGERPRINT, TEST_COUNT)
from .data import validate_cache, open_validated_cache, prepare_test

_PROCESSES = {}


def resolve_artifacts(output_root, cache_input_dir=''):
    from huggingface_hub import hf_hub_download
    import kagglehub
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepared = output / 'prepared_data'
    for filename in ('manifest.json', 'train.jsonl', 'val.jsonl', 'test.jsonl'):
        hf_hub_download(repo_id=PREPARED_REPO, repo_type='dataset',
            revision=PREPARED_REVISION, filename=filename, local_dir=prepared, token=False)
    models = {}
    for name, spec in MODELS.items():
        checkpoint = hf_hub_download(repo_id=spec['repo'], revision=spec['revision'],
                                     filename='finetune_best.pt', token=False)
        models[name] = dict(spec, checkpoint=str(Path(checkpoint).resolve()))
    cache_root = Path(cache_input_dir) if cache_input_dir else Path(kagglehub.dataset_download(CACHE_DATASET))
    candidates = [p.parent for p in cache_root.rglob('features.float16.memmap')
                  if (p.parent / 'manifest.json').is_file() and (p.parent / 'index.json').is_file()]
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one complete cache in {cache_root}, found {len(candidates)}')
    return dict(output_root=str(output), cache_dir=str(candidates[0].resolve()),
                prepared_dir=str(prepared.resolve()), models=models)


def prepare_run(artifacts, *, source_commit=None, source_archive_sha256=None):
    output = Path(artifacts['output_root'])
    descriptor = validate_cache(artifacts['cache_dir'], artifacts['prepared_dir'],
                                output, expected_fingerprint=CACHE_FINGERPRINT)
    reader = open_validated_cache(descriptor)
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=False)
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=False)
    _configure_processor(processor, config)
    identity = dict(model_id=MODEL_ID, revision=MODEL_REVISION,
                    processor_policy=processor_policy(processor), geometry=_geometry(config),
                    compute_dtype='float32', storage_dtype='float16')
    if any(reader.manifest.get(k) != v for k, v in identity.items()):
        raise ValueError('Pinned processor/model does not match cache')
    if reader.counts['test'] != TEST_COUNT:
        raise ValueError('Expected the complete pinned test set')
    for name, spec in artifacts['models'].items():
        spec['prepared_test'] = str(prepare_test(reader, processor, spec['camera_order'],
            output / 'inputs' / name, config.text_config.max_position_embeddings).resolve())
    run = dict(artifacts, cache_descriptor=str(descriptor.resolve()),
               expected_test_count=TEST_COUNT, prepared_repo=PREPARED_REPO,
               prepared_revision=PREPARED_REVISION, source_commit=source_commit,
               source_archive_sha256=source_archive_sha256)
    path = output / 'run.json'
    path.write_text(json.dumps(run, indent=2), encoding='utf-8')
    return path


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def launch_worker(config_path, model_name):
    """Return immediately: separate cells can launch both GPUs before waiting."""
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    spec = config['models'][model_name]
    key = (str(config_path), model_name)
    current = _PROCESSES.get(key)
    if current is not None and current.poll() is None:
        print(f'{model_name}: worker {current.pid} already running')
        return current
    logs = Path(config['output_root']) / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    pidfile = logs / f'{model_name}.pid.json'
    if pidfile.exists():
        saved = json.loads(pidfile.read_text(encoding='utf-8'))
        if _pid_alive(saved['pid']) and saved['pid'] != os.getpid():
            raise RuntimeError(f'{model_name}: previous worker PID {saved["pid"]} still exists; do not launch a duplicate')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(spec['gpu']),
                       TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1',
                       PYTHONUNBUFFERED='1')
    source = str(Path(__file__).resolve().parents[1])
    environment['PYTHONPATH'] = source + os.pathsep + environment.get('PYTHONPATH', '')
    with (logs / f'{model_name}.log').open('a', encoding='utf-8') as log:
        kwargs = {'start_new_session': True} if os.name != 'nt' else {'creationflags': subprocess.CREATE_NO_WINDOW}
        process = subprocess.Popen([sys.executable, '-m', 'smolvlm_attention.worker',
            '--config', str(config_path), '--model', model_name], env=environment,
            stdout=log, stderr=subprocess.STDOUT, **kwargs)
    _PROCESSES[key] = process
    pidfile.write_text(json.dumps(dict(pid=process.pid, config=str(config_path))), encoding='utf-8')
    print(f'{model_name}: PID {process.pid}, physical GPU {spec["gpu"]}, log {logs / (model_name + ".log")}')
    return process


def wait_workers(config_path):
    path = Path(config_path).resolve()
    processes = {name: _PROCESSES.get((str(path), name)) for name in MODELS}
    if any(p is None for p in processes.values()):
        raise RuntimeError('Launch both model cells before waiting')
    try:
        while any(p.poll() is None for p in processes.values()):
            failed = {name: p.returncode for name, p in processes.items()
                      if p.poll() is not None and p.returncode != 0}
            if failed:
                raise RuntimeError(f'Worker failed: {failed}. Read the model log before retrying.')
            config = json.loads(path.read_text(encoding='utf-8'))
            for name in processes:
                log = Path(config['output_root']) / 'logs' / f'{name}.jsonl'
                if log.exists():
                    with log.open(encoding='utf-8') as stream:
                        last = None
                        for line in stream:
                            if line.strip():
                                last = line.strip()
                    if last:
                        print(last, flush=True)
            time.sleep(10)
        failed = {name: p.returncode for name, p in processes.items() if p.returncode != 0}
        if failed:
            raise RuntimeError(f'Worker failed: {failed}; inspect logs/')
    except KeyboardInterrupt:
        print('Workers keep running; rerun this wait cell to reconnect.', flush=True)
        raise
    print('Both model workers completed.')


def load_results(config_path):
    from .aggregate import AttentionAccumulator
    from .data import load_prepared_test
    from .plotting import shared_scales
    from smolvlm_inference.cache import digest, canonical
    from .config import QUERY_POLICY
    import hashlib
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    output = Path(config['output_root'])
    results, metadata = {}, {}
    for name, spec in config['models'].items():
        folder = output / 'data' / name
        info = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        if info.get('status') != 'complete' or info.get('count') != config['expected_test_count']:
            raise ValueError(f'{name}: result is incomplete')
        prepared = load_prepared_test(spec['prepared_test'])
        expected = dict(algorithm=QUERY_POLICY, model=spec['repo'], revision=spec['revision'],
            checkpoint_sha256=info['checkpoint_sha256'], order=spec['camera_order'],
            cache=prepared['manifest_fingerprint'], prepared_sha256=digest(spec['prepared_test']),
            precision='float32', transformers='4.46.3',
            source_archive_sha256=config.get('source_archive_sha256'))
        fingerprint = hashlib.sha256(canonical(expected)).hexdigest()
        if info.get('fingerprint') != fingerprint or info.get('order') != spec['camera_order']:
            raise ValueError(f'{name}: result does not match this notebook run')
        accumulator = AttentionAccumulator.load(folder / 'attention.npz', (30, 384), fingerprint,
                    [row['sample_id'] for row in prepared['records']])
        if accumulator.pending_ids or accumulator.count != config['expected_test_count']:
            raise ValueError(f'{name}: missing test samples')
        results[name], metadata[name] = accumulator.mean, info
    scales = shared_scales(list(results.values()))
    summary = dict(models=metadata, scales={key: value.tolist() if isinstance(value, np.ndarray) else value
                                          for key, value in scales.items()},
                   histogram_population='384 head-and-sample-averaged visual-token positions per layer')
    (output / 'figure_metadata.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return results, metadata, scales, summary
