"""Kaggle stage orchestration. Plot/load paths never import Torch or resolve artifacts."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .config import BATCH_SIZE, DATASET_FINGERPRINT, K, MAX_NEW_TOKENS, POLICY, SEED
from .state import atomic_json, canonical, digest, fingerprint, load_stage, source_hashes

_PROCESSES = {}
STAGES = ('baseline', 'causal', 'localization')
WORKER_STAGES = ('preflight',) + STAGES


def read_run(path):
    path = Path(path).resolve()
    run = json.loads(path.read_text(encoding='utf-8'))
    if run['fingerprint'] != fingerprint(run['identity']):
        raise ValueError('Run fingerprint mismatch')
    inputs = Path(run['inputs'])
    if digest(inputs) != run['inputs_sha256']:
        raise ValueError('Prepared inputs changed')
    return run


def input_rows(run):
    return json.loads(Path(run['inputs']).read_text(encoding='utf-8'))


def shard_rows(run, shard):
    if not 0 <= shard < run['shard_count']:
        raise ValueError('Invalid shard')
    return list(np.array_split(np.asarray(input_rows(run), dtype=object), run['shard_count'])[shard])


def write_run(output_root, config, rows, *, mode='auto'):
    output = Path(output_root).resolve()
    path = output / 'probing_run.json'
    inputs_path = output / 'inputs.json'
    payload = canonical(rows) + b'\n'
    inputs_hash = __import__('hashlib').sha256(payload).hexdigest()
    candidate = dict(config, inputs=str(inputs_path), inputs_sha256=inputs_hash,
                     fingerprint=fingerprint(config['identity']))
    if path.exists():
        existing = read_run(path)
        if mode == 'measure':
            raise ValueError('Run already exists; use resume or a new output directory')
        if (existing['identity'] != candidate['identity'] or existing['inputs_sha256'] != inputs_hash
                or existing.get('shard_count') != candidate.get('shard_count')):
            raise ValueError('Run incompatible; choose a new output directory')
        return path
    if mode == 'resume':
        raise FileNotFoundError('No run exists to resume')
    if inputs_path.exists() and inputs_path.read_bytes() != payload:
        raise ValueError('Orphan prepared inputs incompatible; choose a new output directory')
    if (output / 'state').exists() and any((output / 'state').rglob('*.npz')):
        raise ValueError('State exists without run config; choose a new output directory')
    atomic_json(inputs_path, rows)
    atomic_json(path, candidate)
    return path


def resolve_artifacts(output_root, cache_input_dir='', prepared_dir=None):
    from huggingface_hub import hf_hub_download
    import kagglehub
    from smolvlm_attention.config import CACHE_DATASET, MODELS, PREPARED_REPO, PREPARED_REVISION
    output = Path(output_root).resolve()
    prepared = Path(prepared_dir).resolve() if prepared_dir else output / 'prepared_data'
    for filename in ('manifest.json', 'train.jsonl', 'val.jsonl', 'test.jsonl'):
        if not (prepared / filename).is_file():
            hf_hub_download(repo_id=PREPARED_REPO, repo_type='dataset', revision=PREPARED_REVISION,
                            filename=filename, local_dir=prepared, token=False)
    model = dict(MODELS['base'])
    model['checkpoint'] = str(Path(hf_hub_download(repo_id=model['repo'], revision=model['revision'],
        filename='finetune_best.pt', token=False)).resolve())
    if cache_input_dir:
        root = Path(cache_input_dir)
    else:
        from smolvlm_attention.config import CACHE_FINGERPRINT
        mounted = []
        for payload in Path('/kaggle/input').rglob('features.float16.memmap'):
            manifest = payload.parent / 'manifest.json'
            if manifest.is_file() and (payload.parent / 'index.json').is_file():
                if json.loads(manifest.read_text(encoding='utf-8')).get('fingerprint') == CACHE_FINGERPRINT:
                    mounted.append(payload.parent)
        if len(mounted) > 1:
            raise ValueError('Several matching mounted caches; set CACHE_INPUT explicitly')
        root = mounted[0] if mounted else Path(kagglehub.dataset_download(CACHE_DATASET))
    candidates = [p.parent for p in root.rglob('features.float16.memmap')
                  if (p.parent / 'manifest.json').is_file() and (p.parent / 'index.json').is_file()]
    if len(candidates) != 1:
        raise ValueError(f'Expected exactly one feature cache in {root}; found {len(candidates)}')
    return dict(output_root=str(output), prepared_dir=str(prepared),
                cache_dir=str(candidates[0].resolve()), model=model)


def _library_versions():
    return {name: importlib.metadata.version(name) for name in (
        'torch', 'transformers', 'tokenizers', 'numpy', 'huggingface-hub')}


def _all_complete(run):
    ids = [r['sample_id'] for r in input_rows(run)]
    return all(set(load_stage(run['output_root'], stage, ids, run['identity'][stage])) == set(ids)
               for stage in STAGES)


def resolve_run_mode(output_root, mode='auto'):
    if mode not in ('auto', 'measure', 'resume', 'plot'):
        raise ValueError('Mode must be auto, measure, resume or plot')
    path = Path(output_root) / 'probing_run.json'
    if not path.exists():
        if mode in ('plot', 'resume'):
            raise FileNotFoundError('No existing probing run')
        return 'measure'
    run = read_run(path)
    if mode == 'measure':
        raise ValueError('Run exists; choose a new output directory')
    if mode == 'plot' or (mode == 'auto' and _all_complete(run)):
        if not _all_complete(run):
            raise ValueError('Plot requires all stages complete')
        return 'plot'
    return 'resume'


def prepare_run(output_root, *, mode='auto', cache_input_dir='', prepared_dir=None,
                dataset_dir=None, source_commit=None, source_sha256=None, gpu_count=None,
                artifacts=None, receipt_path=None):
    if mode not in ('auto', 'measure', 'resume', 'plot'):
        raise ValueError('Mode must be auto, measure, resume or plot')
    output = Path(output_root).resolve()
    path = output / 'probing_run.json'
    if mode == 'plot':
        run = read_run(path)
        if not _all_complete(run):
            raise ValueError('Plot requires all stages complete')
        return path
    if path.exists():
        previous = read_run(path)
        if mode == 'measure':
            raise ValueError('Run already exists; choose a new output directory')
        if mode == 'auto' and _all_complete(previous):
            print('Complete results found; plot without downloading model or scanning cache.', flush=True)
            return path
        if previous['measurement_sources'] != measurement_sources() or previous['versions'] != _library_versions():
            raise ValueError('Measurement source/library versions changed; use a new output directory')
    elif mode == 'resume':
        raise FileNotFoundError('No run exists to resume')
    import torch
    from transformers import AutoConfig, AutoProcessor
    from smolvlm_inference.model import MODEL_ID, MODEL_REVISION, _configure_processor, _geometry, processor_policy
    from smolvlm_attention.generated_runtime import _cache_receipt
    from smolvlm_attention.data import open_validated_cache
    from .data import load_dataset, prepare_inputs
    from .config import CAMERA_ORDER
    artifacts = artifacts or resolve_artifacts(output, cache_input_dir, prepared_dir)
    # Candidate validation must never overwrite a receipt referenced by an old
    # run, even when compatibility later fails. Stat changes get a new receipt.
    cache_folder = Path(artifacts['cache_dir'])
    payload_stat = (cache_folder / 'features.float16.memmap').stat()
    validation_key = fingerprint(dict(cache=str(cache_folder.resolve()),
        prepared=str(Path(artifacts['prepared_dir']).resolve()),
        manifest=digest(cache_folder / 'manifest.json'), index=digest(cache_folder / 'index.json'),
        payload_size=payload_stat.st_size, payload_mtime_ns=payload_stat.st_mtime_ns))
    validation_artifacts = dict(artifacts, output_root=str(output / 'validation' / validation_key))
    descriptor = _cache_receipt(validation_artifacts, receipt_path)
    reader = open_validated_cache(descriptor)
    dataset = load_dataset(dataset_dir)
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=False)
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=False)
    _configure_processor(processor, config)
    if (reader.manifest['processor_policy'] != processor_policy(processor)
            or reader.manifest['geometry'] != _geometry(config)
            or config.text_config.num_hidden_layers != 30 or _geometry(config)['visual_tokens_per_view'] != 64):
        raise ValueError('Pinned processor/model differs from cache geometry')
    rows = prepare_inputs(dataset, reader, processor, config)
    count = min(2, torch.cuda.device_count()) if gpu_count is None else gpu_count
    if count not in (1, 2):
        raise ValueError('Select one or two available CUDA GPUs')
    sources = measurement_sources()
    versions = _library_versions()
    checkpoint_hash = digest(artifacts['model']['checkpoint'])
    generation = fingerprint(dict(policy=POLICY, model=artifacts['model']['repo'],
        revision=artifacts['model']['revision'], checkpoint=checkpoint_hash,
        prepared=reader.manifest['prepared_hashes'], cache=reader.manifest['fingerprint'],
        cache_payload_stat=json.loads(Path(descriptor).read_text())['payload_stat'],
        dataset=DATASET_FINGERPRINT, inputs=fingerprint(rows), max_new_tokens=MAX_NEW_TOKENS,
        precision='float32_no_tf32', sources=sources, versions=versions, shard_count=count,
        camera_order=CAMERA_ORDER, do_sample=False, num_beams=1))
    baseline = fingerprint(dict(generation=generation, query='original_generated_position_including_self',
        score='mean_content_logprob_predecessor', matching='exact_literal_original_prefix_offsets_v1'))
    identities = dict(generation=generation, baseline=baseline, preflight=baseline,
        causal=fingerprint(dict(baseline=baseline, k=K, seed=SEED, repeats=3, layers=[1,7,13,19,25])),
        localization=fingerprint(dict(baseline=baseline, metric='global_peak_grid_center_in_bbox_inclusive')))
    run = dict(version=1, output_root=str(output), artifacts=artifacts,
        cache_descriptor=str(descriptor), identity=identities, shard_count=count,
        checkpoint_sha256=checkpoint_hash,
        measurement_sources=sources, versions=versions, dataset_fingerprint=DATASET_FINGERPRINT,
        source_commit=source_commit, source_sha256=source_sha256)
    path = write_run(output, run, rows, mode=mode)
    print(f'Ready: {len(rows)} questions, {count} GPU shards, batch starts at {BATCH_SIZE}. {path}', flush=True)
    return path


def measurement_sources():
    values = source_hashes()
    root = Path(__file__).resolve().parents[1]
    for relative in ('smolvlm_inference/model.py', 'smolvlm_inference/text.py', 'smolvlm_inference/cache.py',
                     'smolvlm_attention/collector.py', 'smolvlm_attention/data.py',
                     'smolvlm_attention/generated_runtime.py'):
        values[relative] = digest(root / relative)
    return values


def process_pending(rows, completed, measure, save, *, batch_size=BATCH_SIZE, progress=None):
    import gc
    import torch
    if batch_size < 1:
        raise ValueError('Batch must be positive')
    completed = set(completed)
    if not completed <= {r['sample_id'] for r in rows}:
        raise ValueError('Unknown completed sample')
    pending = sorted((row for row in rows if row['sample_id'] not in completed),
                     key=lambda row: (-len(row.get('input_ids', [])), row['sample_id']))
    cursor = 0
    while cursor < len(pending):
        batch = pending[cursor:cursor + batch_size]
        try:
            results = measure(batch)
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            batch_size //= 2
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if progress:
                progress(dict(stage='oom_retry', processed=len(completed), batch_size=batch_size))
            continue
        if [r['sample_id'] for r in results] != [r['sample_id'] for r in batch]:
            raise ValueError('Measured IDs differ from batch')
        for result in results:
            save(result)
            completed.add(result['sample_id'])
        cursor += len(batch)
        if progress:
            progress(dict(stage='measuring', processed=len(completed), batch_size=batch_size))
    return batch_size


def _pid_path(run, stage, shard):
    return Path(run['output_root']) / 'logs' / f'{stage}_{shard}.pid.json'


def _pid_alive(path):
    try:
        pid = json.loads(Path(path).read_text())['pid']
        os.kill(pid, 0)
        return True
    except (FileNotFoundError, OSError, ProcessLookupError):
        return False


def launch_stage(config_path, stage):
    from filelock import FileLock
    run = read_run(config_path)
    logs = Path(run['output_root']) / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    # Serialize PID/status publication across notebook kernels. The worker's
    # own shard lock alone is too late to prevent two competing launchers.
    with FileLock(str(logs / 'launch.lock'), timeout=0):
        return _launch_stage(config_path, stage)


def _launch_stage(config_path, stage):
    if stage not in WORKER_STAGES:
        raise ValueError('Unknown stage')
    path = Path(config_path).resolve()
    run = read_run(path)
    if measurement_sources() != run['measurement_sources'] or _library_versions() != run['versions']:
        raise ValueError('Measurement code/library versions changed; choose a new output directory')
    import torch
    if stage != 'localization' and torch.cuda.device_count() < run['shard_count']:
        raise RuntimeError('Not enough CUDA GPUs for this run topology')
    ids = [r['sample_id'] for r in input_rows(run)]
    prerequisite = 'baseline' if stage in ('causal', 'localization') else None
    if prerequisite and set(load_stage(run['output_root'], prerequisite, ids, run['identity'][prerequisite])) != set(ids):
        raise ValueError('Finish baseline stage first')
    logs = Path(run['output_root']) / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    for shard in range(run['shard_count']):
        key = (str(path), stage, shard)
        current = _PROCESSES.get(key)
        if current is not None and current.poll() is None:
            continue
        pid_path = _pid_path(run, stage, shard)
        if _pid_alive(pid_path):
            print(f'Reconnecting to {stage} shard {shard}.', flush=True)
            continue
        for other in WORKER_STAGES:
            if other != stage and _pid_alive(_pid_path(run, other, shard)):
                raise RuntimeError('Another measurement stage is running; wait before launching')
        status = logs / f'{stage}_{shard}.json'
        atomic_json(status, dict(status='starting', stage=stage, processed=0,
            total=min(2, len(shard_rows(run, shard))) if stage == 'preflight' else len(shard_rows(run, shard)),
            fingerprint=run['identity'][stage]))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), TOKENIZERS_PARALLELISM='false',
                   PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1')
        source = str(Path(__file__).resolve().parents[1])
        env['PYTHONPATH'] = source + os.pathsep + env.get('PYTHONPATH', '')
        with (logs / f'{stage}_{shard}.log').open('a', encoding='utf-8') as stream:
            process = subprocess.Popen([sys.executable, '-m', 'smolvlm_probing.worker', '--config', str(path),
                '--stage', stage, '--shard', str(shard)], stdout=stream, stderr=subprocess.STDOUT,
                env=env, cwd=source)
        _PROCESSES[key] = process
        atomic_json(pid_path, {'pid': process.pid})
    return config_path


def wait_stage(config_path, stage, *, poll_seconds=1):
    from tqdm.auto import tqdm
    path = Path(config_path).resolve()
    run = read_run(path)
    totals = [min(2, len(shard_rows(run, shard))) if stage == 'preflight' else len(shard_rows(run, shard))
              for shard in range(run['shard_count'])]
    bars = [tqdm(total=totals[shard], desc=f'{stage} shard {shard}', unit='question',
                 position=shard) for shard in range(run['shard_count'])]
    total = tqdm(total=sum(totals), desc=f'{stage} total', unit='question', position=len(bars))
    try:
        while True:
            finished, processed = True, 0
            for shard, bar in enumerate(bars):
                status_path = Path(run['output_root']) / 'logs' / f'{stage}_{shard}.json'
                status = json.loads(status_path.read_text(encoding='utf-8'))
                if status['fingerprint'] != run['identity'][stage]:
                    raise ValueError('Progress fingerprint differs')
                process = _PROCESSES.get((str(path), stage, shard))
                code = process.poll() if process is not None else None
                alive = code is None if process is not None else _pid_alive(_pid_path(run, stage, shard))
                if status['status'] == 'failed' or (code is not None and code != 0):
                    raise RuntimeError(f"{stage} shard {shard}: {status.get('error', code)}; log: {status_path.with_suffix('.log')}")
                if not alive and status['status'] != 'complete':
                    raise RuntimeError(f'Worker exited without complete status; read {status_path.with_suffix(".log")}')
                finished = finished and not alive and status['status'] == 'complete'
                bar.update(status['processed'] - bar.n)
                bar.set_postfix_str(f"{status.get('stage', stage)}, batch={status.get('batch_size', '-')}", refresh=False)
                processed += status['processed']
            total.update(processed - total.n)
            if finished:
                return
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print('Workers keep running. Rerun this wait cell to reconnect.', flush=True)
        raise
    finally:
        for bar in bars + [total]:
            bar.close()


def load_results(config_path, *, plot_source_commit=None, plot_source_sha256=None):
    from .localization import aggregate_localization
    run = read_run(config_path)
    rows = input_rows(run)
    ids = [r['sample_id'] for r in rows]
    loaded = {stage: load_stage(run['output_root'], stage, ids, run['identity'][stage]) for stage in STAGES}
    if any(set(records) != set(ids) for records in loaded.values()):
        raise ValueError('All stages must cover every selected question before aggregation')
    causal, localization, responses = [], [], []
    for row in rows:
        sid = row['sample_id']
        c, _ = loaded['causal'][sid]
        if c['status'] == 'complete':
            causal.append(c)
        base, arrays = loaded['baseline'][sid]
        if arrays['ranking'].shape != (5, 384) or arrays['target_maps'].ndim != 3 or arrays['target_maps'].shape[1:] != (6, 384):
            raise ValueError('Saved baseline geometry differs')
        responses.append({key: base.get(key) for key in (
            'sample_id', 'generated_ids', 'response', 'stop_reason', 'generation_status', 'score')})
        record, _ = loaded['localization'][sid]
        for partition in ('with_coordinates', 'without_coordinates'):
            targets = record['partitions'][partition]
            expected = row['localization_targets'][partition]
            if len(targets) != len(expected) or any(target['target'] != source for target, source in zip(targets, expected)):
                raise ValueError('Localization audit differs from the frozen target allowlist')
            for target in targets:
                if target['status'] == 'eligible':
                    target['attention_maps'] = arrays['target_maps'][target['map_index']].tolist()
            localization.append(dict(sample_id=sid, frame_id=row['frame_id'], partition=partition,
                question=row['question'], response=base['response'], camera_paths=row['camera_paths'], targets=targets))
    summary = aggregate_localization(localization)
    expected_totals = {partition: sum(len(row['localization_targets'][partition]) for row in rows)
                       for partition in ('with_coordinates', 'without_coordinates')}
    if any(summary[partition]['total_targets'] != count for partition, count in expected_totals.items()):
        raise ValueError('Localization target coverage incomplete')
    metadata = dict(measurement_fingerprint=run['fingerprint'], measurement_commit=run.get('source_commit'),
        measurement_source_sha256=run.get('source_sha256'), plot_commit=plot_source_commit,
        plot_source_sha256=plot_source_sha256, dataset_fingerprint=run['dataset_fingerprint'],
        input_questions=len(rows), causal_valid_questions=len(causal), causal_excluded_questions=len(rows)-len(causal),
        causal_exclusion_reasons={reason: sum(r[0].get('reason') == reason for r in loaded['causal'].values())
            for reason in sorted({r[0].get('reason') for r in loaded['causal'].values() if r[0].get('reason')})},
        max_new_tokens=64, k=32, random_repeats=3, seed=42,
        query='generated token representation including own key; not prediction-step attention',
        probability='exp(mean content-token log probability), scoring at predecessor positions',
        intervention='zero post-connector fused visual embeddings before first decoder layer; masks/positions unchanged',
        localization='global 384-token peak; correct camera and grid center inside bbox; exact literal eligible targets',
        limitations='Metadata freeze only; exact mention is not proof of semantic correctness. Zero embedding is not pixel deletion.')
    output = Path(run['output_root'])
    atomic_json(output / 'summary.json', summary)
    causal_summary = []
    for layer in (1, 7, 13, 19, 25):
        methods = {}
        for method in ('top', 'random', 'bottom'):
            conditions = [record['random'] if method == 'random' else
                next(item for item in record['layers'] if item['layer'] == layer)[method] for record in causal]
            methods[method] = dict(count=len(conditions), **{
                metric: dict(mean=float(np.mean([condition[metric] for condition in conditions])) if conditions else None,
                    median=float(np.median([condition[metric] for condition in conditions])) if conditions else None)
                for metric in ('comp', 'sgap')})
        causal_summary.append(dict(layer=layer, methods=methods))
    atomic_json(output / 'causal_summary.json', causal_summary)
    atomic_json(output / 'figure_metadata.json', metadata)
    atomic_json(output / 'causal_results.json', causal)
    atomic_json(output / 'responses.json', responses)
    # Numeric maps are already retained in baseline NPZ files; audit stays small.
    audit = [{**row, 'targets': [{k: v for k, v in target.items() if k != 'attention_maps'}
                for target in row['targets']]} for row in localization if row['targets']]
    atomic_json(output / 'localization_audit.json', audit)
    return dict(causal=causal, localization=localization, summary=summary, metadata=metadata)
