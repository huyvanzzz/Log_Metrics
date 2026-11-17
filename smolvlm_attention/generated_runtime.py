"""Kaggle orchestration for greedy-generation attention, independent of prompt measurements."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoProcessor, GenerationConfig
from tqdm.auto import tqdm

from smolvlm_inference.cache import CAMERA_ORDER, canonical, digest, records, record_key
from smolvlm_inference.model import (MODEL_ID, MODEL_REVISION, load_model, _configure_processor,
                                    _geometry, processor_policy)
from smolvlm_inference.text import CachedBatchCollator, chat_messages
from .config import MODELS, PREPARED_REPO, PREPARED_REVISION, CACHE_DATASET, CACHE_FINGERPRINT, TEST_COUNT
from .data import _json_write, validate_cache, open_validated_cache, make_batch

POLICY = 'generated_positions_including_self_excluding_eos_mean_heads_tokens_samples_v1'
_PROCESSES = {}


def generation_config(native, max_new_tokens=32):
    """Use only native special IDs; never inherit sampling or repetition settings."""
    previous = native.generation_config
    return GenerationConfig(bos_token_id=previous.bos_token_id, eos_token_id=previous.eos_token_id,
        pad_token_id=previous.pad_token_id, max_new_tokens=max_new_tokens,
        do_sample=False, num_beams=1, num_return_sequences=1, use_cache=True,
        repetition_penalty=1.0, min_length=0, output_attentions=False,
        output_hidden_states=False, return_dict_in_generate=False)


def process_batches(rows, completed, measure, save, *, batch_size, progress=None):
    """Only persist successful measurements; durable per-sample files support partial-batch resume."""
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    completed = set(completed)
    if not completed <= {r['sample_id'] for r in rows}:
        raise ValueError('Unknown completed sample')
    remaining = sorted((r for r in rows if r['sample_id'] not in completed),
                       key=lambda r: (-len(r['input_ids']), r['sample_id']))
    cursor = 0
    while cursor < len(remaining):
        batch = remaining[cursor:cursor + batch_size]
        try:
            measured = measure(batch)
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if progress:
                progress(dict(stage='oom_retry', processed=len(completed), batch_size=batch_size))
            continue
        if [r['sample_id'] for r in measured] != [r['sample_id'] for r in batch]:
            raise ValueError('Measurement IDs differ from requested batch')
        for result in measured:
            save(result)
            completed.add(result['sample_id'])
        cursor += len(batch)
        if progress:
            progress(dict(stage='measuring', processed=len(completed), batch_size=batch_size))
    return batch_size


def resolve_generated_artifacts(output_root, cache_input_dir=''):
    """Resolve only the base checkpoint and pinned prepared files, never copy the cache."""
    from huggingface_hub import hf_hub_download
    import kagglehub
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepared = output / 'prepared_data'
    for filename in ('manifest.json', 'train.jsonl', 'val.jsonl', 'test.jsonl'):
        hf_hub_download(repo_id=PREPARED_REPO, repo_type='dataset', revision=PREPARED_REVISION,
                        filename=filename, local_dir=prepared, token=False)
    spec = dict(MODELS['base'])
    spec['checkpoint'] = str(Path(hf_hub_download(repo_id=spec['repo'], revision=spec['revision'],
        filename='finetune_best.pt', token=False)).resolve())
    cache_root = Path(cache_input_dir) if cache_input_dir else Path(kagglehub.dataset_download(CACHE_DATASET))
    candidates = [p.parent for p in cache_root.rglob('features.float16.memmap')
                  if (p.parent / 'manifest.json').is_file() and (p.parent / 'index.json').is_file()]
    if len(candidates) != 1:
        raise ValueError(f'Expected exactly one cache in {cache_root}; found {len(candidates)}')
    return dict(output_root=str(output), prepared_dir=str(prepared.resolve()),
                cache_dir=str(candidates[0].resolve()), model=spec)


def _cache_receipt(artifacts, receipt_path=None):
    output = Path(artifacts['output_root'])
    candidates = [Path(receipt_path)] if receipt_path else [output / 'validated_cache.json',
                        Path('/kaggle/working/attention_outputs/validated_cache.json')]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            reader = open_validated_cache(path)
            if (reader.directory != Path(artifacts['cache_dir']).resolve() or
                    reader.manifest['fingerprint'] != CACHE_FINGERPRINT):
                continue
            # A receipt may refer to an earlier notebook's identical prepared files.
            if any(digest(Path(artifacts['prepared_dir']) / name) != value
                   for name, value in reader.manifest['prepared_hashes'].items()):
                continue
            print(f'Reusing fully validated cache receipt: {path}', flush=True)
            return path.resolve()
        except (ValueError, OSError, KeyError, TypeError):
            continue
    print('Validating prepared-data hashes and full feature cache (CPU/disk).', flush=True)
    bar = None
    stage = None
    def progress(event):
        nonlocal bar, stage
        if stage != event['stage']:
            if bar is not None:
                bar.close()
            stage = event['stage']
            bar = tqdm(total=event['total'], desc=f'Validate {stage}',
                       unit='B' if stage == 'features' else 'rows', unit_scale=stage == 'features')
        bar.update(event['completed'] - bar.n)
    try:
        return validate_cache(artifacts['cache_dir'], artifacts['prepared_dir'], output,
                              CACHE_FINGERPRINT, progress=progress).resolve()
    finally:
        if bar is not None:
            bar.close()


def prepare_generated_run(artifacts, *, source_archive_sha256, source_commit=None,
                          sample_count=1000, seed=42, max_new_tokens=32, receipt_path=None):
    from .generated_state import select_samples
    if sample_count != 1000 or max_new_tokens not in (32, 64):
        raise ValueError('Production run requires 1000 samples and a 32/64 token budget')
    descriptor = _cache_receipt(artifacts, receipt_path)
    reader = open_validated_cache(descriptor)
    if reader.counts['test'] != TEST_COUNT:
        raise ValueError('Expected the complete pinned test split')
    print('Loading pinned processor and selecting 1000 test questions.', flush=True)
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=False)
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=False)
    _configure_processor(processor, config)
    identity = dict(model_id=MODEL_ID, revision=MODEL_REVISION, processor_policy=processor_policy(processor),
                    geometry=_geometry(config), compute_dtype='float32', storage_dtype='float16')
    if any(reader.manifest.get(k) != v for k, v in identity.items()):
        raise ValueError('Processor/model does not match validated cache')
    selected = select_samples(list(records(reader.prepared / 'test.jsonl')), sample_count, seed)
    collator = CachedBatchCollator(processor, image_seq_len=64, max_length=config.text_config.max_position_embeddings,
                                   max_new_tokens=max_new_tokens, prompt_camera_order=CAMERA_ORDER)
    rows = []
    for row in tqdm(selected, desc='Prepare selected prompts', unit='question'):
        question = str(row['question'])
        if str(processor.image_token) in question:
            raise ValueError('Question contains reserved image placeholder')
        prompt = processor.apply_chat_template(chat_messages(question), tokenize=False, add_generation_prompt=True)
        ids = processor.tokenizer(collator._expand(prompt), add_special_tokens=True, truncation=False)['input_ids']
        if len(ids) + max_new_tokens > config.text_config.max_position_embeddings or ids.count(
                processor.tokenizer.convert_tokens_to_ids(str(processor.image_token))) != 384:
            raise ValueError('Selected prompt exceeds budget or has wrong image-token count')
        rows.append(dict(sample_id=row['sample_id'], eval_id=row['eval_id'], question=question,
                         feature_index=reader.index[record_key(row)], input_ids=ids))
    output = Path(artifacts['output_root'])
    sample_path = _json_write(output / 'selected_samples.json', dict(seed=seed, count=len(rows), records=rows))
    print('Hashing base checkpoint for reproducible resume.', flush=True)
    identity = dict(algorithm=POLICY, model=artifacts['model']['repo'], revision=artifacts['model']['revision'],
        checkpoint_sha256=digest(artifacts['model']['checkpoint']), cache=reader.manifest['fingerprint'],
        prepared_hashes=reader.manifest['prepared_hashes'], selected_sha256=digest(sample_path), seed=seed,
        sample_count=sample_count, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
        repetition_penalty=1., eos_policy='exclude', query_includes_self=True,
        precision='float32', transformers='4.46.3', source_archive_sha256=source_archive_sha256,
        camera_order=CAMERA_ORDER)
    fingerprint = hashlib.sha256(canonical(identity)).hexdigest()
    run = dict(artifacts, cache_descriptor=str(descriptor), selected_samples=str(sample_path.resolve()),
               identity=identity, fingerprint=fingerprint, source_commit=source_commit,
               prepared_repo=PREPARED_REPO, prepared_revision=PREPARED_REVISION,
               shard_count=min(2, max(1, torch.cuda.device_count())))
    path = _json_write(output / 'generated_run.json', run)
    print(f'Ready: {len(rows)} questions, max_new_tokens={max_new_tokens}; {path}', flush=True)
    return path


@torch.inference_mode()
def measure_generated_batch(model, batch, rows, *, max_new_tokens=32):
    from .generated_collector import collect_generated_attention
    from .generated_state import trim_generation
    device = next(model.native.parameters()).device
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
             for key, value in batch.items()}
    config = generation_config(model.native, max_new_tokens)
    generated = model.generate_from_features(input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'], visual_features=batch['visual_features'],
        generation_config=config).cpu().tolist()
    content, reasons = trim_generation(generated, config.eos_token_id)
    measured = collect_generated_attention(model, batch, content)
    return [dict(sample_id=row['sample_id'], visual=measured['visual'][i].numpy(),
                 groups=measured['groups'][i].numpy(), generated_ids=generated[i], content_ids=content[i],
                 response=model.processor.tokenizer.decode(content[i], skip_special_tokens=True),
                 stop_reason=reasons[i], token_count=len(content[i])) for i, row in enumerate(rows)]


def _read_run(path):
    run = json.loads(Path(path).read_text(encoding='utf-8'))
    if hashlib.sha256(canonical(run['identity'])).hexdigest() != run['fingerprint']:
        raise ValueError('Run fingerprint mismatch')
    if digest(run['selected_samples']) != run['identity']['selected_sha256']:
        raise ValueError('Selected samples changed since preparation')
    return run


def _shard_rows(run, shard):
    rows = json.loads(Path(run['selected_samples']).read_text(encoding='utf-8'))['records']
    if not 0 <= shard < run['shard_count']:
        raise ValueError('Invalid shard')
    return list(np.array_split(np.asarray(rows, dtype=object), run['shard_count'])[shard])


def run_generated_worker(config_path, shard):
    from .generated_collector import collect_generated_attention, reference_generated_attention
    from .generated_state import load_samples, save_sample
    run = _read_run(config_path)
    output = Path(run['output_root'])
    folder = output / 'data' / f'shard_{shard}'
    folder.mkdir(parents=True, exist_ok=True)
    rows = _shard_rows(run, shard)
    ids = [row['sample_id'] for row in rows]
    status_path = output / 'logs' / f'shard_{shard}.json'
    start = time.monotonic()
    status = dict(status='running', stage='starting', processed=0, total=len(rows), shard=shard,
                  fingerprint=run['fingerprint'])
    def progress(event):
        status.update(event)
        status['elapsed_seconds'] = round(time.monotonic() - start, 2)
        _json_write(status_path, status)
        print(json.dumps(status), flush=True)
    try:
        progress({})
        if not torch.cuda.is_available() or transformers.__version__ != '4.46.3':
            raise RuntimeError('CUDA and transformers==4.46.3 are required')
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device('cuda:0')
        completed = load_samples(folder, ids, run['fingerprint'])
        progress(dict(stage='loading_model', processed=len(completed)))
        if len(completed) == len(rows):
            progress(dict(status='complete', stage='complete'))
            return
        reader = open_validated_cache(run['cache_descriptor'])
        if reader.manifest['fingerprint'] != run['identity']['cache']:
            raise ValueError('Worker cache differs from prepared run')
        if digest(run['model']['checkpoint']) != run['identity']['checkpoint_sha256']:
            raise ValueError('Worker checkpoint differs from prepared run')
        model = load_model(run['model']['checkpoint'], device=device, expected_prompt_camera_order=CAMERA_ORDER)
        if (processor_policy(model.processor) != reader.manifest['processor_policy'] or
                model.geometry != reader.manifest['geometry'] or
                model.native.config.text_config.num_hidden_layers != 30 or model.visual_token_len != 384):
            raise ValueError('Worker model/processor differs from validated cache')
        budget = run['identity']['max_new_tokens']
        def batch_for(selected):
            return make_batch(reader, selected, model.processor, CAMERA_ORDER)
        def measure(selected):
            return measure_generated_batch(model, batch_for(selected), selected, max_new_tokens=budget)
        progress(dict(stage='native_parity'))
        probe_row = min(rows, key=lambda r: len(r['input_ids']))
        probe_batch = batch_for([probe_row])
        probe = measure_generated_batch(model, probe_batch, [probe_row], max_new_tokens=min(8, budget))[0]
        # Immediate EOS still receives a nonempty numerical oracle check using a valid vocabulary ID.
        content = probe['content_ids'] or [model.processor.tokenizer.bos_token_id]
        actual = collect_generated_attention(model, probe_batch, [content])
        expected = reference_generated_attention(model, probe_batch, [content])
        for key in ('visual', 'groups'):
            torch.testing.assert_close(actual[key], expected[key], atol=1e-6, rtol=1e-4)
        del actual, expected, probe_batch, probe
        batch_size = 16
        progress(dict(stage='measuring', batch_size=batch_size))
        final_size = process_batches(rows, set(completed), measure,
            lambda record: save_sample(folder, record, run['fingerprint']),
            batch_size=batch_size, progress=progress)
        verified = load_samples(folder, ids, run['fingerprint'])
        if set(verified) != set(ids):
            raise ValueError('Incomplete shard coverage')
        progress(dict(status='complete', stage='complete', processed=len(ids), final_batch_size=final_size,
            gpu=torch.cuda.get_device_name(device), peak_vram_bytes=torch.cuda.max_memory_reserved(device),
            torch=torch.__version__, python=platform.python_version(), checkpoint_metadata=model.checkpoint_metadata))
    except Exception as exc:
        progress(dict(status='failed', stage='failed', error=repr(exc)))
        raise


def launch_generated_workers(config_path):
    """Launch one base-model worker per shard, without blocking notebook progress display."""
    from .runtime import _pid_alive
    path = Path(config_path).resolve()
    run = _read_run(path)
    if torch.cuda.device_count() < run['shard_count']:
        raise RuntimeError('Not enough GPUs for prepared shard count')
    logs = Path(run['output_root']) / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    for shard in range(run['shard_count']):
        key = (str(path), shard)
        existing = _PROCESSES.get(key)
        if existing is not None and existing.poll() is None:
            processes.append(existing)
            continue
        pidfile = logs / f'shard_{shard}.pid.json'
        if pidfile.is_file():
            pid = json.loads(pidfile.read_text())['pid']
            if _pid_alive(pid):
                raise RuntimeError(f'Shard {shard} PID {pid} is still alive; do not launch a duplicate')
        _json_write(logs / f'shard_{shard}.json', dict(status='starting', stage='starting',
            processed=0, total=len(_shard_rows(run, shard)), fingerprint=run['fingerprint']))
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), TOKENIZERS_PARALLELISM='false',
                           OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1')
        environment['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]) + os.pathsep + environment.get('PYTHONPATH', '')
        kwargs = {'start_new_session': True} if os.name != 'nt' else {'creationflags': subprocess.CREATE_NO_WINDOW}
        with (logs / f'shard_{shard}.log').open('a', encoding='utf-8') as stream:
            process = subprocess.Popen([sys.executable, '-m', 'smolvlm_attention.generated_runtime',
                '--config', str(path), '--shard', str(shard)], env=environment, stdout=stream,
                stderr=subprocess.STDOUT, **kwargs)
        _PROCESSES[key] = process
        _json_write(pidfile, dict(pid=process.pid))
        processes.append(process)
        print(f'Base shard {shard}: physical GPU {shard}, PID {process.pid}', flush=True)
    return processes


def wait_generated_workers(config_path):
    path = Path(config_path).resolve()
    run = _read_run(path)
    processes = [_PROCESSES.get((str(path), shard)) for shard in range(run['shard_count'])]
    if any(process is None for process in processes):
        raise RuntimeError('Run launch_generated_workers first in this kernel')
    bars = [tqdm(total=len(_shard_rows(run, shard)), desc=f'Base GPU {shard}', unit='question', position=shard)
            for shard in range(run['shard_count'])]
    total = tqdm(total=run['identity']['sample_count'], desc='Total', unit='question', position=len(bars))
    try:
        while True:
            processed = 0
            for shard, (process, bar) in enumerate(zip(processes, bars)):
                status_path = Path(run['output_root']) / 'logs' / f'shard_{shard}.json'
                status = json.loads(status_path.read_text(encoding='utf-8'))
                if status['fingerprint'] != run['fingerprint']:
                    raise ValueError('Progress status belongs to another run')
                count = status['processed']
                bar.update(count - bar.n)
                bar.set_postfix_str(status['stage'], refresh=False)
                processed += count
                code = process.poll()
                if status['status'] == 'failed' or (code is not None and code != 0):
                    raise RuntimeError(f'Shard {shard} failed: {status.get("error", code)}. Read {status_path.with_suffix(".log")}')
            total.update(processed - total.n)
            if all(process.poll() is not None for process in processes):
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print('Workers keep running. Rerun this wait cell to reconnect.', flush=True)
        raise
    finally:
        for bar in bars + [total]:
            bar.close()
    print('All selected questions processed. Run the aggregation cell next.', flush=True)


def load_generated_results(config_path):
    from .generated_state import load_samples, aggregate_samples
    run = _read_run(config_path)
    output = Path(run['output_root'])
    combined, shards = {}, []
    for shard in range(run['shard_count']):
        ids = [row['sample_id'] for row in _shard_rows(run, shard)]
        status = json.loads((output / 'logs' / f'shard_{shard}.json').read_text(encoding='utf-8'))
        if status['status'] != 'complete' or status['fingerprint'] != run['fingerprint']:
            raise ValueError('Shard is incomplete or belongs to another run')
        loaded = load_samples(output / 'data' / f'shard_{shard}', ids, run['fingerprint'])
        if set(loaded) != set(ids) or set(combined) & set(loaded):
            raise ValueError('Incomplete or duplicate shard coverage')
        combined.update(loaded)
        shards.append(status)
    selected = json.loads(Path(run['selected_samples']).read_text(encoding='utf-8'))['records']
    ids = [row['sample_id'] for row in selected]
    result = aggregate_samples(combined, ids)
    metadata = dict(run['identity'], fingerprint=run['fingerprint'], source_commit=run.get('source_commit'),
        processed_count=result['processed_count'], valid_count=result['valid_count'], empty_count=result['empty_count'],
        shard_count=run['shard_count'], shards=shards, averaging='heads -> generated content tokens -> equal samples',
        groups=['Image', 'Prompt text', 'Generated (including self)'], standard_deviation='across sample means, ddof=1',
        histogram_population='384 averaged image-token positions per layer',
        caption='Greedy base model; EOS excluded. Shading is mean +/- 1 sample SD.')
    _json_write(output / 'figure_metadata.json', metadata)
    # A single small aggregate file allows plots to be regenerated without model inference.
    temporary = output / 'summary.npz.tmp'
    with temporary.open('wb') as stream:
        np.savez(stream, **result)
    temporary.replace(output / 'summary.npz')
    temporary = output / 'responses.jsonl.tmp'
    with temporary.open('w', encoding='utf-8') as stream:
        for row in selected:
            response = {key: value for key, value in combined[row['sample_id']].items() if key not in ('visual', 'groups')}
            stream.write(json.dumps(dict(response, question=row['question'], eval_id=row['eval_id']), ensure_ascii=False) + '\n')
    temporary.replace(output / 'responses.jsonl')
    print({key: result[key] for key in ('processed_count', 'valid_count', 'empty_count')})
    return result, metadata


def main():
    from filelock import FileLock
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--shard', required=True, type=int)
    args = parser.parse_args()
    run = _read_run(args.config)
    lock = Path(run['output_root']) / 'logs' / f'shard_{args.shard}.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock), timeout=0):
        run_generated_worker(args.config, args.shard)


if __name__ == '__main__':
    main()
