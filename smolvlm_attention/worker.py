"""One independent FP32 model worker per GPU; no distributed training runtime."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch
import transformers

from smolvlm_inference.cache import CAMERA_ORDER, canonical, digest
from smolvlm_inference.model import load_model, processor_policy
from .aggregate import AttentionAccumulator
from .collector import collect_last_query_attention, reference_last_query_attention
from .config import QUERY_POLICY
from .data import load_prepared_test, make_batch, open_validated_cache


def process_pending(rows, accumulator, measure, *, batch_size, device, checkpoint=None,
                    progress=None):
    """Commit a batch only after successful measurement; OOM retries never count twice."""
    pending = set(accumulator.pending_ids)
    remaining = sorted((r for r in rows if r['sample_id'] in pending),
                       key=lambda r: (-len(r['input_ids']), r['sample_id']))
    cursor, successful = 0, 0
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    while cursor < len(remaining):
        batch = remaining[cursor:cursor + batch_size]
        try:
            attention = measure(batch)
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            gc.collect()
            if torch.device(device).type == 'cuda':
                torch.cuda.empty_cache()
            if progress:
                progress(dict(event='oom_retry', batch_size=batch_size,
                              processed=accumulator.count))
            continue
        accumulator.add([r['sample_id'] for r in batch], attention)
        del attention
        cursor += len(batch)
        successful += 1
        if progress:
            progress(dict(event='batch', processed=accumulator.count, batch_size=len(batch)))
        if checkpoint and successful % 100 == 0:
            accumulator.save(checkpoint)
    if checkpoint:
        accumulator.save(checkpoint)
    return batch_size


def tune_batch(rows, measure, device):
    """Profile longest prompts, keeping at least 20% VRAM free."""
    if torch.device(device).type != 'cuda':
        return 1
    longest = sorted(rows, key=lambda r: -len(r['input_ids']))[:8]
    selected = 0
    total = torch.cuda.get_device_properties(device).total_memory
    for size in (1, 2, 4, 8):
        if size > len(longest):
            break
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            result = measure(longest[:size])
            del result
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_reserved(device)
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            break
        if size == 1:
            selected = 1
        if peak > total * .8:
            break
        selected = size
    if selected == 0:
        raise RuntimeError('FP32 batch 1 does not fit this GPU')
    return selected


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def run_worker(config_path, model_name):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    spec = config['models'][model_name]
    root = Path(config['output_root'])
    output = root / 'data' / model_name
    output.mkdir(parents=True, exist_ok=True)
    log_path = root / 'logs' / f'{model_name}.jsonl'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    def progress(event):
        event = dict(event, elapsed_seconds=round(time.time() - start, 2), model=model_name)
        with log_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event) + '\n')
        if event['event'] != 'batch' or event['processed'] % 100 == 0:
            print(json.dumps(event), flush=True)
    status = output / 'metadata.json'
    _write_json(status, dict(status='running', model=model_name))
    try:
        if transformers.__version__ != '4.46.3':
            raise RuntimeError('Transformers 4.46.3 is required')
        if not torch.cuda.is_available():
            raise RuntimeError('This worker requires a Kaggle GPU')
        device = torch.device('cuda:0')  # CUDA_VISIBLE_DEVICES isolates the assigned GPU.
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        reader = open_validated_cache(config['cache_descriptor'])
        prepared = load_prepared_test(spec['prepared_test'])
        rows = prepared['records']
        model = load_model(spec['checkpoint'], device=device,
                           expected_prompt_camera_order=spec['camera_order'])
        identity = dict(backend='smolvlm_no_pruner', version=1, model_id=model.model_id,
            revision=model.model_revision, processor_policy=processor_policy(model.processor),
            geometry=model.geometry, storage_dtype='float16', compute_dtype='float32',
            camera_order=CAMERA_ORDER)
        if any(reader.manifest.get(k) != v for k, v in identity.items()):
            raise ValueError('Worker model does not match validated cache identity')
        if prepared['camera_order'] != spec['camera_order']:
            raise ValueError('Prepared prompts do not match checkpoint camera order')
        if (prepared['manifest_fingerprint'] != reader.manifest['fingerprint'] or
                prepared['test_sha256'] != reader.manifest['prepared_hashes']['test.jsonl'] or
                prepared['image_token_id'] != model.native.config.image_token_id or
                prepared['image_seq_len'] != model.geometry['visual_tokens_per_view']):
            raise ValueError('Prepared prompts do not match validated test data/model')
        if len(rows) != config['expected_test_count']:
            raise ValueError('Incomplete test set')
        def batch_on_device(selected):
            batch = make_batch(reader, selected, model.processor, model.prompt_camera_order)
            return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
        def measure(selected):
            return collect_last_query_attention(model, batch_on_device(selected))
        # Full native attentions are only retained for single-sample preflight checks.
        for row in sorted(rows, key=lambda r: len(r['input_ids']))[:2]:
            batch = batch_on_device([row])
            actual = collect_last_query_attention(model, batch)
            reference = reference_last_query_attention(model, batch)
            torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-4)
            del batch, actual, reference
        progress(dict(event='native_parity_passed', samples=min(2, len(rows))))
        shape = (model.native.config.text_config.num_hidden_layers, model.visual_token_len)
        if shape != (30, 384):
            raise ValueError(f'Unexpected production attention geometry: {shape}')
        fingerprint_data = dict(algorithm=QUERY_POLICY, model=spec['repo'],
            revision=spec['revision'], checkpoint_sha256=digest(spec['checkpoint']),
            order=spec['camera_order'], cache=reader.manifest['fingerprint'],
            prepared_sha256=digest(spec['prepared_test']), precision='float32',
            transformers=transformers.__version__,
            source_archive_sha256=config.get('source_archive_sha256'))
        fingerprint = hashlib.sha256(canonical(fingerprint_data)).hexdigest()
        ids = [r['sample_id'] for r in rows]
        state = output / 'attention.npz'
        accumulator = (AttentionAccumulator.load(state, shape, fingerprint, ids)
                       if state.exists() else AttentionAccumulator(shape, fingerprint, ids))
        batch_size = tune_batch(rows, measure, device) if accumulator.pending_ids else 1
        progress(dict(event='batch_selected', batch_size=batch_size, processed=accumulator.count))
        final_batch = process_pending(rows, accumulator, measure, batch_size=batch_size,
            device=device, checkpoint=state, progress=progress)
        if accumulator.count != len(ids) or accumulator.pending_ids:
            raise RuntimeError('Measurement did not cover every test sample')
        metadata = dict(status='complete', model=model_name, **fingerprint_data,
            count=accumulator.count, shape=list(shape), fingerprint=fingerprint,
            checkpoint_metadata=model.checkpoint_metadata,
            cache_descriptor=config['cache_descriptor'], source_commit=config.get('source_commit'),
            prepared_repo=config['prepared_repo'], prepared_revision=config['prepared_revision'],
            python=platform.python_version(), torch=torch.__version__,
            gpu=torch.cuda.get_device_name(device), physical_gpu=spec['gpu'],
            final_batch_size=final_batch, elapsed_seconds=round(time.time()-start, 2),
            peak_vram_bytes=torch.cuda.max_memory_reserved(device),
            histogram_population='384 head-and-sample-averaged visual positions per layer')
        _write_json(status, metadata)
        progress(dict(event='complete', processed=accumulator.count))
    except Exception as exc:
        _write_json(status, dict(status='failed', model=model_name, error=repr(exc)))
        progress(dict(event='failed', error=repr(exc)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', required=True, choices=['base', 'reverse'])
    args = parser.parse_args()
    from filelock import FileLock
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    lock = Path(config['output_root']) / 'logs' / f'{args.model}.worker.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock), timeout=0):
        run_worker(args.config, args.model)


if __name__ == '__main__':
    main()
