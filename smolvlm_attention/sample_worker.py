"""One GPU worker writing per-question attention without sample/layer reduction."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch

from .sample_store import (ShardStore, atomic_json, digest, input_rows, read_run, shard_bounds)


def process_pending(rows, store, measure, *, commit_every=100, progress=None):
    if commit_every < 1:
        raise ValueError('commit_every must be positive')
    pending = sorted((r for r in rows if r['row_index'] not in store.completed),
                     key=lambda r: (-len(r['input_ids']), r['row_index']))
    cursor, successful, batch_size = 0, 0, store.batch_size
    def report(stage):
        if progress:
            progress(dict(stage=stage, processed=store.processed,
                          committed=len(store.completed), batch_size=batch_size))
    report('measuring')
    try:
        while cursor < len(pending):
            selected = pending[cursor:cursor + batch_size]
            try:
                values = measure(selected)
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size //= 2
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Persist reduced batch even if no sample has yet succeeded.
                store.commit(batch_size=batch_size)
                report('oom_retry')
                continue
            if isinstance(values, torch.Tensor):
                if values.device.type != 'cpu':
                    raise ValueError('Collector must return CPU attention')
                values = values.numpy()
            store.write([r['row_index'] for r in selected], values)
            del values
            cursor += len(selected)
            successful += 1
            if successful % commit_every == 0:
                store.commit(batch_size=batch_size)
            report('measuring')
        store.commit(batch_size=batch_size)
        report('committed')
    except Exception:
        # Successful batches remain usable after a later measurement failure.
        store.commit(batch_size=batch_size)
        raise
    return batch_size


def preflight(model, rows, make, *, progress=None):
    from .collector import collect_last_query_attention, reference_last_query_attention
    ordered = sorted(rows, key=lambda r: len(r['input_ids']))
    bounded = [r for r in ordered if len(r['input_ids']) <= 1024]
    if not bounded:
        raise ValueError('No complete prompt fits the 1024-token native-oracle budget; no prompt was truncated')
    probes = [bounded[0], bounded[-1]] if len(bounded) > 1 else [bounded[0]]
    singles = []
    for row in probes:
        batch = make([row])
        actual = collect_last_query_attention(model, batch)
        expected = reference_last_query_attention(model, batch)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-4)
        singles.append(actual)
        del batch, expected
    if len(probes) == 2:
        actual = collect_last_query_attention(model, make(probes))
        torch.testing.assert_close(actual, torch.cat(singles), atol=1e-6, rtol=1e-4)
    if ordered[-1] not in probes:
        # Longest real prompt is still collected without truncation, using the
        # production path rather than retaining 30 quadratic native matrices.
        collect_last_query_attention(model, make([ordered[-1]]))
    if progress:
        progress(dict(stage='native_parity_passed', oracle_prompt_lengths=[len(r['input_ids']) for r in probes],
                      longest_prompt_length=len(ordered[-1]['input_ids'])))


def run_worker(config_path, shard):
    from filelock import FileLock
    from smolvlm_inference.cache import CAMERA_ORDER
    from smolvlm_inference.model import load_model, processor_policy
    from .data import make_batch, open_validated_cache
    from .collector import collect_last_query_attention
    from .sample_runtime import assert_measurement_compatible
    run = read_run(config_path)
    output = Path(run['output_root'])
    logs = output / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()
    start, end = shard_bounds(run['count'], run['shard_count'], shard)
    status = dict(status='running', stage='starting', total=end-start, processed=0, committed=0,
                  fingerprint=run['fingerprint'], batch_size=32)
    def progress(event):
        status.update(event, elapsed_seconds=round(time.monotonic()-start_time, 2))
        atomic_json(logs / f'samples_{shard}.status.json', status)
        if status['stage'] != 'measuring' or status['processed'] % 100 == 0:
            print(json.dumps(status), flush=True)
    with FileLock(str(logs / f'samples_{shard}.worker.lock'), timeout=0):
        try:
            assert_measurement_compatible(run)
            if not torch.cuda.is_available():
                raise RuntimeError('This worker requires CUDA')
            torch.set_num_threads(1)
            torch.set_float32_matmul_precision('highest')
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            device = torch.device('cuda:0')
            selected = input_rows(run)[start:end]
            store = ShardStore(output / 'shards' / str(shard), run['fingerprint'],
                               list(range(start, end)), run['geometry'])
            progress(dict(stage='loading_model', processed=store.processed,
                          committed=len(store.completed), batch_size=store.batch_size))
            spec = run['artifacts']['model']
            if digest(spec['checkpoint']) != run['identity']['checkpoint_sha256']:
                raise ValueError('Checkpoint changed')
            reader = open_validated_cache(run['cache_descriptor'])
            model = load_model(spec['checkpoint'], device=device, expected_prompt_camera_order=CAMERA_ORDER)
            if (model.native.config.text_config.num_hidden_layers != 30 or model.visual_token_len != 384
                    or model.geometry != reader.manifest['geometry']
                    or processor_policy(model.processor) != reader.manifest['processor_policy']
                    or model.native.config.image_token_id != run['identity']['data']['image_token_id']):
                raise ValueError('Worker model/cache geometry or processor mismatch')
            def make(rows):
                batch = make_batch(reader, rows, model.processor, CAMERA_ORDER)
                return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            progress(dict(stage='native_parity'))
            with torch.inference_mode():
                preflight(model, selected, make, progress=progress)
                process_pending(selected, store, lambda rows: collect_last_query_attention(model, make(rows)),
                                progress=progress)
            if store.completed != set(range(start, end)):
                raise ValueError('Missing test samples')
            progress(dict(status='complete', stage='complete', processed=end-start, committed=end-start,
                gpu=torch.cuda.get_device_name(device), peak_vram_bytes=torch.cuda.max_memory_reserved(device)))
        except Exception as exc:
            progress(dict(status='failed', stage='failed', error=repr(exc)))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--shard', type=int, required=True)
    args = parser.parse_args()
    run_worker(args.config, args.shard)


if __name__ == '__main__':
    main()
