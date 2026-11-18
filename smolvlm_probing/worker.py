"""One stage/shard subprocess. Only inference stages load a model."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from .config import BATCH_SIZE, CAMERA_ORDER, LAYER_GROUPS, RANKING_LAYERS
from .runtime import (_library_versions, _pid_path, input_rows, measurement_sources,
                      process_pending, read_run, shard_rows)
from .state import atomic_json, digest, load_stage, save_record


def trim_response(tokenizer, raw_ids, image_token_id, *, eos_token_id=None):
    eos = tokenizer.eos_token_id if eos_token_id is None else eos_token_id
    eos = set(eos if isinstance(eos, (tuple, list)) else [eos])
    end = next((i for i, token in enumerate(raw_ids) if token in eos), len(raw_ids))
    ids = list(map(int, raw_ids[:end]))
    special = set(tokenizer.all_special_ids)
    valid = any(token not in special for token in ids)
    status = 'invalid_multimodal_generation' if image_token_id in ids else ('valid' if valid else 'empty')
    return dict(generated_ids=ids, raw_generated_ids=list(map(int, raw_ids)),
        special_token_mask=[token in special for token in ids],
        response=tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
        stop_reason='eos' if end < len(raw_ids) else 'length', generation_status=status)


def decisions_for(row, tokenizer, generation):
    from .localization import match_targets
    return {partition: match_targets(tokenizer, generation['generated_ids'], targets,
                generation_status=generation['generation_status'])
            for partition, targets in row['localization_targets'].items()}


def _flatten_decisions(partitions):
    return [decision for partition in ('with_coordinates', 'without_coordinates') for decision in partitions[partition]]


def _batch(reader, rows, processor, device):
    from smolvlm_attention.data import make_batch
    import torch
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in make_batch(reader, rows, processor, CAMERA_ORDER).items()}


def _inference_context(run, rows):
    import torch
    import transformers
    from smolvlm_attention.data import open_validated_cache
    from smolvlm_inference.model import load_model, processor_policy
    if not torch.cuda.is_available() or transformers.__version__ != '4.46.3':
        raise RuntimeError('CUDA and transformers==4.46.3 are required')
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reader = open_validated_cache(run['cache_descriptor'])
    artifact = run['artifacts']['model']
    if digest(artifact['checkpoint']) != run['checkpoint_sha256']:
        raise ValueError('Checkpoint changed after preparation')
    model = load_model(artifact['checkpoint'], device='cuda:0', expected_prompt_camera_order=CAMERA_ORDER)
    if (model.visual_token_len != 384 or model.native.config.text_config.num_hidden_layers != 30
            or model.geometry != reader.manifest['geometry']
            or processor_policy(model.processor) != reader.manifest['processor_policy']):
        raise ValueError('Worker model/processor differs from validated cache')
    return model, reader


def run_worker(config_path, stage, shard):
    from .localization import evaluate_targets
    run = read_run(config_path)
    if run['measurement_sources'] != measurement_sources() or run['versions'] != _library_versions():
        raise ValueError('Worker measurement code/library identity differs')
    rows = shard_rows(run, shard)
    ids = [row['sample_id'] for row in rows]
    all_ids = [row['sample_id'] for row in input_rows(run)]
    output = Path(run['output_root'])
    status_path = output / 'logs' / f'{stage}_{shard}.json'
    start = time.monotonic()
    status = dict(status='running', stage='starting', processed=0, total=len(rows), fingerprint=run['identity'][stage])
    def progress(event):
        status.update(event)
        status['elapsed_seconds'] = round(time.monotonic()-start, 2)
        atomic_json(status_path, status)
        print(json.dumps(status, allow_nan=False), flush=True)
    try:
        if stage == 'preflight':
            import torch
            from .preflight import numerical_preflight as native_preflight
            model, reader = _inference_context(run, rows)
            probe = sorted(rows, key=lambda row: len(row['input_ids']))
            selected = [probe[0]] if len(probe) == 1 else [probe[0], probe[-1]]
            receipt = native_preflight(model, _batch(reader, selected, model.processor, 'cuda:0'))
            atomic_json(output / 'logs' / f'preflight_{shard}_receipt.json', dict(receipt,
                fingerprint=run['identity']['baseline'], gpu=torch.cuda.get_device_name(0)))
            progress(dict(status='complete', stage='complete', total=len(selected), processed=len(selected)))
            return
        # Files are shared by shards; load the global allowlist, then select this shard.
        complete = load_stage(output, stage, all_ids, run['identity'][stage])
        completed = set(complete) & set(ids)
        progress(dict(processed=len(completed)))
        if len(completed) == len(rows):
            progress(dict(status='complete', stage='complete'))
            return
        if stage == 'localization':
            baseline = load_stage(output, 'baseline', all_ids, run['identity']['baseline'])
            for row in rows:
                if row['sample_id'] in completed:
                    continue
                base, arrays = baseline[row['sample_id']]
                maps = arrays['target_maps']
                partitions, cursor = {}, 0
                for partition in ('with_coordinates', 'without_coordinates'):
                    decisions = base['target_decisions'][partition]
                    count = sum(decision['status'] == 'eligible' for decision in decisions)
                    evaluated = evaluate_targets(decisions, maps[cursor:cursor+count])
                    local = cursor
                    for target in evaluated:
                        if target['status'] == 'eligible':
                            target['map_index'] = local
                            local += 1
                    partitions[partition] = evaluated
                    cursor += count
                if cursor != len(maps):
                    raise ValueError('Target-map count differs from eligible targets')
                record = dict(sample_id=row['sample_id'], status='complete', partitions=partitions)
                save_record(output, stage, record, run['identity'][stage])
                completed.add(row['sample_id'])
                progress(dict(stage='localization', processed=len(completed)))
        else:
            import torch
            from .measurement import collect_baseline, measure_causal
            from smolvlm_attention.generated_runtime import generation_config
            progress(dict(stage='loading_model'))
            model, reader = _inference_context(run, rows)
            progress(dict(stage='native_preflight'))
            probe = sorted(rows, key=lambda row: len(row['input_ids']))
            probe_rows = [probe[0]] if len(probe) == 1 else [probe[0], probe[-1]]
            from .preflight import numerical_preflight as native_preflight
            earlier = output / 'logs' / f'preflight_{shard}_receipt.json'
            receipt = json.loads(earlier.read_text()) if earlier.exists() else None
            if receipt is None or receipt.get('fingerprint') != run['identity']['baseline']:
                receipt = native_preflight(model, _batch(reader, probe_rows, model.processor, 'cuda:0'))
            atomic_json(output / 'logs' / f'{stage}_{shard}_preflight.json', dict(receipt,
                fingerprint=run['identity'][stage], gpu=torch.cuda.get_device_name(0)))
            generation = load_stage(output, 'generation', all_ids, run['identity']['generation'])
            baseline = load_stage(output, 'baseline', all_ids, run['identity']['baseline'])
            def measure(selected):
                if stage == 'baseline':
                    missing = [row for row in selected if row['sample_id'] not in generation]
                    if missing:
                        batch = _batch(reader, missing, model.processor, 'cuda:0')
                        with torch.inference_mode():
                            generated = model.generate_from_features(input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask'], visual_features=batch['visual_features'],
                                generation_config=generation_config(model.native, max_new_tokens=64)).cpu().tolist()
                        for row, raw in zip(missing, generated):
                            result = dict(sample_id=row['sample_id'], status='complete',
                                **trim_response(model.processor.tokenizer, raw, model.native.config.image_token_id,
                                    eos_token_id=model.native.generation_config.eos_token_id))
                            save_record(output, 'generation', result, run['identity']['generation'])
                            generation[row['sample_id']] = (result, {})
                    active, decisions, records = [], {}, {}
                    for row in selected:
                        gen = generation[row['sample_id']][0]
                        matches = decisions_for(row, model.processor.tokenizer, gen)
                        decisions[row['sample_id']] = matches
                        record = dict(gen, target_decisions=matches)
                        if gen['generation_status'] == 'valid':
                            active.append(row)
                        else:
                            record.update(score={'mean_logprob': None, 'token_logprobs': [], 'token_count': 0},
                                ranking_layers=list(RANKING_LAYERS), _ranking=np.zeros((5,384), np.float32),
                                _target_maps=np.zeros((0,6,384), np.float32))
                        records[row['sample_id']] = record
                    if active:
                        generated_ids = [generation[row['sample_id']][0]['generated_ids'] for row in active]
                        target_queries = [[decision['token_indices'] for decision in _flatten_decisions(decisions[row['sample_id']])
                            if decision['status'] == 'eligible'] for row in active]
                        measured = collect_baseline(model, _batch(reader, active, model.processor, 'cuda:0'),
                            generated_ids, target_queries=target_queries)
                        for i, row in enumerate(active):
                            records[row['sample_id']].update(score=measured['scores'][i],
                                ranking_layers=measured['ranking_layers'], _ranking=measured['ranking'][i].numpy(),
                                _target_maps=measured['target_maps'][i].numpy())
                    return [records[row['sample_id']] for row in selected]
                active, records = [], {}
                for row in selected:
                    base = baseline[row['sample_id']][0]
                    if base['generation_status'] != 'valid' or base['score']['token_count'] == 0:
                        records[row['sample_id']] = dict(sample_id=row['sample_id'], status='excluded',
                            reason=base['generation_status'] if base['generation_status'] != 'valid' else 'empty')
                    else:
                        active.append(row)
                if active:
                    bases = [baseline[row['sample_id']] for row in active]
                    measured = measure_causal(model, _batch(reader, active, model.processor, 'cuda:0'),
                        [base[0]['generated_ids'] for base in bases],
                        dict(scores=[base[0]['score'] for base in bases],
                             ranking=torch.from_numpy(np.stack([base[1]['ranking'] for base in bases])),
                             ranking_layers=list(RANKING_LAYERS)), sample_ids=[row['sample_id'] for row in active])
                    for row, result in zip(active, measured):
                        records[row['sample_id']] = dict(sample_id=row['sample_id'], status='complete', **result)
                return [records[row['sample_id']] for row in selected]
            def save(record):
                arrays = {key[1:]: record.pop(key) for key in list(record) if key.startswith('_')}
                save_record(output, stage, record, run['identity'][stage], **arrays)
            progress(dict(stage='measuring', batch_size=BATCH_SIZE,
                          conditions_per_question=26 if stage == 'causal' else 1))
            final_size = process_pending(rows, completed, measure, save, batch_size=BATCH_SIZE, progress=progress)
            progress(dict(final_batch_size=final_size, peak_vram_bytes=torch.cuda.max_memory_reserved(0)))
        verified = load_stage(output, stage, all_ids, run['identity'][stage])
        if not set(ids) <= set(verified):
            raise ValueError('Shard coverage incomplete')
        progress(dict(status='complete', stage='complete', processed=len(rows)))
    except Exception as exc:
        progress(dict(status='failed', stage='failed', error=repr(exc)))
        raise


def main():
    from filelock import FileLock
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', required=True, choices=('preflight', 'baseline', 'causal', 'localization'))
    parser.add_argument('--shard', required=True, type=int)
    args = parser.parse_args()
    run = read_run(args.config)
    logs = Path(run['output_root']) / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    pid_path = _pid_path(run, args.stage, args.shard)
    try:
        with FileLock(str(logs / f'shard_{args.shard}.lock'), timeout=0):
            run_worker(args.config, args.stage, args.shard)
    finally:
        if pid_path.exists() and json.loads(pid_path.read_text())['pid'] == os.getpid():
            pid_path.unlink()


if __name__ == '__main__':
    main()
