"""Post-block-two predictor QA evaluation with exact shards and safe resume."""
import json
import os
from pathlib import Path
import time

from filelock import FileLock
import torch
from torch.utils.data import DataLoader

from smolvlm_attention.sample_store import atomic_json, digest, fingerprint
from .generation import PredictorGreedyDecoder
from .config import METHOD
from . import checkpoint
from .data import PredictorCollator, PredictorDataset, read_receipt
from .training import ProgressWriter, close_loader, make_predictor, memory, move_batch, precision


@torch.inference_mode()
def predict_batch(backbone, predictor, batch, *, keep_tokens=32, max_new_tokens=512, adapter=None):
    """Run one native prefix, select once, then decode through the compact suffix."""
    if predictor.training or backbone.training:
        raise ValueError('Predictor and backbone must be in eval mode')
    device_type = next(predictor.parameters()).device.type
    if torch.is_autocast_enabled(device_type) or any(p.dtype != torch.float32 for p in predictor.parameters()):
        raise ValueError('Predictor generation requires FP32 with autocast disabled')
    decoder = adapter or PredictorGreedyDecoder(backbone, predictor, keep_tokens=keep_tokens)
    return decoder.generate(batch, max_new_tokens=max_new_tokens)


def evaluation_identity(run, predictor_checkpoint):
    if run['config'].get('method') != METHOD:
        raise ValueError('Unsupported after2 evaluation method')
    settings = run['config']['evaluation']
    return dict(version='map_after2_lastprompt_evaluation_v1', method=run['config']['method'], inputs=run['identity'],
        input_sources=run['config']['inputs'],
        predictor_sha256=digest(predictor_checkpoint), architecture=run['config']['architecture'],
        generation=dict(batch_size=settings['batch_size'], keep_tokens=settings['keep_tokens'],
            max_new_tokens=settings['max_new_tokens'], do_sample=False, num_beams=1,
            repetition_penalty=1.0, pruning='after_block_2_original_positions', prefix_layers=[0, 1, 2],
            query_policy=METHOD['query_policy'], predictor_input='whole_decoder_block_2_output'),
        precision='fp32', tf32=False, autocast=False, partition='test_original_rows_rank_stride_v1')


def _read_commit(path, identity, rank, batch_index, rows, items, keep_tokens):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    expected = dict(version=1, identity_sha256=fingerprint(identity), rank=rank,
                    batch_index=batch_index, rows=rows, sample_ids=[x['row']['sample_id'] for x in items])
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError('Evaluation batch identity, schedule, or sample IDs mismatch')
    records = payload.get('records', [])
    if len(records) != len(items) or fingerprint(records) != payload.get('records_sha256'):
        raise ValueError('Evaluation batch checksum or coverage mismatch')
    for record, item, row in zip(records, items, rows):
        source = item['row']
        ids = record.get('visual_indices', [])
        generated = record.get('generated_ids', [])
        if (record.get('prepared_row') != row or record.get('sample_id') != source['sample_id'] or
                record.get('eval_id') != source.get('eval_id', source['sample_id']) or
                record.get('reference') != source.get('answer', '') or record.get('question') != source['question'] or
                len(ids) != keep_tokens or ids != sorted(set(ids)) or
                any(type(i) is not int or not 0 <= i < identity['architecture']['visual_tokens'] for i in ids) or
                not isinstance(record.get('prediction'), str) or
                not isinstance(record.get('raw_prediction'), str) or
                record['prediction'] != record['raw_prediction'].strip() or
                not isinstance(generated, list) or any(type(i) is not int or i < 0 for i in generated) or
                not 0 < len(generated) <= identity['generation']['max_new_tokens'] or
                record.get('stop_reason') not in ('eos', 'max_new_tokens') or
                record.get('generated_token_count') != len(generated)):
            raise ValueError('Evaluation batch record differs from prepared sample or selection protocol')
    return payload


def _stress_gate(run, directory, rank, world, identity, details):
    """A fresh launcher session prevents old stress files from authorizing a new launch."""
    session = os.environ.get('SMOLVLM_MAP_AFTER2_EVALUATION_SESSION', run.get('_evaluation_session'))
    if not session or not isinstance(session, str) or not session.replace('-', '').replace('_', '').isalnum():
        raise ValueError('Evaluation requires a fresh launcher stress session')
    gate = directory / 'stress' / session
    gate.mkdir(parents=True, exist_ok=True)
    stamp = dict(identity_sha256=fingerprint(identity), session=session)
    atomic_json(gate / f'rank_{rank}.json', dict(stamp, rank=rank, **details))
    started = time.monotonic()
    while True:
        files = [gate / f'rank_{r}.json' for r in range(world)]
        if all(path.is_file() for path in files):
            for r, path in enumerate(files):
                value = json.loads(path.read_text(encoding='utf-8'))
                if any(value.get(key) != item for key, item in stamp.items()) or value.get('rank') != r or not value.get('ok'):
                    raise ValueError('Evaluation peer stress receipt mismatch')
            return
        if time.monotonic() - started > 3600:
            raise TimeoutError('Evaluation peer did not pass stress within one hour')
        time.sleep(.25)


def _evaluation_dataset(receipt):
    if read_receipt(receipt, include_targets=False).get('version') == 2:
        from smolvlm_predictor_io.compact import CompactDataset
        return CompactDataset(receipt, 'test')
    return PredictorDataset(receipt, 'test')


def _evaluation_collator(receipt, processor, maximum):
    if read_receipt(receipt, include_targets=False).get('version') == 2:
        from smolvlm_predictor_io.compact import CompactCollator
        return CompactCollator(receipt, max_new_tokens=maximum)
    return PredictorCollator(receipt, processor, max_new_tokens=maximum)


def evaluate_worker(run, rank, predictor_checkpoint, *, device='cuda:0', backbone=None):
    """Evaluate exact, nonduplicated test rows; production has one process per CUDA GPU."""
    description = read_receipt(run['receipt'], include_targets=False)
    if description['identity'] != run['identity']:
        raise ValueError('Evaluation input identity mismatch')
    world = int(run.get('_evaluation_world_size', 2))
    if not 0 <= rank < world:
        raise ValueError('Invalid evaluation rank')
    precision()
    device = torch.device(device)
    if backbone is None:
        from smolvlm_inference.model import load_model, CAMERA_ORDER
        backbone = load_model(run['backbone_checkpoint'], device=str(device), expected_prompt_camera_order=CAMERA_ORDER)
    backbone = backbone.to(device=device, dtype=torch.float32).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    # Predictor initialization is irrelevant after strict loading and must not consume caller RNG.
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
        torch.manual_seed(run['config']['training']['seed'])
        predictor = make_predictor(backbone, run['config']).to(device=device, dtype=torch.float32).eval()
    payload = checkpoint.read(predictor_checkpoint, predictor, run['identity'])
    predictor.load_state_dict(payload['model_state_dict'], strict=True)
    identity = evaluation_identity(run, predictor_checkpoint)
    directory = Path(run['output_dir']) / 'evaluation'
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / 'identity.lock'), timeout=60):
        path = directory / 'identity.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != identity:
            raise ValueError('Evaluation output belongs to another predictor/configuration')
        atomic_json(path, identity)
    with FileLock(str(directory / f'rank_{rank}.lock'), timeout=0):
        dataset = _evaluation_dataset(run['receipt'])
        size = run['config']['evaluation']['batch_size']
        keep = run['config']['evaluation']['keep_tokens']
        maximum = run['config']['evaluation']['max_new_tokens']
        rows = list(range(rank, len(dataset), world))
        if not rows:
            raise ValueError('Empty evaluation shard')
        schedules = [rows[i:i + size] for i in range(0, len(rows), size)]
        pending = []
        shard = directory / f'rank_{rank}'; shard.mkdir(exist_ok=True)
        for index, scheduled in enumerate(schedules):
            path = shard / f'batch_{index:06d}.json'
            if path.exists():
                _read_commit(path, identity, rank, index, scheduled, [dataset[i] for i in scheduled], keep)
            else:
                pending.append(index)
        lengths = dataset.prompt_lengths
        stress = sorted(rows, key=lambda row: (-int(lengths[row]), row))[:size]
        settings = run['config']['training']
        options = dict(num_workers=settings['num_workers'], pin_memory=settings['pin_memory'],
            generator=torch.Generator().manual_seed(settings['seed']), persistent_workers=settings['num_workers'] > 0)
        if settings['num_workers']:
            options.update(prefetch_factor=settings['prefetch_factor'], multiprocessing_context='spawn')
        loader = DataLoader(dataset, batch_sampler=[stress] + [schedules[i] for i in pending],
            collate_fn=_evaluation_collator(run['receipt'], backbone.processor, maximum), **options)
        progress = ProgressWriter(run['output_dir'], filename=f'evaluate_rank_{rank}.jsonl')
        from smolvlm_predictor_io.progress import StatusWriter
        live = StatusWriter(run['output_dir'], rank)
        adapter = PredictorGreedyDecoder(backbone, predictor, keep_tokens=keep)
        started = time.perf_counter()
        try:
            iterator = iter(loader)
            batch = move_batch(next(iterator), device)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            predict_batch(backbone, predictor, batch, keep_tokens=keep, max_new_tokens=maximum, adapter=adapter)
            details = dict(ok=True, rows=stress, samples=len(stress), **memory(device))
            _stress_gate(run, directory, rank, world, identity, details)
            progress.write('stress_passed', rank=rank, **details)
            print(json.dumps(dict(phase='evaluation_stress_passed', rank=rank, **details)), flush=True)
            processed = len(rows) - sum(len(schedules[i]) for i in pending)
            resumed = processed
            live.update('test', processed, len(rows), force=True)
            started = time.perf_counter()
            for index, batch in zip(pending, iterator):
                scheduled = schedules[index]
                if batch['rows'].tolist() != scheduled:
                    raise ValueError('Evaluation loader changed fixed batch schedule')
                outputs = predict_batch(backbone, predictor, move_batch(batch, device), keep_tokens=keep,
                                        max_new_tokens=maximum, adapter=adapter)
                items = [dataset[i] for i in scheduled]
                records = [dict(result, prepared_row=row, sample_id=item['row']['sample_id'],
                    eval_id=item['row'].get('eval_id', item['row']['sample_id']), question=item['row']['question'],
                    reference=item['row'].get('answer', '')) for result, item, row in zip(outputs, items, scheduled)]
                commit = dict(version=1, identity_sha256=fingerprint(identity), rank=rank, batch_index=index,
                    rows=scheduled, sample_ids=batch['sample_ids'], records=records, records_sha256=fingerprint(records))
                atomic_json(shard / f'batch_{index:06d}.json', commit)
                processed += len(scheduled)
                live.update('test', processed, len(rows))
                elapsed = time.perf_counter() - started
                throughput = (processed - resumed) / elapsed if elapsed else None
                status = dict(rank=rank, processed=processed, total=len(rows), batch=index,
                    resumed_samples=resumed, newly_processed_samples=processed - resumed,
                    elapsed_seconds=elapsed, samples_per_second=throughput,
                    estimated_remaining_seconds=(len(rows) - processed) / throughput if throughput else None,
                    **memory(device))
                progress.write('evaluation_batch', **status)
                print(json.dumps(dict(phase='evaluation_batch', **status)), flush=True)
            live.update('test', processed, len(rows), force=True)
            checksums = {f'batch_{i:06d}.json': digest(shard / f'batch_{i:06d}.json') for i in range(len(schedules))}
            atomic_json(shard / 'complete.json', dict(identity_sha256=fingerprint(identity), rank=rank,
                count=len(rows), batch_checksums=checksums))
            progress.write('evaluation_complete', rank=rank, processed=processed, total=len(rows),
                resumed_samples=resumed, elapsed_seconds=time.perf_counter() - started)
            return dict(rank=rank, count=len(rows), resumed_batches=len(schedules) - len(pending))
        finally:
            close_loader(loader)


def finalize_evaluation(run, predictor_checkpoint):
    """Merge in prepared order only after all shards pass coverage and checksums."""
    from smolvlm_topk_benchmark.scoring import score_rows
    identity = evaluation_identity(run, predictor_checkpoint)
    root = Path(run['output_dir']); directory = root / 'evaluation'
    if json.loads((directory / 'identity.json').read_text(encoding='utf-8')) != identity:
        raise ValueError('Evaluation finalization identity mismatch')
    dataset = _evaluation_dataset(run['receipt'])
    world = int(run.get('_evaluation_world_size', 2))
    size = run['config']['evaluation']['batch_size']; keep = run['config']['evaluation']['keep_tokens']
    merged = {}
    for rank in range(world):
        rows = list(range(rank, len(dataset), world)); shard = directory / f'rank_{rank}'
        complete = json.loads((shard / 'complete.json').read_text(encoding='utf-8'))
        schedules = [rows[i:i + size] for i in range(0, len(rows), size)]
        expected = {f'batch_{i:06d}.json': digest(shard / f'batch_{i:06d}.json') for i in range(len(schedules))}
        if complete != dict(identity_sha256=fingerprint(identity), rank=rank, count=len(rows), batch_checksums=expected):
            raise ValueError('Evaluation completed shard checksum/coverage mismatch')
        for index, scheduled in enumerate(schedules):
            commit = _read_commit(shard / f'batch_{index:06d}.json', identity, rank, index,
                                  scheduled, [dataset[i] for i in scheduled], keep)
            for record in commit['records']:
                if record['prepared_row'] in merged:
                    raise ValueError('Duplicate prepared row in evaluation')
                merged[record['prepared_row']] = record
    if set(merged) != set(range(len(dataset))):
        raise ValueError('Incomplete evaluation test coverage')
    records = [merged[i] for i in range(len(dataset))]
    result = score_rows(records, expected_count=len(dataset))
    temporary = root / 'predictions.jsonl.tmp'
    with temporary.open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, root / 'predictions.jsonl')
    for name in ('metrics', 'metrics_display', 'scoring_identity'):
        atomic_json(root / f'{name}.json', result[name])
    summary = dict(identity=identity, diagnostics=result['diagnostics'], metrics_display=result['metrics_display'],
        prediction_sha256=digest(root / 'predictions.jsonl'), scope='greedy_QA_quality_no_latency_or_FLOPs')
    atomic_json(root / 'evaluation_summary.json', summary)
    return summary
