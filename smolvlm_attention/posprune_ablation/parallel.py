"""Run independent GPU workers and merge a single paired PosPrune experiment."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter, sleep

import numpy as np
import torch
from tqdm.auto import tqdm

from smolvlm_attention.data import IDENTITY_KEYS, _json_write, open_validated_cache, validate_cache
from smolvlm_inference.cache import CAMERA_ORDER
from smolvlm_inference.model import load_model, processor_policy
from smolvlm_inference.session import InferenceSession
from smolvlm_inference.text import CachedBatchCollator
from .runtime import (GENERATION, GENERATION_BATCH, KEEP_COUNTS, _config, _fit_curve, _generate_rows,
                      _mean_difference, _validate_settings, select_rows)


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


class _WorkerSession(InferenceSession):
    def __init__(self, checkpoint, receipt, camera_order, rank):
        self.device = torch.device('cuda', rank)
        torch.cuda.set_device(self.device)
        self.model = load_model(checkpoint, device=self.device, expected_prompt_camera_order=camera_order)
        self.processor = self.model.processor
        self.cache = open_validated_cache(receipt)
        identity = dict(backend='smolvlm_no_pruner', version=1, model_id=self.model.model_id,
                        revision=self.model.model_revision, processor_policy=processor_policy(self.processor),
                        geometry=self.model.geometry, storage_dtype='float16', compute_dtype='float32',
                        camera_order=CAMERA_ORDER)
        if {key: self.cache.manifest[key] for key in IDENTITY_KEYS} != identity:
            raise ValueError('Validated cache does not match the worker checkpoint')
        self.collator = CachedBatchCollator(
            self.processor, image_seq_len=self.model.geometry['visual_tokens_per_view'],
            max_length=self.model.native.config.text_config.max_position_embeddings,
            generation=True, max_new_tokens=GENERATION['max_new_tokens'], prompt_camera_order=camera_order)
        self.checkpoint = checkpoint


def _worker(folder):
    job = _read(folder / 'job.json')
    session = _WorkerSession(job['checkpoint'], job['receipt'], job['camera_order'], job['rank'])
    model = session.model
    count = model.visual_token_len
    _validate_settings(count, len(model.native.model.text_model.layers), job['calibration_batch'],
                       job['generation_batch'], job['keep_counts'], job['selection_examples'])
    calibration, evaluation = job['calibration'], job['evaluation']
    print(f'{session.device} · {torch.cuda.get_device_name(session.device)}', flush=True)
    observed = _mean_difference(session, calibration, job['calibration_batch'], 'Calibration prefill')
    held_out = _mean_difference(session, [row for _, row in evaluation], job['calibration_batch'], 'Held-out prefill')
    _json_write(folder / 'prefill.json', dict(session=session.describe(),
        gpu=torch.cuda.get_device_name(session.device), per_view=model.geometry['visual_tokens_per_view'],
        observed=observed.tolist(), held_out=held_out.tolist()))
    bias_path = folder.parent.parent / 'bias_curve.json'
    while not bias_path.is_file():
        sleep(.4)
    fitted = _read(bias_path)['fitted']
    rows = _generate_rows(session, evaluation, fitted, job['warmup'], job['keep_counts'],
                          job['selection_examples'], job['generation_batch'])
    with (folder / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
        for completed, row in enumerate(rows, 1):
            row['device'] = str(session.device)
            stream.write(json.dumps(row, allow_nan=False, ensure_ascii=False) + '\n')
            stream.flush()
            if completed % 4 == 0 or completed == len(evaluation):
                _json_write(folder / 'progress.json', completed)


def _weighted_mean(means, counts):
    return np.average(np.asarray(means, dtype=np.float64), axis=0, weights=counts)


def _merge_predictions(folders, sample_ids, output):
    """Restore the original evaluation order and require complete coverage."""
    temporary = output.with_suffix('.tmp')
    with ExitStack() as files:
        sources = [files.enter_context((folder / 'predictions.jsonl').open(encoding='utf-8')) for folder in folders]
        stream = files.enter_context(temporary.open('w', encoding='utf-8'))
        for index, sample_id in enumerate(sample_ids):
            line = sources[index % len(sources)].readline()
            if not line or json.loads(line)['sample_id'] != sample_id:
                raise ValueError(f'Invalid or missing GPU result at question {index}: {sample_id}')
            stream.write(line)
        if any(source.readline() for source in sources):
            raise ValueError('A GPU worker produced extra results')
    temporary.replace(output)


def _check_workers(processes, folders):
    codes = [process.poll() for process in processes]
    for code, folder in zip(codes, folders):
        if code not in (None, 0):
            log = folder / 'worker.log'
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 4000))
                tail = stream.read().decode(errors='replace')
            raise RuntimeError(f'{folder.name} exited with code {code}.\n{tail}')
    return all(code == 0 for code in codes)


def run_parallel(checkpoint, cache, prepared, output, *, cache_fingerprint, camera_order,
                 gpu_count=2, calibration_size=800, evaluation_size=100, evaluation_fraction=None, calibration_batch=4,
                 generation_batch=GENERATION_BATCH, keep_counts=KEEP_COUNTS, seed=42, selection_examples=3):
    """Shard questions across GPUs and publish one fitted curve and one paired result."""
    if gpu_count < 1 or gpu_count > torch.cuda.device_count():
        raise ValueError(f'Requested {gpu_count} GPUs, found {torch.cuda.device_count()}')
    if generation_batch < 1 or calibration_batch < 1:
        raise ValueError('Calibration and generation batch sizes must be positive')
    calibration, evaluation, policy, distribution = select_rows(
        Path(prepared), calibration_size, evaluation_size, seed, evaluation_fraction=evaluation_fraction)
    if distribution:
        print(f'{len(evaluation):,} test questions selected by type:', flush=True)
        for group in distribution:
            print(f"  {group['question_type']}: {group['selected_count']:,} of {group['full_count']:,}", flush=True)
    count = min(gpu_count, len(evaluation))
    output = Path(output).resolve()
    workers = output / 'workers'
    workers.mkdir(parents=True, exist_ok=False)
    print(f'Validating the shared feature cache once for {count} GPU worker(s).', flush=True)
    validation_start = perf_counter()
    receipt = validate_cache(cache, prepared, workers, expected_fingerprint=cache_fingerprint)
    validation_elapsed = perf_counter() - validation_start
    environment = dict(os.environ, TOKENIZERS_PARALLELISM='false')
    source = str(Path(__file__).resolve().parents[2])
    environment['PYTHONPATH'] = os.pathsep.join(filter(None, (source, environment.get('PYTHONPATH'))))
    folders = [workers / f'gpu-{rank}' for rank in range(count)]
    numbered = list(enumerate(evaluation))
    calibration_shards = [calibration[rank::count] for rank in range(count)]
    evaluation_shards = [numbered[rank::count] for rank in range(count)]
    processes = []
    prefill_start = perf_counter()
    try:
        for rank, folder in enumerate(folders):
            _json_write(folder / 'job.json', dict(checkpoint=str(Path(checkpoint).resolve()), receipt=str(receipt),
                camera_order=camera_order, rank=rank, calibration_batch=calibration_batch,
                generation_batch=generation_batch, keep_counts=keep_counts,
                selection_examples=selection_examples, warmup=calibration[0],
                calibration=calibration_shards[rank], evaluation=evaluation_shards[rank]))
            with (folder / 'worker.log').open('w', encoding='utf-8') as log:
                command = [sys.executable, '-u', '-m', 'smolvlm_attention.posprune_ablation.parallel', str(folder)]
                processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                   env=environment, cwd=source))

        print(f'Measuring calibration and held-out attention across {count} GPU worker(s).', flush=True)
        paths = [folder / 'prefill.json' for folder in folders]
        while not all(path.is_file() for path in paths):
            if _check_workers(processes, folders):
                raise RuntimeError('GPU workers exited without returning prefill statistics')
            sleep(.5)
        stats = [_read(path) for path in paths]
        primary = stats[0]
        config = _config(primary['session'], primary['per_view'], calibration, evaluation, policy, seed,
                         calibration_batch, generation_batch, keep_counts, evaluation_fraction, distribution)
        config['workers'] = [dict(device=item['session']['device'], gpu=item['gpu'],
                                 calibration_examples=len(a), evaluation_examples=len(b))
                             for item, a, b in zip(stats, calibration_shards, evaluation_shards)]
        _json_write(output / 'config.json', config)
        observed = _weighted_mean([item['observed'] for item in stats], [len(shard) for shard in calibration_shards])
        held_out = _weighted_mean([item['held_out'] for item in stats], [len(shard) for shard in evaluation_shards])
        generation_start = perf_counter()
        _json_write(output / 'bias_curve.json', _fit_curve(observed, held_out))
        progress = [folder / 'progress.json' for folder in folders]
        with tqdm(total=len(evaluation), desc='Paired generation', unit='question') as bar:
            while not _check_workers(processes, folders):
                total = sum(_read(path) for path in progress if path.is_file())
                bar.update(total - bar.n)
                sleep(.6)
            bar.update(len(evaluation) - bar.n)
        generation_elapsed = perf_counter() - generation_start
        _merge_predictions(folders, config['evaluation_ids'], output / 'predictions.jsonl')
        _json_write(output / 'parallel_timing.json', dict(workers=count,
            cache_validation_wall_seconds=validation_elapsed,
            load_and_prefill_wall_seconds=generation_start - prefill_start,
            paired_generation_wall_seconds=generation_elapsed))
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
    return output


if __name__ == '__main__':
    _worker(Path(sys.argv[1]))
