"""Run independent GPU workers and merge a single paired PosPrune experiment."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter, sleep

import numpy as np
import torch
from tqdm.auto import tqdm

from smolvlm_attention.data import IDENTITY_KEYS, _json_write, open_validated_cache, validate_cache
from smolvlm_inference.cache import CAMERA_ORDER, digest
from smolvlm_inference.model import load_model, processor_policy
from smolvlm_inference.session import InferenceSession
from smolvlm_inference.text import CachedBatchCollator
from .decoder import ATTENTION_QUERY_LIMIT, PREDICTOR_METHODS, RAW_METHODS, RAW_MMR_METHODS, SPATIAL_ALPHA, SPATIAL_SIGMA
from .image_features import IMAGE_SOURCE, build_frames, frame_plan, image_cache
from .runtime import (DEEP, GENERATION, GENERATION_BATCH, KEEP_COUNTS, SHALLOW, WEIGHT, _choose_spline_smoothing,
                      _config, _fit_camera_curves, _fit_curve, _generate_rows, _mean_difference, _validate_settings,
                      mmr_sweep_variants, select_evaluation, select_rows)
from .selection import stratified_subset


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


class _WorkerSession(InferenceSession):
    def __init__(self, checkpoint, receipt, camera_order, rank, *, features=None):
        self.device = torch.device('cuda', rank)
        torch.cuda.set_device(self.device)
        self.model = load_model(checkpoint, device=self.device, expected_prompt_camera_order=camera_order)
        self.processor = self.model.processor
        if features is None:
            self.cache = open_validated_cache(receipt)
            identity = dict(backend='smolvlm_no_pruner', version=1, model_id=self.model.model_id,
                            revision=self.model.model_revision, processor_policy=processor_policy(self.processor),
                            geometry=self.model.geometry, storage_dtype='float16', compute_dtype='float32',
                            camera_order=CAMERA_ORDER)
            if {key: self.cache.manifest[key] for key in IDENTITY_KEYS} != identity:
                raise ValueError('Validated cache does not match the worker checkpoint')
        else:
            self.cache = image_cache(features['directory'], features['frames'], features['counts'])
        self.collator = CachedBatchCollator(
            self.processor, image_seq_len=self.model.geometry['visual_tokens_per_view'],
            max_length=self.model.native.config.text_config.max_position_embeddings,
            generation=True, max_new_tokens=GENERATION['max_new_tokens'], prompt_camera_order=camera_order)
        self.checkpoint = checkpoint


def _worker(folder):
    job = _read(folder / 'job.json')
    features = job['features']
    session = _WorkerSession(job['checkpoint'], job['receipt'], job['camera_order'], job['rank'], features=features)
    model = session.model
    count = model.visual_token_len
    matrix = job['matrix']
    pruning_layer = job['pruning_layer']
    calibration_layers = job['calibration_layers']
    _validate_settings(count, len(model.native.model.text_model.layers), job['calibration_batch'],
                       job['generation_batch'], job['keep_counts'], job['selection_examples'],
                       required_layer=calibration_layers[-1] if calibration_layers else pruning_layer)
    calibration, evaluation = job['calibration'], job['evaluation']
    predictor = job.get('predictor')
    if predictor is not None:
        from .predictor import load_after2_predictor

        session.predictor = load_after2_predictor(predictor['path'], model, predictor['base_sha256'],
                                                  predictor['prepared_hashes'], session.device)
    print(f'{session.device} · {torch.cuda.get_device_name(session.device)}', flush=True)
    if features is not None:
        build_frames(model, features['frames'][job['rank']::features['workers']], features['directory'])
        _json_write(folder / 'features_ready.json', True)
        while not (folder.parent / 'features_ready.json').is_file():
            sleep(.4)
    stats = dict(session=session.describe(), gpu=torch.cuda.get_device_name(session.device),
                 per_view=model.geometry['visual_tokens_per_view'])
    if calibration_layers:
        if predictor is None:
            observed = _mean_difference(session, calibration, job['calibration_batch'], 'Calibration prefill',
                                        layers=calibration_layers)
            stats['observed'] = observed.tolist()
        else:
            stats['recording_means'] = [_mean_difference(session, group, job['calibration_batch'],
                f'Calibration group {i + 1}', layers=calibration_layers).tolist()
                for i, group in enumerate(job['recording_groups'])]
        held_out = _mean_difference(session, [row for _, row in evaluation], job['calibration_batch'], 'Held-out prefill',
                                    layers=calibration_layers)
        stats['held_out'] = held_out.tolist()
    _json_write(folder / 'prefill.json', stats)
    release = folder.parent.parent / ('bias_curve.json' if calibration_layers else 'generation_ready.json')
    while not release.is_file():
        sleep(.4)
    fitted = _read(release)['fitted'] if calibration_layers else None
    rows = _generate_rows(session, evaluation, fitted, job['warmup'], job['keep_counts'],
                          job['selection_examples'], job['generation_batch'], matrix=matrix,
                          mmr_weight=job['mmr_weight'], attention_normalization=job['attention_normalization'],
                          pruning_layer=pruning_layer)
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


def _wait_for_files(processes, folders, paths, stage):
    while not all(path.is_file() for path in paths):
        if _check_workers(processes, folders):
            raise RuntimeError(f'GPU workers exited before completing {stage}')
        sleep(.5)


def run_parallel(checkpoint, cache, prepared, output, *, cache_fingerprint, camera_order,
                 gpu_count=2, calibration_size=800, evaluation_size=100, evaluation_fraction=None, calibration_batch=4,
                 generation_batch=GENERATION_BATCH, keep_counts=KEEP_COUNTS, seed=42, selection_examples=3,
                 raw_mmr=False, reuse_raw_96=False, expected_evaluation_ids=None, camera_mmr=False,
                 mmr_weight=WEIGHT, attention_normalization='none', evaluation_subset_size=None, mmr_sweep=None,
                 pruning_layer=SHALLOW, mmr_sweep_methods=None, camera_bias_layers=None, spline_smoothing=None,
                 predictor_checkpoint=None):
    """Shard questions across GPUs and publish one fitted curve and one paired result."""
    if gpu_count < 1 or gpu_count > torch.cuda.device_count():
        raise ValueError(f'Requested {gpu_count} GPUs, found {torch.cuda.device_count()}')
    if generation_batch < 1 or calibration_batch < 1:
        raise ValueError('Calibration and generation batch sizes must be positive')
    if type(pruning_layer) is not int or pruning_layer < 1:
        raise ValueError('The pruning layer must be a positive one-based decoder block number')
    if pruning_layer != SHALLOW and (not raw_mmr or reuse_raw_96):
        raise ValueError('A different pruning layer requires raw MMR and a current top-k reference')
    calibration_layers = None if raw_mmr else (SHALLOW, DEEP)
    predictor = None
    if predictor_checkpoint is not None:
        if (raw_mmr or camera_bias_layers != (3, 23) or spline_smoothing is not None or
                tuple(keep_counts) != (32,) or mmr_weight != .9 or attention_normalization != 'minmax'):
            raise ValueError('The paired predictor run requires recording-tuned splines on layers 3/23 and MMR 32 at 0.9')
        path = Path(predictor_checkpoint).resolve(strict=True)
        predictor = dict(path=str(path), sha256=digest(path), base_sha256=digest(Path(checkpoint)),
                         prepared_hashes={name: digest(Path(prepared) / name)
                                          for name in ('train.jsonl', 'val.jsonl', 'test.jsonl')})
    if spline_smoothing is not None and (camera_bias_layers is None or set(spline_smoothing) != set(camera_order)
            or any(not np.isfinite(weight) or weight <= 0 for weight in spline_smoothing.values())):
        raise ValueError('Spline fitting requires per-camera calibration and one positive parameter per camera')
    if camera_bias_layers is not None:
        shallow, deep = camera_bias_layers
        if raw_mmr or type(shallow) is not int or type(deep) is not int or not 1 <= shallow < deep:
            raise ValueError('Camera bias fitting requires two increasing, one-based decoder layers and calibration')
        calibration_layers = (shallow, deep)
        pruning_layer = shallow
    if reuse_raw_96 and (not raw_mmr or 96 not in keep_counts or expected_evaluation_ids is None):
        raise ValueError('Reusing raw_96 requires the raw MMR experiment, budget 96, and the historical cohort')
    if reuse_raw_96 and evaluation_subset_size is not None:
        raise ValueError('Full-cohort historical predictions cannot be reused with an evaluation subset')
    if camera_mmr and not raw_mmr:
        raise ValueError('Camera MMR variants require the raw MMR experiment')
    sweep_methods = RAW_MMR_METHODS if mmr_sweep_methods is None else tuple(mmr_sweep_methods)
    if mmr_sweep_methods is not None and (mmr_sweep is None or not sweep_methods
            or len(set(sweep_methods)) != len(sweep_methods) or any(method not in RAW_MMR_METHODS for method in sweep_methods)):
        raise ValueError('Select unique supported MMR methods for the lambda sweep')
    if mmr_sweep is not None:
        if not raw_mmr or reuse_raw_96 or not mmr_sweep or len(set(mmr_sweep)) != len(mmr_sweep):
            raise ValueError('The MMR sweep requires a raw MMR run, unique weights, and a current top-k reference')
        if any(not 0 <= weight <= 1 for weight in mmr_sweep):
            raise ValueError('All sweep weights must be in [0, 1]')
    if attention_normalization not in ('none', 'minmax') or not 0 <= mmr_weight <= 1:
        raise ValueError('Use attention normalization none or minmax and an MMR weight in [0, 1]')
    if not raw_mmr and camera_bias_layers is None and (attention_normalization != 'none' or mmr_weight != WEIGHT):
        raise ValueError('MMR normalization and weight overrides require raw MMR or per-camera bias fitting')
    methods = RAW_METHODS if camera_mmr else RAW_METHODS[:2]
    matrix = None
    if raw_mmr:
        calibration, policy = [], None
        evaluation, distribution = select_evaluation(
            Path(prepared), evaluation_size, seed, evaluation_fraction=evaluation_fraction)
        matrix = (mmr_sweep_variants(keep_counts, mmr_sweep, methods=sweep_methods) if mmr_sweep is not None
                  else [(method, keep) for keep in keep_counts for method in methods
                        if not (reuse_raw_96 and method == 'raw' and keep == 96)])
    else:
        calibration, evaluation, policy, distribution = select_rows(
            Path(prepared), calibration_size, evaluation_size, seed, evaluation_fraction=evaluation_fraction)
        if camera_bias_layers is not None:
            methods = ('mmr', *PREDICTOR_METHODS) if predictor is not None else ('raw', 'corrected', 'mmr', 'raw_mmr')
            matrix = [(method, keep) for keep in keep_counts for method in methods]
    if expected_evaluation_ids is not None and [row['sample_id'] for row in evaluation] != expected_evaluation_ids:
        raise ValueError('Evaluation questions differ from the historical comparison cohort')
    parent_size = len(evaluation)
    if evaluation_subset_size is not None:
        evaluation, distribution = stratified_subset(evaluation, distribution, evaluation_subset_size, seed)
    if distribution:
        print(f'{len(evaluation):,} test questions selected by type:', flush=True)
        for group in distribution:
            print(f"  {group['question_type']}: {group['selected_count']:,} of {group['full_count']:,}", flush=True)
    count = min(gpu_count, len(evaluation))
    output = Path(output).resolve()
    workers = output / 'workers'
    workers.mkdir(parents=True, exist_ok=False)
    receipt = None
    validation_elapsed = 0.0
    frames = None
    if cache is None:
        frames = frame_plan([*calibration, *evaluation])
        print(f'Downloading DriveLM images and building features for {len(frames):,} unique frames across {count} GPU worker(s).',
              flush=True)
    else:
        print(f'Validating the shared feature cache once for {count} GPU worker(s).', flush=True)
        validation_start = perf_counter()
        receipt = str(validate_cache(cache, prepared, workers, expected_fingerprint=cache_fingerprint))
        validation_elapsed = perf_counter() - validation_start
    environment = dict(os.environ, TOKENIZERS_PARALLELISM='false')
    source = str(Path(__file__).resolve().parents[2])
    environment['PYTHONPATH'] = os.pathsep.join(filter(None, (source, environment.get('PYTHONPATH'))))
    folders = [workers / f'gpu-{rank}' for rank in range(count)]
    numbered = list(enumerate(evaluation))
    calibration_shards = [calibration[rank::count] for rank in range(count)]
    evaluation_shards = [numbered[rank::count] for rank in range(count)]
    recording_shards = [None] * count
    if predictor is not None:
        from .calibration_sweep import split_by_recording

        recording_groups = [[calibration[i] for i in group] for group in split_by_recording(calibration)]
        recording_shards = [[group[rank::count] for group in recording_groups] for rank in range(count)]
    processes = []
    prefill_start = perf_counter()
    temporary = TemporaryDirectory(prefix='posprune-frames-') if frames is not None else None
    try:
        features = None
        feature_elapsed = 0.0
        if temporary is not None:
            dataset_counts = _read(Path(prepared) / 'manifest.json')['counts']
            features = dict(frames=frames, directory=temporary.name, workers=count, counts=dataset_counts)
        for rank, folder in enumerate(folders):
            _json_write(folder / 'job.json', dict(checkpoint=str(Path(checkpoint).resolve()), receipt=receipt,
                features=features, matrix=matrix, mmr_weight=mmr_weight, pruning_layer=pruning_layer,
                calibration_layers=calibration_layers, predictor=predictor, recording_groups=recording_shards[rank],
                attention_normalization=attention_normalization,
                camera_order=camera_order, rank=rank, calibration_batch=calibration_batch,
                generation_batch=generation_batch, keep_counts=keep_counts,
                selection_examples=selection_examples, warmup=calibration[0] if calibration else evaluation[0],
                calibration=calibration_shards[rank], evaluation=evaluation_shards[rank]))
            with (folder / 'worker.log').open('w', encoding='utf-8') as log:
                command = [sys.executable, '-u', '-m', 'smolvlm_attention.posprune_ablation.parallel', str(folder)]
                processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                   env=environment, cwd=source))

        if features is not None:
            _wait_for_files(processes, folders, [folder / 'features_ready.json' for folder in folders], 'feature extraction')
            feature_elapsed = perf_counter() - prefill_start
            _json_write(workers / 'features_ready.json', True)
        if calibration_layers:
            print(f'Measuring calibration and held-out attention across {count} GPU worker(s).', flush=True)
        paths = [folder / 'prefill.json' for folder in folders]
        _wait_for_files(processes, folders, paths, 'prefill statistics')
        stats = [_read(path) for path in paths]
        primary = stats[0]
        config = _config(primary['session'], primary['per_view'], calibration, evaluation, policy, seed,
                         calibration_batch, generation_batch, keep_counts, evaluation_fraction, distribution)
        if evaluation_subset_size is not None:
            config.update(evaluation_parent_questions=parent_size, evaluation_parent_fraction=evaluation_fraction,
                          evaluation_fraction=len(evaluation) / sum(row['full_count'] for row in distribution))
        if raw_mmr:
            config.update(experiment='raw_mmr', shallow=pruning_layer, pruning_layer_index=pruning_layer - 1,
                          deep=None, calibration_split=None, calibration_batch=None,
                          weight=mmr_weight, attention_normalization=attention_normalization,
                          variants=[dict(name=f'{method}_{keep}', method=method, keep=keep,
                              source='historical' if reuse_raw_96 and method == 'raw' and keep == 96 else 'current')
                              for keep in keep_counts for method in methods])
        if camera_bias_layers is not None:
            config.update(experiment='camera_posprune', shallow=pruning_layer, deep=calibration_layers[-1],
                          pruning_layer_index=pruning_layer - 1, weight=mmr_weight,
                          bias_fit='per_camera_cubic' if spline_smoothing is None else 'per_camera_smoothing_spline',
                          attention_normalization=attention_normalization,
                          variants=[dict(name=f'{method}_{keep}', method=method, keep=keep) for method, keep in matrix])
            if spline_smoothing is not None:
                config['spline_smoothing'] = spline_smoothing
        if predictor is not None:
            group_counts = [len(group) for group in recording_groups]
            group_means = [_weighted_mean([item['recording_means'][i] for item in stats],
                                          [len(groups[i]) for groups in recording_shards]) for i in range(2)]
            spline_smoothing = _choose_spline_smoothing(group_means, group_counts, config['cameras'], config['per_view'])
            config.update(experiment='spline_vs_predictor', bias_fit='per_camera_smoothing_spline',
                          spline_smoothing=spline_smoothing, predictor_sha256=predictor['sha256'],
                          spline_cv_group_ids=[[row['sample_id'] for row in group] for group in recording_groups])
        if mmr_sweep is not None:
            config.update(variants=matrix, weight=None, mmr_lambdas=list(mmr_sweep))
            if 'attention_mmr' in sweep_methods:
                config['attention_redundancy'] = dict(query_window=ATTENTION_QUERY_LIMIT,
                    representation='incoming_attention_by_text_query_and_head')
        if (mmr_sweep is None and camera_mmr) or (mmr_sweep is not None and 'spatial_aware_mmr' in sweep_methods):
            config['spatial_redundancy'] = dict(alpha=SPATIAL_ALPHA, sigma_grid_cells=SPATIAL_SIGMA)
        config['feature_source'] = (dict(mode='regenerated', unique_frames=len(frames), **IMAGE_SOURCE)
                                    if frames is not None else dict(mode='mounted', fingerprint=cache_fingerprint))
        config['workers'] = [dict(device=item['session']['device'], gpu=item['gpu'],
                                 calibration_examples=len(a), evaluation_examples=len(b))
                             for item, a, b in zip(stats, calibration_shards, evaluation_shards)]
        _json_write(output / 'config.json', config)
        generation_start = perf_counter()
        if calibration_layers:
            observed = (_weighted_mean([item['observed'] for item in stats], [len(shard) for shard in calibration_shards])
                        if predictor is None else _weighted_mean(group_means, group_counts))
            held_out = _weighted_mean([item['held_out'] for item in stats], [len(shard) for shard in evaluation_shards])
            bias = (_fit_curve(observed, held_out) if camera_bias_layers is None else
                    _fit_camera_curves(observed, held_out, config['cameras'], config['per_view'],
                                       spline_smoothing=spline_smoothing))
            _json_write(output / 'bias_curve.json', bias)
        else:
            _json_write(output / 'generation_ready.json', True)
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
            load_and_feature_build_wall_seconds=feature_elapsed,
            load_and_prefill_wall_seconds=generation_start - prefill_start - feature_elapsed,
            paired_generation_wall_seconds=generation_elapsed))
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        if temporary is not None:
            temporary.cleanup()
    return output


if __name__ == '__main__':
    _worker(Path(sys.argv[1]))
