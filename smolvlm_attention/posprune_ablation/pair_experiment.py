"""Run the pair-similarity experiment with one worker per GPU."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter, sleep

import torch
from tqdm.auto import tqdm

from smolvlm_attention.config import TEST_COUNT
from smolvlm_attention.data import _json_write
from .image_features import IMAGE_SOURCE, build_frames, frame_plan
from .pair_similarity import (DROP_LAYER, GROUPS, LAYERS, QUERY_LIMIT, choose_pairs, correlations,
                              generate_pair_outputs, js_divergence, measure_pairs)
from .parallel import _WorkerSession, _check_workers, _merge_predictions, _read, _wait_for_files
from .runtime import GENERATION, _inputs, select_evaluation


def _answers(session, outputs):
    return session.processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)


def _measure_batch(session, batch, seed):
    rows = [row for _, row in batch]
    inputs = _inputs(session, rows)
    measurements = measure_pairs(session.model, inputs)
    results, chosen = [], []
    for offset, (index, row) in enumerate(batch):
        similarities, mass = measurements[DROP_LAYER]
        pairs = choose_pairs(similarities[offset], mass[offset], seed + index)
        chosen.append(pairs)
        results.append(dict(sample_id=row['sample_id'], question=row['question'], question_type=row['question_type'],
            ground_truth=row['answer'], correlations={str(layer): correlations(*measurements[layer][0][offset])
                                                      for layer in LAYERS}, baseline=None, ablations={}))
    valid = [i for i, pairs in enumerate(chosen) if pairs]
    if not valid:
        return results
    selected = torch.tensor(valid, device=session.device)
    inputs = {key: tensor.index_select(0, selected) for key, tensor in inputs.items()}
    baseline_tokens, baseline_probs = generate_pair_outputs(session.model, inputs)
    baseline_answers = _answers(session, baseline_tokens)
    for i, index in enumerate(valid):
        results[index]['baseline'] = baseline_answers[i]
    per_view = session.model.geometry['visual_tokens_per_view']
    cameras = session.model.prompt_camera_order
    for group in GROUPS:
        pairs = [chosen[index][group] for index in valid]
        drop = torch.tensor([[pair['dropped']] for pair in pairs], device=session.device)
        tokens, probabilities = generate_pair_outputs(session.model, inputs, drop)
        answers = _answers(session, tokens)
        distances = js_divergence(baseline_probs, probabilities).tolist()
        for i, index in enumerate(valid):
            pair = pairs[i]
            results[index]['ablations'][group] = dict(pair, prediction=answers[i],
                dropped_camera=cameras[pair['dropped'] // per_view], kept_camera=cameras[pair['kept'] // per_view],
                first_token_jsd=distances[i], first_token_changed=bool(tokens[i, 0] != baseline_tokens[i, 0]),
                answer_changed=answers[i] != baseline_answers[i])
    return results


def _worker(folder):
    job = _read(folder / 'job.json')
    features = job['features']
    session = _WorkerSession(job['checkpoint'], None, job['camera_order'], job['rank'], features=features)
    build_frames(session.model, features['frames'][job['rank']::features['workers']], features['directory'])
    _json_write(folder / 'features_ready.json', session.describe())
    while not (folder.parent / 'features_ready.json').is_file():
        sleep(.4)
    rows = job['rows']
    batch_size = job['batch_size']
    start = 0
    with (folder / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
        while start < len(rows):
            batch = rows[start:start + batch_size]
            try:
                results = _measure_batch(session, batch, job['seed'])
            except torch.cuda.OutOfMemoryError:
                if len(batch) == 1:
                    raise
                batch_size = len(batch) // 2
            else:
                for result in results:
                    stream.write(json.dumps(result, allow_nan=False, ensure_ascii=False) + '\n')
                stream.flush()
                start += len(batch)
                _json_write(folder / 'progress.json', start)
                continue
            torch.cuda.empty_cache()
            print(f'{session.device}: retrying the same questions with batch size {batch_size}', flush=True)


def run_pair_experiment(checkpoint, prepared, output, *, camera_order, questions=800, seed=42, gpu_count=2, batch_size=4):
    if not 1 <= gpu_count <= torch.cuda.device_count() or not 1 <= questions <= TEST_COUNT or batch_size < 1:
        raise ValueError('Requested GPUs, batch size, and test sample size must be available and positive')
    fraction = min(1., math.nextafter(questions / TEST_COUNT, 1.))
    evaluation, distribution = select_evaluation(Path(prepared), None, seed, evaluation_fraction=fraction)
    if len(evaluation) != questions:
        raise ValueError('The pinned test file did not yield the requested number of stratified questions')
    frames = frame_plan(evaluation)
    count = min(gpu_count, len(evaluation))
    output = Path(output).resolve()
    workers = output / 'workers'
    workers.mkdir(parents=True, exist_ok=False)
    folders = [workers / f'gpu-{rank}' for rank in range(count)]
    numbered = list(enumerate(evaluation))
    ids = [row['sample_id'] for row in evaluation]
    source = str(Path(__file__).resolve().parents[2])
    environment = dict(os.environ, TOKENIZERS_PARALLELISM='false')
    environment['PYTHONPATH'] = os.pathsep.join(filter(None, (source, environment.get('PYTHONPATH'))))
    processes = []
    started = perf_counter()
    print(f'Preparing {len(frames):,} six-camera frames for {len(evaluation):,} questions on {count} GPU(s).', flush=True)
    with TemporaryDirectory(prefix='pair-experiment-features-') as directory:
        try:
            features = dict(frames=frames, directory=directory, workers=count,
                            counts=_read(Path(prepared) / 'manifest.json')['counts'])
            for rank, folder in enumerate(folders):
                _json_write(folder / 'job.json', dict(checkpoint=str(Path(checkpoint).resolve()), features=features,
                    camera_order=camera_order, rank=rank, rows=numbered[rank::count], seed=seed, batch_size=batch_size))
                with (folder / 'worker.log').open('w', encoding='utf-8') as log:
                    command = [sys.executable, '-u', '-m', __spec__.name, str(folder)]
                    processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                       env=environment, cwd=source))
            ready = [folder / 'features_ready.json' for folder in folders]
            _wait_for_files(processes, folders, ready, 'feature extraction')
            _json_write(output / 'config.json', dict(seed=seed, sample_ids=ids, layers=LAYERS, drop_after_layer=DROP_LAYER,
                max_post_image_queries=QUERY_LIMIT, groups=GROUPS, hidden_attention_percentiles=[25, 75],
                generation=GENERATION, question_type_distribution=distribution, vision_source=IMAGE_SOURCE,
                unique_frames=len(frames), workers=[_read(path) for path in ready], batch_size=batch_size))
            _json_write(workers / 'features_ready.json', True)
            progress = [folder / 'progress.json' for folder in folders]
            with tqdm(total=len(ids), desc='Correlations and paired ablations', unit='question') as bar:
                while not _check_workers(processes, folders):
                    bar.update(sum(_read(path) for path in progress if path.is_file()) - bar.n)
                    sleep(.6)
                bar.update(len(ids) - bar.n)
            _merge_predictions(folders, ids, output / 'results.jsonl')
            _json_write(output / 'timing.json', dict(wall_seconds=perf_counter() - started))
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                process.wait()
    return output


if __name__ == '__main__':
    _worker(Path(sys.argv[1]))
