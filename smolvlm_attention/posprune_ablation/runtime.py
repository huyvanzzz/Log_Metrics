"""Calibration and paired generation using the existing checkpoint and cache."""

import json
from itertools import islice
from pathlib import Path, PurePosixPath
import random
import re

import numpy as np
import torch
from tqdm.auto import tqdm

from smolvlm_inference.cache import record_key, records
from .decoder import attention_difference, generate
from .selection import stratified_test


SHALLOW, DEEP = 3, 22
WEIGHT = 0.9
KEEP_COUNTS = (192, 96, 48)
GENERATION_BATCH = 4
GENERATION = dict(max_new_tokens=512, num_beams=1, do_sample=False, early_stopping=False)


def _scene(row):
    for field in ('scene_token', 'scene_id', 'scene_name', 'scene'):
        value = row.get(field)
        if isinstance(value, (str, int)) and str(value):
            return str(value)
    # nuScenes filenames encode the recording log before __CAM_FRONT__.
    front = PurePosixPath(row['camera_paths']['Front'].replace('\\', '/')).name
    log = re.fullmatch(r'(n\d+-[\d+\-]+)__CAM_FRONT__\d+\.[^.]+', front)
    return log.group(1) if log else None


def _select_test(test, test_file, evaluation_size, seed, evaluation_fraction):
    if not test or (evaluation_size is not None and not 1 <= evaluation_size <= len(test)):
        raise ValueError('Evaluation size must be within the test split or None for the full split')
    if evaluation_fraction is not None and evaluation_size is not None:
        raise ValueError('Specify an evaluation fraction or an evaluation size')
    if evaluation_fraction is not None:
        return stratified_test(test, test_file, evaluation_fraction, seed)
    if evaluation_size is not None:
        evaluation = [test[i] for i in sorted(random.Random(seed).sample(range(len(test)), evaluation_size))]
    else:
        evaluation = test
    return evaluation, None


def select_evaluation(prepared, evaluation_size, seed, *, evaluation_fraction=None):
    test_file = prepared / 'test.jsonl'
    return _select_test(list(records(test_file)), test_file, evaluation_size, seed, evaluation_fraction)


def select_rows(prepared, calibration_size, evaluation_size, seed, *, evaluation_fraction=None):
    """Use validation for fitting, test for evaluation, and exclude any overlap."""
    val = list(records(prepared / 'val.jsonl'))
    test_file = prepared / 'test.jsonl'
    test = list(records(test_file))
    with_scenes = all(_scene(row) is not None for row in (*val, *test))
    test_frames = {record_key(row) for row in test}
    test_scenes = {_scene(row) for row in test} if with_scenes else set()
    eligible = [row for row in val if record_key(row) not in test_frames and _scene(row) not in test_scenes]
    if calibration_size < 1 or len(eligible) < calibration_size:
        raise ValueError(f'Only {len(eligible)} disjoint validation questions for {calibration_size} calibration examples')
    calibration = random.Random(seed).sample(eligible, calibration_size)
    evaluation, distribution = _select_test(test, test_file, evaluation_size, seed, evaluation_fraction)
    overlap_policy = 'scene ID or recording log and exact camera frame' if with_scenes else 'exact camera frame'
    return calibration, evaluation, overlap_policy, distribution


def _inputs(session, rows):
    items = []
    for row in rows:
        features = session.cache.array[session.cache.index[record_key(row)]]
        items.append(dict(row, visual_features=torch.from_numpy(np.array(features, copy=True))))
    batch = session.collator(items)
    return dict(input_ids=batch['input_ids'].to(session.device),
                attention_mask=batch['attention_mask'].to(session.device),
                visual_features=batch['visual_features'].to(session.device, dtype=torch.float32))


def _mean_difference(session, rows, batch_size, name):
    total = np.zeros(session.model.visual_token_len, dtype=np.float64)
    for start in tqdm(range(0, len(rows), batch_size), desc=name, unit='batch'):
        inputs = _inputs(session, rows[start:start + batch_size])
        values = attention_difference(session.model, inputs, SHALLOW, DEEP)
        total += values.sum(axis=0, dtype=np.float64)
    return total / len(rows) if rows else total


def _write_json(path, payload):
    path.write_text(json.dumps(payload, allow_nan=False, indent=2) + '\n', encoding='utf-8')


def _answers(session, inputs, curve, method=None, count=None, *, mmr_weight=WEIGHT, attention_normalization='none'):
    tokens, timing, details = generate(session.model, inputs, GENERATION, SHALLOW,
                                       method=method, curve=curve, count=count, weight=mmr_weight,
                                       attention_normalization=attention_normalization)
    sequences = tokens.tolist()
    predictions = session.processor.tokenizer.batch_decode(sequences, skip_special_tokens=True)
    eos = session.model.native.generation_config.eos_token_id
    end_tokens = set(eos) if isinstance(eos, (list, tuple)) else {eos}
    results = []
    for sequence, prediction in zip(sequences, predictions):
        length = next((i + 1 for i, token in enumerate(sequence) if token in end_tokens), len(sequence))
        results.append(dict(prediction=prediction, generated_tokens=length, **timing))
    return results, [None] * len(results) if details is None else details


def _variants(keep_counts):
    return [(method, count) for count in keep_counts for method in ('raw', 'corrected', 'mmr')]


def _validate_settings(visual_count, layers, calibration_batch, generation_batch, keep_counts, selection_examples,
                       *, required_layer=None):
    required_layer = DEEP if required_layer is None else required_layer
    if (visual_count < 4 or layers < required_layer or calibration_batch < 1 or generation_batch < 1 or selection_examples < 1
            or not keep_counts or len(set(keep_counts)) != len(keep_counts)
            or any(type(count) is not int or not 1 <= count < visual_count for count in keep_counts)):
        raise ValueError('Invalid layer, token budget, or sample configuration')


def _config(session, per_view, calibration, evaluation, overlap_policy, seed, calibration_batch,
            generation_batch, keep_counts, evaluation_fraction, question_type_distribution):
    return dict(session=session, shallow=SHALLOW, deep=DEEP, weight=WEIGHT,
                seed=seed, generation=GENERATION, calibration_batch=calibration_batch,
                generation_batch=generation_batch,
                visual_count=session['visual_tokens'], per_view=per_view, cameras=session['camera_order'],
                calibration_split='val', evaluation_split='test', overlap_policy=overlap_policy,
                evaluation_fraction=evaluation_fraction, question_type_distribution=question_type_distribution,
                calibration_unique_frames=len({record_key(row) for row in calibration}),
                evaluation_unique_frames=len({record_key(row) for row in evaluation}),
                calibration_ids=[row['sample_id'] for row in calibration],
                evaluation_ids=[row['sample_id'] for row in evaluation],
                variants=[dict(name=f'{method}_{count}', method=method, keep=count)
                          for method, count in _variants(keep_counts)])


def _fit_curve(observed, held_out):
    # Scaling fixed-position indices to [0, 1] improves numerical conditioning.
    position = np.linspace(0, 1, len(observed))
    polynomial = np.polynomial.Polynomial.fit(position, observed, deg=3).convert()
    return dict(degree=3, coefficients=polynomial.coef.tolist(), normalized_positions=position.tolist(),
                calibration_observed=observed.tolist(), fitted=polynomial(position).tolist(),
                evaluation_observed=held_out.tolist())


def _generate_batch(session, batch, curve, matrix, offset, selection_examples, *,
                    mmr_weight=WEIGHT, attention_normalization='none'):
    inputs = _inputs(session, [row for _, row in batch])
    per_view = session.model.geometry['visual_tokens_per_view']
    camera_count = len(session.model.prompt_camera_order)
    batch_id = batch[0][0]
    paired = [dict(sample_id=row['sample_id'], question=row['question'], ground_truth=row['answer'],
                   batch_id=batch_id, batch_size=len(batch), variants={}) for _, row in batch]
    for (_, row), sample in zip(batch, paired, strict=True):
        if 'question_type' in row:
            sample['question_type'] = row['question_type']
    for method, count in matrix[offset:] + matrix[:offset]:
        answers, selections = _answers(session, inputs, curve, method, count, mmr_weight=mmr_weight,
                                       attention_normalization=attention_normalization)
        name = 'baseline' if method is None else f'{method}_{count}'
        for (index, _), sample, result, details in zip(batch, paired, answers, selections, strict=True):
            if details is not None:
                result['camera_counts'] = np.bincount(np.array(details['picks']) // per_view,
                                                     minlength=camera_count).tolist()
                if details['penalties'] is not None and len(details['picks']) > 1:
                    penalty = np.abs(details['penalties'][1:])
                    relevance = np.abs(np.array(details['relevance'])[details['picks'][1:]])
                    result['mean_abs_weighted_relevance'] = float(mmr_weight * relevance.mean())
                    result['mean_abs_weighted_similarity'] = float((1 - mmr_weight) * penalty.mean())
                if index < selection_examples:
                    result['selection'] = details
            sample['variants'][name] = result
    return paired


def _generate_rows(session, evaluation, fitted, warmup_row, keep_counts, selection_examples, batch_size, *, matrix=None,
                   mmr_weight=WEIGHT, attention_normalization='none'):
    curve = None if fitted is None else torch.as_tensor(fitted, device=session.device, dtype=torch.float32)

    # Warm both cases before comparing timings; rotate execution order across questions.
    warmup = _inputs(session, [warmup_row])
    if matrix is None:
        _answers(session, warmup, curve)
        _answers(session, warmup, curve, 'mmr', min(keep_counts))
        matrix = [(None, None), *_variants(keep_counts)]
    else:
        for method, count in matrix:
            _answers(session, warmup, curve, method, count, mmr_weight=mmr_weight,
                     attention_normalization=attention_normalization)
    evaluation = iter(evaluation)
    pending = []
    offset = 0
    while True:
        pending.extend(islice(evaluation, max(batch_size - len(pending), 0)))
        if not pending:
            return
        batch = pending[:batch_size]
        try:
            paired = _generate_batch(session, batch, curve, matrix, offset, selection_examples,
                                     mmr_weight=mmr_weight, attention_normalization=attention_normalization)
        except torch.cuda.OutOfMemoryError:
            if session.device.type != 'cuda' or len(batch) == 1:
                raise
            batch_size = len(batch) // 2
        else:
            del pending[:len(batch)]
            offset = (offset + 1) % len(matrix)
            yield from paired
            continue
        with torch.cuda.device(session.device):
            torch.cuda.empty_cache()
        print(f'{session.device}: CUDA OOM. Retrying the same questions at generation batch size {batch_size}.', flush=True)


def run(session, output, *, calibration_size=800, evaluation_size=100, evaluation_fraction=None, calibration_batch=4,
        generation_batch=GENERATION_BATCH,
        keep_counts=KEEP_COUNTS, seed=42, selection_examples=3):
    """Run the entire paired matrix, with a fixed curve and the canonical generation settings."""
    output = Path(output)
    model = session.model
    _validate_settings(model.visual_token_len, len(model.native.model.text_model.layers),
                       calibration_batch, generation_batch, keep_counts, selection_examples)
    calibration, evaluation, overlap_policy, distribution = select_rows(
        session.cache.prepared, calibration_size, evaluation_size, seed, evaluation_fraction=evaluation_fraction)
    output.mkdir(parents=True, exist_ok=False)
    config = _config(session.describe(), model.geometry['visual_tokens_per_view'], calibration, evaluation,
                     overlap_policy, seed, calibration_batch, generation_batch, keep_counts,
                     evaluation_fraction, distribution)
    _write_json(output / 'config.json', config)
    observed = _mean_difference(session, calibration, calibration_batch, 'Fit positional bias')
    held_out = _mean_difference(session, evaluation, calibration_batch, 'Measure held-out residuals')
    bias = _fit_curve(observed, held_out)
    _write_json(output / 'bias_curve.json', bias)
    rows = enumerate(tqdm(evaluation, desc='Paired generation', unit='question'))
    with (output / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
        for paired in _generate_rows(session, rows, bias['fitted'], calibration[0], keep_counts,
                                     selection_examples, generation_batch):
            stream.write(json.dumps(paired, allow_nan=False, ensure_ascii=False) + '\n')
            stream.flush()
    return output
