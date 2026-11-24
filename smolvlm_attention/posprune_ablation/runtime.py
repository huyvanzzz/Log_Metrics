"""Calibration and paired generation using the existing checkpoint and cache."""

import json
from pathlib import Path, PurePosixPath
import random
import re

import numpy as np
import torch
from tqdm.auto import tqdm

from smolvlm_inference.cache import record_key, records
from .decoder import attention_difference, generate


SHALLOW, DEEP = 3, 22
WEIGHT = 0.9
KEEP_COUNTS = (192, 96, 48)
GENERATION = dict(max_new_tokens=512, num_beams=3, length_penalty=1.0, early_stopping=True)


def _scene(row):
    for field in ('scene_token', 'scene_id', 'scene_name', 'scene'):
        value = row.get(field)
        if isinstance(value, (str, int)) and str(value):
            return str(value)
    # nuScenes filenames encode the recording log before __CAM_FRONT__.
    front = PurePosixPath(row['camera_paths']['Front'].replace('\\', '/')).name
    log = re.fullmatch(r'(n\d+-[\d+\-]+)__CAM_FRONT__\d+\.[^.]+', front)
    return log.group(1) if log else None


def select_rows(prepared, calibration_size, evaluation_size, seed):
    """Use validation for fitting, test for evaluation, and exclude any overlap."""
    val = list(records(prepared / 'val.jsonl'))
    test = list(records(prepared / 'test.jsonl'))
    with_scenes = all(_scene(row) is not None for row in (*val, *test))
    test_frames = {record_key(row) for row in test}
    test_scenes = {_scene(row) for row in test} if with_scenes else set()
    eligible = [row for row in val if record_key(row) not in test_frames and _scene(row) not in test_scenes]
    if calibration_size < 1 or len(eligible) < calibration_size:
        raise ValueError(f'Only {len(eligible)} disjoint validation questions for {calibration_size} calibration examples')
    if not test or (evaluation_size is not None and not 1 <= evaluation_size <= len(test)):
        raise ValueError('Evaluation size must be within the test split or None for the full split')
    calibration = random.Random(seed).sample(eligible, calibration_size)
    evaluation = test if evaluation_size is None else [
        test[i] for i in sorted(random.Random(seed).sample(range(len(test)), evaluation_size))]
    return calibration, evaluation, 'scene ID or recording log and exact camera frame' if with_scenes else 'exact camera frame'


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
    return total / len(rows)


def _write_json(path, payload):
    path.write_text(json.dumps(payload, allow_nan=False, indent=2) + '\n', encoding='utf-8')


def _answer(session, inputs, curve, method=None, count=None):
    tokens, timing, details = generate(session.model, inputs, GENERATION, SHALLOW,
                                       method=method, curve=curve, count=count, weight=WEIGHT)
    prediction = session.processor.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
    return dict(prediction=prediction, **timing), details


def run(session, output, *, calibration_size=800, evaluation_size=100, calibration_batch=4,
        keep_counts=KEEP_COUNTS, seed=42, selection_examples=3):
    """Run the entire paired matrix, with a fixed curve and the canonical generation settings."""
    output = Path(output)
    visual_count = session.model.visual_token_len
    if (visual_count < 4 or len(session.model.native.model.text_model.layers) < DEEP
            or calibration_batch < 1 or selection_examples < 1
            or not keep_counts or len(set(keep_counts)) != len(keep_counts)
            or any(type(count) is not int or not 1 <= count < visual_count for count in keep_counts)):
        raise ValueError('Invalid layer, token budget, or sample configuration')
    calibration, evaluation, overlap_policy = select_rows(
        session.cache.prepared, calibration_size, evaluation_size, seed)
    output.mkdir(parents=True, exist_ok=False)
    cameras = list(session.model.prompt_camera_order)
    per_view = session.model.geometry['visual_tokens_per_view']
    variants = [(method, count) for count in keep_counts for method in ('raw', 'corrected', 'mmr')]
    config = dict(session=session.describe(), shallow=SHALLOW, deep=DEEP, weight=WEIGHT,
                  seed=seed, generation=GENERATION, calibration_batch=calibration_batch,
                  visual_count=visual_count, per_view=per_view, cameras=cameras,
                  calibration_split='val', evaluation_split='test', overlap_policy=overlap_policy,
                  calibration_unique_frames=len({record_key(row) for row in calibration}),
                  evaluation_unique_frames=len({record_key(row) for row in evaluation}),
                  calibration_ids=[row['sample_id'] for row in calibration],
                  evaluation_ids=[row['sample_id'] for row in evaluation],
                  variants=[dict(name=f'{method}_{count}', method=method, keep=count) for method, count in variants])
    _write_json(output / 'config.json', config)
    # Scaling fixed-position indices to [0, 1] improves numerical conditioning.
    position = np.linspace(0, 1, visual_count)
    observed = _mean_difference(session, calibration, calibration_batch, 'Fit positional bias')
    polynomial = np.polynomial.Polynomial.fit(position, observed, deg=3).convert()
    fitted = polynomial(position)
    held_out = _mean_difference(session, evaluation, calibration_batch, 'Measure held-out residuals')
    _write_json(output / 'bias_curve.json', dict(degree=3, coefficients=polynomial.coef.tolist(),
        normalized_positions=position.tolist(), calibration_observed=observed.tolist(), fitted=fitted.tolist(),
        evaluation_observed=held_out.tolist()))
    curve = torch.as_tensor(fitted, device=session.device, dtype=torch.float32)

    # Warm both cases before comparing timings; rotate execution order across questions.
    warmup = _inputs(session, [calibration[0]])
    _answer(session, warmup, curve)
    _answer(session, warmup, curve, 'mmr', min(keep_counts))
    matrix = [(None, None), *variants]
    with (output / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
        for index, row in enumerate(tqdm(evaluation, desc='Paired generation', unit='question')):
            inputs = _inputs(session, [row])
            records_for_sample = {}
            offset = index % len(matrix)
            for method, count in matrix[offset:] + matrix[:offset]:
                result, details = _answer(session, inputs, curve, method, count)
                name = 'baseline' if method is None else f'{method}_{count}'
                if details is not None:
                    result['camera_counts'] = np.bincount(np.array(details['picks']) // per_view,
                                                         minlength=len(cameras)).tolist()
                    if details['penalties'] is not None:
                        picks = details['picks'][1:]
                        penalty = np.abs(details['penalties'][1:])
                        relevance = np.abs(np.array(details['corrected'])[picks])
                        if len(picks):
                            result['mean_abs_weighted_relevance'] = float(WEIGHT * relevance.mean())
                            result['mean_abs_weighted_similarity'] = float((1 - WEIGHT) * penalty.mean())
                    if index < selection_examples:
                        result['selection'] = details
                records_for_sample[name] = result
            paired = dict(sample_id=row['sample_id'], question=row['question'], ground_truth=row['answer'],
                          variants=records_for_sample)
            stream.write(json.dumps(paired, allow_nan=False, ensure_ascii=False) + '\n')
            stream.flush()
    return output
