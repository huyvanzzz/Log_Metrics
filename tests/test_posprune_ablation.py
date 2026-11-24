from contextlib import nullcontext
import csv
from functools import partial
from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from smolvlm_attention.collector import collect_last_query_attention
from smolvlm_attention.posprune_ablation.decoder import attention_difference, generate, mmr
from smolvlm_inference.cache import CAMERA_ORDER, record_key
from smolvlm_inference.text import CachedBatchCollator
from test_attention_collector import fixture as padded_fixture
from test_smolvlm_model import fixture


def test_calibration_matches_existing_attention_collector():
    model, batch, _ = fixture()
    expected = collect_last_query_attention(model, batch)
    observed = attention_difference(model, batch, 1, 2)
    np.testing.assert_allclose(observed, (expected[:, 0] - expected[:, 1]).numpy(), atol=1e-8)


def test_mmr_uses_relevance_then_penalizes_redundancy():
    scores = torch.tensor([0.05, 0.048, 0.045])
    states = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
    indices, penalties = mmr(scores, states, 2, .9)
    assert indices.tolist() == [0, 2]
    assert penalties.tolist() == [0., 0.]


@pytest.mark.parametrize('beams', [1, 3])
def test_native_generation_uses_compact_layer_local_cache(beams):
    model, batch, _ = fixture()
    inputs = {key: batch[key] for key in ('input_ids', 'attention_mask', 'visual_features')}
    settings = dict(num_beams=beams, do_sample=False, max_new_tokens=4, min_new_tokens=4, early_stopping=beams > 1)
    tokens = model.generate_from_features(**inputs, **settings)
    curve = torch.zeros(model.visual_token_len)
    baseline, timing, _ = generate(model, inputs, settings, 1)
    assert torch.equal(tokens, baseline) and timing['first_token_step_ms'] > 0
    all_kept, _, _ = generate(model, inputs, settings, 1, method='raw', curve=curve, count=len(curve))
    assert torch.equal(tokens, all_kept)

    text = model.native.model.text_model
    layer_lengths = []
    handle = text.register_forward_hook(lambda _, args, output: layer_lengths.append(
        tuple(output.past_key_values.get_seq_length(layer) for layer in range(2))))
    try:
        pruned, _, details = generate(model, inputs, settings, 1, method='mmr', curve=curve, count=5)
    finally:
        handle.remove()
    assert pruned.shape == tokens.shape
    assert len(set(details[0]['picks'])) == 5
    width = inputs['input_ids'].shape[1]
    assert layer_lengths == [(width + i, width + i - len(curve) + 5) for i in range(4)]
    assert torch.equal(model.generate_from_features(**inputs, **settings), tokens)


@pytest.mark.parametrize('method', ['raw', 'corrected', 'mmr'])
def test_padded_batch_preserves_each_questions_selection_and_greedy_tokens(method):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2)
        model, batch, items, collate = padded_fixture(questions=('What?', 'What is visible? A road. A car.'))
    keys = ('input_ids', 'attention_mask', 'visual_features')
    settings = dict(num_beams=1, do_sample=False, max_new_tokens=5, min_new_tokens=5)
    pruning = dict(method=method, curve=torch.linspace(-1e-7, 1e-7, model.visual_token_len), count=7)
    assert batch['attention_mask'][0, 0] == 0 and batch['attention_mask'][1, 0] == 1
    batched, _, details = generate(model, {key: batch[key] for key in keys}, settings, 1, **pruning)
    assert details[0]['picks'] != details[1]['picks']
    for index, item in enumerate(items):
        solo = collate([item])
        tokens, _, selection = generate(model, {key: solo[key] for key in keys}, settings, 1, **pruning)
        assert torch.equal(batched[index], tokens[0])
        assert details[index]['picks'] == selection[0]['picks']
        assert details[index]['corrected'] == pytest.approx(selection[0]['corrected'], abs=1e-7)


def test_oom_retries_all_pending_questions_with_a_smaller_batch(monkeypatch):
    from smolvlm_attention.posprune_ablation import runtime

    session = SimpleNamespace(device=torch.device('cuda:0'))
    calls, cleared = [], []
    monkeypatch.setattr(runtime.torch, 'as_tensor', lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime, '_inputs', lambda *args: None)
    monkeypatch.setattr(runtime, '_answers', lambda *args: None)
    monkeypatch.setattr(torch.cuda, 'device', lambda *args: nullcontext())
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: cleared.append(True))

    def generate_batch(session, batch, curve, matrix, offset, selection_examples):
        indices = [index for index, _ in batch]
        calls.append(indices)
        if len(indices) > 2:
            raise torch.cuda.OutOfMemoryError('fixture')
        return indices

    monkeypatch.setattr(runtime, '_generate_batch', generate_batch)
    rows = enumerate([{}] * 5)
    assert list(runtime._generate_rows(session, rows, [0], {}, (5,), 3, 4)) == list(range(5))
    assert calls == [[0, 1, 2, 3], [0, 1], [2, 3], [4]] and cleared == [True]


def test_finished_question_does_not_stop_its_batch_and_padding_is_not_counted(monkeypatch):
    from smolvlm_attention.posprune_ablation import runtime

    model, batch, _, _ = padded_fixture(questions=('What?', 'What is visible?'))
    eos = model.native.generation_config.eos_token_id

    def force_tokens(input_ids, scores):
        scores.fill_(-torch.inf)
        scores[0, eos] = 0
        scores[1, eos + 1] = 0
        return scores

    monkeypatch.setattr(runtime, 'SHALLOW', 1)
    monkeypatch.setattr(runtime, 'GENERATION', dict(runtime.GENERATION, max_new_tokens=4, logits_processor=[force_tokens]))
    session = SimpleNamespace(model=model, processor=model.processor)
    inputs = {key: batch[key] for key in ('input_ids', 'attention_mask', 'visual_features')}
    results, details = runtime._answers(session, inputs, torch.zeros(model.visual_token_len), 'raw', 5)
    assert [row['generated_tokens'] for row in results] == [1, 4]
    assert len(details) == 2


def test_paired_run_exports_all_figures_and_selection_indices(tmp_path, monkeypatch):
    from smolvlm_attention.posprune_ablation import report, runtime

    model, _, fixture_item = fixture()
    frames = [dict(sample_id=f'id-{i}', scene_id=scene, question='What is visible?',
                   answer='A road.', camera_paths={camera: f'{i}/{camera}.jpg' for camera in CAMERA_ORDER})
              for i, scene in enumerate(['calibration', 'evaluation', 'evaluation', 'evaluation'])]
    for split, rows in [('val', frames[:2]), ('test', frames[2:])]:
        for index, row in enumerate(rows):
            row['eval_id'] = index
        (tmp_path / f'{split}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    label_path = tmp_path / 'question_types.json'
    label_path.write_text(json.dumps(dict(
        prepared_sha256=sha256((tmp_path / 'test.jsonl').read_bytes()).hexdigest(),
        question_types=['perception', 'planning'], counts=dict(perception=1, planning=1),
        labels_by_eval_id='01')), encoding='utf-8')
    features = np.stack([fixture_item['visual_features'].numpy().astype(np.float16)] * len(frames))
    cache = SimpleNamespace(prepared=tmp_path, index={record_key(row): i for i, row in enumerate(frames)}, array=features)
    session = SimpleNamespace(model=model, cache=cache, device=torch.device('cpu'), processor=model.processor,
                              describe=lambda: dict(visual_tokens=model.visual_token_len, camera_order=CAMERA_ORDER),
                              collator=CachedBatchCollator(model.processor, image_seq_len=4, max_length=512,
                                                           max_new_tokens=3))
    monkeypatch.setattr(runtime, 'SHALLOW', 1)
    monkeypatch.setattr(runtime, 'DEEP', 2)
    monkeypatch.setattr(runtime, 'GENERATION', dict(runtime.GENERATION, max_new_tokens=3))
    monkeypatch.setattr(runtime, 'stratified_test', partial(runtime.stratified_test, labels_path=label_path))
    output = runtime.run(session, tmp_path / 'results', calibration_size=1, evaluation_size=None,
                         evaluation_fraction=1.0, keep_counts=(5,))
    config = json.loads((output / 'config.json').read_text(encoding='utf-8'))
    assert config['generation']['num_beams'] == 1 and not config['generation']['do_sample']
    assert config['calibration_ids'] == ['id-0']
    assert config['evaluation_ids'] == ['id-2', 'id-3']
    monkeypatch.setattr(report, '_quality', lambda rows, names: {name: {'overall': 1.} for name in names})
    paths = report.report(output)
    assert len(paths) == 8 and all(path.stat().st_size for path in paths)
    assert len((output / 'indexed_selections.csv').read_text(encoding='utf-8').splitlines()) == 31
    summary = list(csv.DictReader((output / 'comparison.csv').read_text(encoding='utf-8').splitlines()))
    assert all(row['measured_batches'] == '1' and row['mean_batch_size'] == '2.0' for row in summary)
    answers = list(csv.DictReader((output / 'paired_answers.csv').read_text(encoding='utf-8').splitlines()))
    assert [row['question_type'] for row in answers] == ['perception', 'planning'] and 'mmr_5' in answers[0]
    distribution = list(csv.DictReader((output / 'question_type_distribution.csv').read_text().splitlines()))
    assert [row['selected_count'] for row in distribution] == ['1', '1']


def test_half_split_preserves_question_type_proportions_and_pinned_identity(tmp_path):
    from collections import Counter
    from smolvlm_attention.posprune_ablation.selection import stratified_test

    test = [dict(eval_id=i, sample_id=f'test-{i}') for i in range(21)]
    prepared = tmp_path / 'test.jsonl'
    payload = ''.join(json.dumps(row) + '\n' for row in test).encode()
    prepared.write_bytes(payload)
    labels = tmp_path / 'question_types.json'
    labels.write_text(json.dumps(dict(prepared_sha256=sha256(payload).hexdigest(),
        question_types=['behavior', 'perception', 'planning', 'prediction'],
        counts=dict(behavior=3, perception=8, planning=4, prediction=6),
        labels_by_eval_id='000' + '1' * 8 + '2' * 4 + '3' * 6)), encoding='utf-8')

    chosen, distribution = stratified_test(test, prepared, .5, 42, labels_path=labels)
    indices = [row['eval_id'] for row in chosen]
    assert len(indices) == len(set(indices)) == 10
    assert indices == sorted(indices)
    assert (chosen, distribution) == stratified_test(test, prepared, .5, 42, labels_path=labels)
    assert Counter(row['question_type'] for row in chosen) == dict(behavior=1, perception=4, planning=2, prediction=3)
    assert {row['question_type']: row['selected_count'] for row in distribution} == Counter(
        row['question_type'] for row in chosen)

    prepared.write_bytes(payload + b'\n')
    with pytest.raises(ValueError, match='do not match'):
        stratified_test(test, prepared, .5, 42, labels_path=labels)


def test_two_gpu_coordination_fits_one_weighted_curve_and_merges_in_order(tmp_path, monkeypatch):
    from pathlib import Path
    from smolvlm_attention.posprune_ablation import parallel

    rows = [dict(sample_id=str(i), camera_paths={camera: f'{i}/{camera}.jpg' for camera in CAMERA_ORDER})
            for i in range(8)]
    monkeypatch.setattr(parallel, 'select_rows', lambda *args, **kwargs: (rows[:3], rows[3:], 'fixture', None))
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    validations = []

    def validate(*args, **kwargs):
        validations.append(kwargs['expected_fingerprint'])
        return tmp_path / 'receipt.json'

    jobs = []

    class FakeProcess:
        def __init__(self, command, **kwargs):
            self.folder = Path(command[-1])
            self.job = parallel._read(self.folder / 'job.json')
            jobs.append(self.job)
            rank = self.job['rank']
            self.bias = self.folder.parent.parent / 'bias_curve.json'
            self.done = False
            parallel._json_write(self.folder / 'prefill.json', dict(
                session=dict(device=f'cuda:{rank}', camera_order=CAMERA_ORDER, visual_tokens=24), per_view=4,
                gpu='T4 fixture', observed=[float(rank + 1)] * 24, held_out=[float(rank + 1)] * 24))

        def poll(self):
            if self.bias.exists() and not self.done:
                with (self.folder / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
                    for _, row in self.job['evaluation']:
                        stream.write(json.dumps({'sample_id': row['sample_id']}) + '\n')
                self.done = True
            return 0 if self.done else None

        def wait(self):
            assert self.done

    monkeypatch.setattr(parallel, 'validate_cache', validate)
    monkeypatch.setattr(parallel.subprocess, 'Popen', FakeProcess)
    output = parallel.run_parallel('checkpoint.pt', tmp_path, tmp_path, tmp_path / 'results',
                                   cache_fingerprint='expected', camera_order=CAMERA_ORDER, keep_counts=(5,))
    assert validations == ['expected']
    assert [job['generation_batch'] for job in jobs] == [4, 4]
    assert [[index for index, _ in job['evaluation']] for job in jobs] == [[0, 2, 4], [1, 3]]
    assert parallel._read(output / 'bias_curve.json')['fitted'] == pytest.approx([4 / 3] * 24)
    assert parallel._read(output / 'bias_curve.json')['evaluation_observed'] == pytest.approx([1.4] * 24)
    assert [json.loads(line)['sample_id'] for line in (output / 'predictions.jsonl').read_text().splitlines()] == list('34567')
    assert parallel._read(output / 'parallel_timing.json')['workers'] == 2
