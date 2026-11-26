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


def test_raw_mmr_uses_uncorrected_attention_without_a_bias_curve():
    model, batch, _ = fixture()
    inputs = {key: batch[key] for key in ('input_ids', 'attention_mask', 'visual_features')}
    settings = dict(num_beams=1, max_new_tokens=3, min_new_tokens=3)
    raw, _, details = generate(model, inputs, settings, 1, method='raw_mmr', count=5)
    reference, _, zero_bias = generate(model, inputs, settings, 1, method='mmr', count=5,
                                      curve=torch.zeros(model.visual_token_len))
    assert torch.equal(raw, reference)
    assert details[0]['corrected'] is None
    assert details[0]['picks'] == zero_bias[0]['picks']
    assert details[0]['penalties'] == zero_bias[0]['penalties']


def test_attention_mmr_uses_attention_patterns_and_preserves_final_query_relevance(monkeypatch):
    from smolvlm_attention.posprune_ablation import decoder

    model, batch, _, _ = padded_fixture(questions=('What?', 'What is visible? A road. A car.'))
    inputs = {key: batch[key] for key in ('input_ids', 'attention_mask', 'visual_features')}
    settings = dict(num_beams=1, max_new_tokens=2, min_new_tokens=2)
    native_mmr = decoder.mmr
    widths = []

    def spy(relevance, features, count, weight, **kwargs):
        widths.append(features.shape[-1])
        return native_mmr(relevance, features, count, weight, **kwargs)

    monkeypatch.setattr(decoder, 'mmr', spy)
    _, _, raw = generate(model, inputs, settings, 1, method='raw_mmr', count=5, attention_normalization='minmax')
    _, _, incoming = generate(model, inputs, settings, 1, method='attention_mmr', count=5, attention_normalization='minmax')
    text = model.native.model.text_model
    assert widths == [text.config.hidden_size, text.layers[0].self_attn.num_heads * decoder.ATTENTION_QUERY_LIMIT]
    for a, b in zip(raw, incoming, strict=True):
        np.testing.assert_allclose(a['raw'], b['raw'], atol=1e-8)
        assert a['picks'][0] == b['picks'][0] and len(set(b['picks'])) == 5


def test_lambda_sweep_dispatches_and_reports_all_methods_at_their_own_weights(tmp_path, monkeypatch):
    from smolvlm_attention.posprune_ablation import report, raw_mmr_report, runtime

    variants = runtime.mmr_sweep_variants((32,), (.5, .7, .9))
    assert len(variants) == len({row['name'] for row in variants}) == 13
    assert variants[0]['name'] == 'raw_32' and variants[0]['mmr_lambda'] is None
    assert {row['method'] for row in variants[1:]} == {
        'raw_mmr', 'intra_camera_mmr', 'spatial_aware_mmr', 'attention_mmr'}
    session = SimpleNamespace(model=SimpleNamespace(geometry={'visual_tokens_per_view': 64}, prompt_camera_order=CAMERA_ORDER))
    monkeypatch.setattr(runtime, '_inputs', lambda *_: {})
    dispatched = []

    def answer(session, inputs, curve, method, count, *, mmr_weight, attention_normalization):
        dispatched.append((method, count, mmr_weight, attention_normalization))
        scores = dict(picks=[0, 1], raw=[.01, .05], relevance=[0., 1.], penalties=None if method == 'raw' else [0., .3])
        return [dict(prediction='A car.', generated_tokens=3, latency_ms=10, first_token_step_ms=4)], [scores]

    monkeypatch.setattr(runtime, '_answers', answer)
    sample = dict(sample_id='question', question='What?', answer='A car.')
    rows = runtime._generate_batch(session, [(0, sample)], None, variants, 0, 1, attention_normalization='minmax')
    for (method, keep, weight, scale), variant in zip(dispatched, variants, strict=True):
        assert (method, keep, scale) == (variant['method'], 32, 'minmax')
        if method != 'raw':
            assert weight == variant['mmr_lambda']
    config = dict(experiment='raw_mmr', variants=variants, evaluation_ids=['question'], visual_count=384,
        cameras=CAMERA_ORDER, per_view=64, question_type_distribution=None, mmr_lambdas=[.5, .7, .9])
    (tmp_path / 'config.json').write_text(json.dumps(config), encoding='utf-8')
    (tmp_path / 'predictions.jsonl').write_text(json.dumps(rows[0]) + '\n', encoding='utf-8')
    metrics = dict.fromkeys(('overall', 'bleu4', 'meteor', 'rouge_l', 'cider', 'yes_no_accuracy'), 1.)
    monkeypatch.setattr(report, '_quality', lambda rows, names: {name: metrics.copy() for name in names})
    assert len(raw_mmr_report.report_raw_mmr(tmp_path)) == 2
    with (tmp_path / 'mmr_vs_raw.csv').open() as stream:
        comparisons = list(csv.DictReader(stream))
    assert [(row['method'], float(row['mmr_lambda'])) for row in comparisons] == [
        (variant['method'], variant['mmr_lambda']) for variant in variants[1:]]


def test_attention_minmax_scales_each_prompt_and_handles_flat_scores():
    from smolvlm_attention.posprune_ablation.decoder import minmax_attention

    scores = torch.tensor([[.005, .010, .025], [.04, .02, .06], [.2, .2, .2]])
    expected = torch.tensor([[0., .25, 1.], [.5, 0., 1.], [0., 0., 0.]])
    torch.testing.assert_close(minmax_attention(scores), expected)


@pytest.mark.parametrize('method', ['raw', 'raw_mmr', 'intra_camera_mmr', 'spatial_aware_mmr'])
def test_minmax_relevance_is_used_by_raw_mmr_variants_only(method):
    from smolvlm_attention.posprune_ablation.decoder import minmax_attention

    model, batch, _, _ = padded_fixture(questions=('What?', 'What is visible? A road. A car.'))
    inputs = {key: batch[key] for key in ('input_ids', 'attention_mask', 'visual_features')}
    settings = dict(num_beams=1, max_new_tokens=2, min_new_tokens=2)
    kwargs = dict(method=method, count=5, weight=.75)
    tokens, _, details = generate(model, inputs, settings, 1, attention_normalization='minmax', **kwargs)
    raw = torch.tensor([row['raw'] for row in details])
    used = torch.tensor([row['relevance'] for row in details])
    torch.testing.assert_close(used, raw if method == 'raw' else minmax_attention(raw))
    if method == 'raw':
        previous, _, reference = generate(model, inputs, settings, 1, **kwargs)
        assert torch.equal(tokens, previous) and details == reference


def test_intra_camera_mmr_selects_globally_and_applies_no_cross_camera_penalty():
    relevance = torch.tensor([1., .9, .1, .0, .8, .01, .02, .03])
    states = torch.ones((8, 2))
    raw_picks, _ = mmr(relevance, states, 3, .9)
    picks, penalties = mmr(relevance, states, 3, .9, method='intra_camera_mmr', per_view=4)
    assert raw_picks.tolist() == [0, 1, 4]
    assert picks.tolist() == [0, 4, 1]
    assert penalties.tolist() == pytest.approx([0, 0, 1])


def test_spatial_mmr_weights_cosine_by_grid_distance_and_camera():
    from smolvlm_attention.posprune_ablation.decoder import _camera_similarity

    similarities = torch.ones((2, 8, 8))
    weighted = _camera_similarity(similarities, 4, 'spatial_aware_mmr')
    assert weighted[0, 0, 0].item() == 1
    assert weighted[0, 0, 1].item() == pytest.approx(.5 + .5 * np.exp(-1 / 8))
    assert weighted[0, 0, 3].item() == pytest.approx(.5 + .5 * np.exp(-2 / 8))
    assert weighted[0, 0, 4].item() == .5
    assert torch.equal(weighted, weighted.transpose(-1, -2))


@pytest.mark.parametrize('method', ['intra_camera_mmr', 'spatial_aware_mmr'])
def test_camera_mmr_prunes_padded_batches_without_positional_correction(method):
    model, batch, _, _ = padded_fixture(questions=('What?', 'What is visible? A road. A car.'))
    inputs = {key: batch[key] for key in ('input_ids', 'attention_mask', 'visual_features')}
    output, _, details = generate(model, inputs, dict(num_beams=1, max_new_tokens=2, min_new_tokens=2),
                                  1, method=method, count=5)
    assert output.shape == (2, 2)
    assert all(item['corrected'] is None and len(set(item['picks'])) == 5 for item in details)
    assert all(np.isfinite(item['penalties']).all() for item in details)


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

    def generate_batch(session, batch, curve, matrix, offset, selection_examples, **kwargs):
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


def test_subset_halves_the_pinned_cohort_counts_without_changing_order():
    from collections import Counter
    from smolvlm_attention.posprune_ablation.selection import stratified_subset

    counts = {'behavior': 2, 'perception': 6, 'planning': 4, 'prediction': 4}
    labels = [name for name, count in counts.items() for _ in range(count)]
    rows = [dict(sample_id=i, question_type=name) for i, name in enumerate(labels)]
    distribution = [dict(question_type=name, full_count=2 * count, selected_count=count)
                    for name, count in counts.items()]
    selected, summary = stratified_subset(rows, distribution, 8, 42)
    ids = [row['sample_id'] for row in selected]
    assert ids == sorted(set(ids)) and len(ids) == 8
    assert selected == [rows[i] for i in ids]
    assert Counter(row['question_type'] for row in selected) == {name: n // 2 for name, n in counts.items()}
    assert [row['parent_count'] for row in summary] == list(counts.values())
    assert (selected, summary) == stratified_subset(rows, distribution, 8, 42)


@pytest.mark.parametrize('mount', ['cache-smolvlm', 'datasets/huyhuy12334/cache-smolvlm', 'custom-mount'])
def test_cache_input_resolves_already_mounted_pinned_cache(tmp_path, mount):
    from smolvlm_attention.posprune_ablation.cache_input import resolve_cache_input
    from smolvlm_inference.cache import FEATURE_FILE

    cache = tmp_path / mount / 'features'
    cache.mkdir(parents=True)
    (cache / FEATURE_FILE).touch()
    (cache / 'index.json').write_text('{}', encoding='utf-8')
    (cache / 'manifest.json').write_text('{"fingerprint":"pinned"}', encoding='utf-8')
    args = dict(dataset='huyhuy12334/cache-smolvlm/versions/1', fingerprint='pinned')
    assert resolve_cache_input(tmp_path, '', **args) == cache.resolve()
    assert resolve_cache_input(tmp_path / 'unavailable', str(cache), **args) == cache.resolve()

    (cache / 'manifest.json').write_text('{"fingerprint":"other"}', encoding='utf-8')
    with pytest.raises(ValueError, match='none match the pinned cache'):
        resolve_cache_input(tmp_path, '', **args)


def test_missing_cache_explains_noninteractive_dataset_attachment(tmp_path):
    from smolvlm_attention.posprune_ablation.cache_input import resolve_cache_input

    with pytest.raises(FileNotFoundError, match='Attach huyhuy12334/cache-smolvlm version 1'):
        resolve_cache_input(tmp_path, '', dataset='huyhuy12334/cache-smolvlm/versions/1', fingerprint='pinned')


@pytest.mark.parametrize('camera_mmr', [False, True])
def test_raw_mmr_report_compares_variants_without_mixing_historical_latency(tmp_path, monkeypatch, camera_mmr):
    from smolvlm_attention.posprune_ablation import report, raw_mmr_report

    methods = ('raw', 'raw_mmr', 'intra_camera_mmr', 'spatial_aware_mmr') if camera_mmr else ('raw', 'raw_mmr')
    budgets = (32,) if camera_mmr else (96, 32)
    variants = [dict(name=f'{method}_{keep}', method=method, keep=keep,
                     source='historical' if (method, keep) == ('raw', 96) else 'current')
                for keep in budgets for method in methods]
    config = dict(experiment='raw_mmr', variants=variants, evaluation_ids=['question'], visual_count=384,
                  cameras=CAMERA_ORDER, per_view=64, question_type_distribution=None)
    if camera_mmr:
        config.update(attention_normalization='minmax', weight=.75)
    (tmp_path / 'config.json').write_text(json.dumps(config), encoding='utf-8')
    selection = dict(picks=[0], raw=[.1], penalties=None)
    measured = dict(prediction='A car.', generated_tokens=3, latency_ms=10,
                    first_token_step_ms=4, camera_counts=[1, 0, 0, 0, 0, 0], selection=selection)
    row = dict(sample_id='question', question='What?', ground_truth='A car.',
               variants={variant['name']: measured for variant in variants if variant['source'] == 'current'})
    (tmp_path / 'predictions.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')
    historical = dict(evaluation_ids=['question'], predictions=[{
        key: measured[key] for key in ('prediction', 'generated_tokens', 'camera_counts', 'selection')}])

    def quality(rows, names):
        return {name: dict.fromkeys(('overall', 'bleu4', 'meteor', 'rouge_l', 'cider', 'yes_no_accuracy'),
                                   2. if 'mmr' in name else 1.) for name in names}

    monkeypatch.setattr(report, '_quality', quality)
    assert len(raw_mmr_report.report_raw_mmr(tmp_path, None if camera_mmr else historical)) == 2
    with (tmp_path / 'mmr_vs_raw.csv').open() as stream:
        deltas = list(csv.DictReader(stream))
    if camera_mmr:
        assert [r['method'] for r in deltas] == list(methods[1:])
        assert all(r['kept_tokens'] == '32' and r['delta_median_latency_ms'] == '0' for r in deltas)
        assert all(r['attention_normalization'] == 'minmax' and r['mmr_lambda'] == '0.75' for r in deltas)
    else:
        first, second = deltas
        assert first['raw_source'] == 'historical' and first['delta_overall'] == '1.0'
        assert first['delta_median_latency_ms'] == '' and second['delta_median_latency_ms'] == '0'


def test_public_image_path_builds_reusable_fp16_features_in_camera_order(tmp_path, monkeypatch):
    from PIL import Image
    from smolvlm_attention.posprune_ablation import image_features

    model, _, _ = fixture()
    row = dict(camera_paths={camera: f"data/nuscenes/samples/CAM_{camera.upper().replace('-', '_')}/{i}.jpg"
                             for i, camera in enumerate(CAMERA_ORDER)})
    frames = image_features.frame_plan([row, row])
    assert len(frames) == 1
    assert [path.rsplit('/', 1)[-1] for path in frames[0][1]] == [f'{i}.jpg' for i in range(6)]

    def fetch(path):
        i = int(path.rsplit('/', 1)[-1].split('.')[0])
        return Image.new('RGB', (16, 16), (i * 40, 20, 100))

    monkeypatch.setattr(image_features, '_fetch_image', fetch)
    image_features.build_frames(model, frames, tmp_path)
    cache = image_features.image_cache(tmp_path, frames, {'val': 1, 'test': 1})
    features = cache.array[cache.index[record_key(row)]]
    assert features.shape == (6, 16, 8) and features.dtype == np.float16
    reference = image_features.encode_frame(model, [fetch(path) for path in frames[0][1]])
    np.testing.assert_array_equal(features, reference)


@pytest.mark.parametrize(('mounted', 'raw_mmr', 'camera_mmr'),
                         [(True, False, False), (False, False, False), (False, True, False), (False, True, True)])
def test_two_gpu_coordination_fits_one_weighted_curve_and_merges_in_order(tmp_path, monkeypatch, mounted, raw_mmr,
                                                                        camera_mmr):
    from pathlib import Path
    from smolvlm_attention.posprune_ablation import parallel

    rows = [dict(sample_id=str(i), camera_paths={
        camera: f"data/nuscenes/samples/CAM_{camera.upper().replace('-', '_')}/{i}.jpg" for camera in CAMERA_ORDER})
            for i in range(8)]
    (tmp_path / 'manifest.json').write_text('{"counts":{"val":3,"test":5}}', encoding='utf-8')
    monkeypatch.setattr(parallel, 'select_rows', lambda *args, **kwargs: (rows[:3], rows[3:], 'fixture', None))
    monkeypatch.setattr(parallel, 'select_evaluation', lambda *args, **kwargs: (rows[3:], None))
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
            self.bias = self.folder.parent.parent / ('generation_ready.json' if raw_mmr else 'bias_curve.json')
            self.done = False
            if self.job['features'] is not None:
                parallel._json_write(self.folder / 'features_ready.json', True)
            parallel._json_write(self.folder / 'prefill.json', dict(
                session=dict(device=f'cuda:{rank}', camera_order=CAMERA_ORDER, visual_tokens=24), per_view=4,
                gpu='T4 fixture', observed=[float(rank + 1)] * 24, held_out=[float(rank + 1)] * 24))

        def poll(self):
            if self.bias.exists() and not self.done:
                if self.job['features'] is not None:
                    assert (self.folder.parent / 'features_ready.json').is_file()
                with (self.folder / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
                    for _, row in self.job['evaluation']:
                        stream.write(json.dumps({'sample_id': row['sample_id']}) + '\n')
                self.done = True
            return 0 if self.done else None

        def wait(self):
            assert self.done

    monkeypatch.setattr(parallel, 'validate_cache', validate)
    monkeypatch.setattr(parallel.subprocess, 'Popen', FakeProcess)
    budgets = (32,) if camera_mmr else (96, 32) if raw_mmr else (5,)
    output = parallel.run_parallel('checkpoint.pt', tmp_path if mounted else None, tmp_path, tmp_path / 'results',
        cache_fingerprint='expected', camera_order=CAMERA_ORDER, keep_counts=budgets,
        raw_mmr=raw_mmr, reuse_raw_96=raw_mmr and not camera_mmr,
        expected_evaluation_ids=list('34567'), camera_mmr=camera_mmr,
        mmr_weight=.75 if camera_mmr else .9, attention_normalization='minmax' if camera_mmr else 'none')
    assert validations == (['expected'] if mounted else [])
    source = parallel._read(output / 'config.json')['feature_source']
    assert source['mode'] == ('mounted' if mounted else 'regenerated')
    if not mounted:
        assert source['unique_frames'] == (5 if raw_mmr else 8)
        assert len(jobs[0]['features']['frames']) == source['unique_frames']
        assert not Path(jobs[0]['features']['directory']).exists()
    assert [job['generation_batch'] for job in jobs] == [4, 4]
    assert [[index for index, _ in job['evaluation']] for job in jobs] == [[0, 2, 4], [1, 3]]
    if raw_mmr:
        assert not (output / 'bias_curve.json').exists()
        assert all(job['calibration'] == [] for job in jobs)
        if camera_mmr:
            assert jobs[0]['matrix'] == [[name, 32] for name in
                                        ('raw', 'raw_mmr', 'intra_camera_mmr', 'spatial_aware_mmr')]
            assert parallel._read(output / 'config.json')['spatial_redundancy'] == {'alpha': .5, 'sigma_grid_cells': 2.}
            assert all(job['mmr_weight'] == .75 and job['attention_normalization'] == 'minmax' for job in jobs)
        else:
            assert jobs[0]['matrix'] == [['raw_mmr', 96], ['raw', 32], ['raw_mmr', 32]]
    else:
        assert parallel._read(output / 'bias_curve.json')['fitted'] == pytest.approx([4 / 3] * 24)
        assert parallel._read(output / 'bias_curve.json')['evaluation_observed'] == pytest.approx([1.4] * 24)
    assert [json.loads(line)['sample_id'] for line in (output / 'predictions.jsonl').read_text().splitlines()] == list('34567')
    assert parallel._read(output / 'parallel_timing.json')['workers'] == 2
