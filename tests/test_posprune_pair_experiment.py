import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from smolvlm_attention.collector import collect_last_query_attention
from smolvlm_attention.posprune_ablation.pair_similarity import (GROUPS, choose_pairs, correlations,
    generate_pair_outputs, js_divergence, measure_pairs)
from test_attention_collector import fixture


def test_incoming_patterns_match_native_last_query_and_cover_unique_pairs():
    model, batch, _, _ = fixture(questions=('What?', 'What is visible? A road. A car.'))
    expected = collect_last_query_attention(model, batch)
    result = measure_pairs(model, batch, layers=(1, 2), query_limit=16)
    for layer, (similarities, mass) in result.items():
        np.testing.assert_allclose(mass, expected[:, layer - 1].numpy(), atol=1e-8)
        assert similarities.shape == (2, 2, math.comb(model.visual_token_len, 2))
        assert np.isfinite(similarities).all()
        assert all(abs(correlations(*row)['spearman']) <= 1 for row in similarities)


def test_pair_selection_covers_four_strata_with_distinct_removals():
    rng = np.random.default_rng(58)
    tokens = 24
    similarities = rng.uniform(0, 1, (2, math.comb(tokens, 2)))
    mass = rng.uniform(0, .1, tokens)
    selected = choose_pairs(similarities, mass, 5)
    assert selected == choose_pairs(similarities, mass, 5)
    assert set(selected) == set(GROUPS)
    assert len({row['dropped'] for row in selected.values()}) == 4
    for group, row in selected.items():
        assert row['dropped'] != row['kept']
        for measure, high in [('hidden', group.startswith('high_hidden')),
                              ('attention', group.endswith('high_attention'))]:
            index = 0 if measure == 'hidden' else 1
            low, upper = np.quantile(similarities[index], [.25, .75])
            score = row[f'{measure}_similarity']
            assert score >= upper if high else score <= low
    assert not choose_pairs(np.ones_like(similarities), mass, 5)


def test_batched_ablation_removes_one_key_only_after_the_selected_layer():
    model, batch, _, _ = fixture(questions=('What?', 'What is visible? A road. A car.'))
    inputs = {key: batch[key] for key in ('input_ids', 'attention_mask', 'visual_features')}
    settings = dict(num_beams=1, do_sample=False, max_new_tokens=3, min_new_tokens=3)
    cache_lengths = []
    text = model.native.model.text_model
    handle = text.register_forward_hook(lambda _, args, result: cache_lengths.append(
        tuple(result.past_key_values.get_seq_length(i) for i in range(2))))
    try:
        tokens, scores = generate_pair_outputs(model, inputs, torch.tensor([[1], [7]]), layer=1, generation=settings)
    finally:
        handle.remove()
    width = inputs['input_ids'].shape[1]
    assert cache_lengths == [(width + i, width - 1 + i) for i in range(3)]
    base, base_scores = generate_pair_outputs(model, inputs, generation=settings)
    assert tokens.shape == base.shape == (2, 3)
    assert np.isfinite(js_divergence(scores, base_scores).numpy()).all()
    assert js_divergence(scores, scores).tolist() == [0, 0]
    assert js_divergence(torch.tensor([[0., 1.]]), torch.tensor([[1., 0.]])).item() == pytest.approx(math.log(2))


def test_report_keeps_all_correlation_rows_and_counts_only_paired_ablations(tmp_path):
    from smolvlm_attention.posprune_ablation.pair_report import report_pairs

    layers = [3, 15, 22, 26]
    config = dict(sample_ids=['a', 'b'], layers=layers, groups=GROUPS, drop_after_layer=22)
    (tmp_path / 'config.json').write_text(json.dumps(config), encoding='utf-8')
    ablated = dict(first_token_jsd=.003, first_token_changed=False, answer_changed=True,
                   dropped_raw_attention=.001, dropped_camera='Front', kept_camera='Back')
    rows = [dict(sample_id=name, question_type='planning', baseline='A car.',
                 correlations={str(layer): dict(pearson=.1, spearman=.2) for layer in layers},
                 ablations={group: ablated for group in GROUPS} if name == 'a' else {}) for name in ['a', 'b']]
    (tmp_path / 'results.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    figures = report_pairs(tmp_path)
    assert len(figures) == 4 and all(p.stat().st_size for p in figures)
    assert len((tmp_path / 'correlation_by_question.csv').read_text().splitlines()) == 9
    assert len((tmp_path / 'ablation_by_question.csv').read_text().splitlines()) == 5


def test_kaggle_workers_share_prepared_features_and_merge_ordered_questions(tmp_path, monkeypatch):
    from smolvlm_attention.posprune_ablation import pair_experiment as experiment
    from smolvlm_inference.cache import CAMERA_ORDER

    rows = [dict(sample_id=str(i), camera_paths={camera: f"data/nuscenes/samples/CAM_{camera.upper().replace('-', '_')}/{i}.jpg"
                                                 for camera in CAMERA_ORDER}) for i in range(3)]
    monkeypatch.setattr(experiment, 'select_evaluation', lambda *args, **kwargs: (rows, []))
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    (tmp_path / 'manifest.json').write_text('{"counts":{"test":3}}', encoding='utf-8')
    jobs = []

    class Worker:
        def __init__(self, command, **kwargs):
            self.folder = Path(command[-1])
            self.job = experiment._read(self.folder / 'job.json')
            jobs.append(self.job)
            self.done = False
            experiment._json_write(self.folder / 'features_ready.json', dict(device=f"cuda:{self.job['rank']}"))

        def poll(self):
            if (self.folder.parent / 'features_ready.json').is_file() and not self.done:
                (self.folder / 'predictions.jsonl').write_text(''.join(
                    json.dumps(dict(sample_id=row['sample_id'])) + '\n' for _, row in self.job['rows']), encoding='utf-8')
                self.done = True
            return 0 if self.done else None

        def wait(self):
            assert self.done

    monkeypatch.setattr(experiment.subprocess, 'Popen', Worker)
    output = experiment.run_pair_experiment('checkpoint.pt', tmp_path, tmp_path / 'output',
                                            camera_order=CAMERA_ORDER, questions=3)
    config = experiment._read(output / 'config.json')
    assert config['layers'] == [3, 15, 22, 26] and config['drop_after_layer'] == 22
    assert [len(job['rows']) for job in jobs] == [2, 1]
    assert jobs[0]['features'] == jobs[1]['features']
    assert [json.loads(line)['sample_id'] for line in (output / 'results.jsonl').read_text().splitlines()] == list('012')
