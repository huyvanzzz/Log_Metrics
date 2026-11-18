import math

import pytest
import torch

from test_attention_collector import fixture


def content_ids(model):
    return model.processor.tokenizer.encode('A road.', add_special_tokens=False)


def core():
    from smolvlm_probing import measurement
    return measurement


def test_replay_preserves_specials_and_original_query_indices():
    m = core()
    model, batch, _, _ = fixture()
    tokens = content_ids(model)
    generated = [tokens[:1] + [model.processor.tokenizer.bos_token_id] + tokens[1:]]
    replay = m.prepare_replay(model, batch, generated)
    positions = replay['query_positions'][0]
    assert replay['input_ids'][0, positions].tolist() == generated[0]
    assert replay['query_valid'][0].tolist() == [True, False] + [True] * (len(tokens)-1)
    assert replay['content_mask'][0].tolist() == replay['query_valid'][0].tolist()
    assert replay['attention_mask'][0, positions[1]].item() == 1
    with pytest.raises(ValueError, match='reserved image'):
        m.prepare_replay(model, batch, [[model.native.config.image_token_id]])


def test_shifted_scores_match_native_loss_and_padding_singles():
    m = core()
    model, batch, items, collator = fixture(questions=('What?', 'What is visible? A road. A car.'))
    tokens = content_ids(model)
    generated = [tokens, tokens[:1] + [model.processor.tokenizer.bos_token_id] + tokens[1:]]
    replay = m.prepare_replay(model, batch, generated)
    scores = m.score_replay(model, replay, logit_chunk_size=1)
    labels = replay['input_ids'].new_full(replay['input_ids'].shape, -100)
    for i in range(2):
        q = replay['query_positions'][i, replay['query_valid'][i]]
        labels[i, q] = replay['input_ids'][i, q]
    with torch.inference_mode():
        native = model.native(input_ids=replay['input_ids'], inputs_embeds=replay['fused'],
                              attention_mask=replay['attention_mask'], position_ids=replay['position_ids'],
                              labels=labels, use_cache=False)
    weighted = sum(s['mean_logprob'] * s['token_count'] for s in scores) / sum(s['token_count'] for s in scores)
    assert weighted == pytest.approx(-native.loss.item(), abs=1e-6)
    for i, item in enumerate(items):
        single = m.score_replay(model, m.prepare_replay(model, collator([item]), [generated[i]]))[0]
        assert scores[i]['token_logprobs'] == pytest.approx(single['token_logprobs'], abs=1e-6)
        assert scores[i]['mean_logprob'] == pytest.approx(single['mean_logprob'], abs=1e-6)


def test_zero_ablation_changes_only_selected_visual_vectors_and_never_refuses(monkeypatch):
    m = core()
    model, batch, _, _ = fixture()
    replay = m.prepare_replay(model, batch, [content_ids(model)])
    original = replay['fused'].clone()
    def forbidden(*a, **kw):
        raise AssertionError('fusion/vision must not rerun')
    monkeypatch.setattr(model.native.model, 'inputs_merger', forbidden)
    monkeypatch.setattr(model.native.model.connector, 'forward', forbidden)
    monkeypatch.setattr(model.native.model.vision_model, 'forward', forbidden)
    seen = []
    def capture(module, args, kw):
        assert 'input_ids' not in kw and kw['use_cache'] is False
        assert 'past_key_values' not in kw
        assert torch.equal(kw['attention_mask'], replay['attention_mask'])
        assert torch.equal(kw['position_ids'], replay['position_ids'])
        seen.append(kw['inputs_embeds'].clone())
    handle = replay['text'].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        baseline = m.score_replay(model, replay)
        assert m.score_replay(model, replay, zero_indices=[[]]) == baseline
        m.score_replay(model, replay, zero_indices=[[0, 3]])
        m.score_replay(model, replay, zero_indices=[list(range(model.visual_token_len))])
    finally:
        handle.remove()
    expected = original.clone()
    expected[0, replay['image_positions'][0, [0, 3]]] = 0
    assert torch.equal(seen[2], expected)
    assert torch.equal(replay['fused'], original)
    assert torch.count_nonzero(seen[3][0, replay['image_positions'][0]]) == 0


def test_baseline_reduced_maps_match_native_eager_with_self_and_special_mask():
    m = core()
    model, batch, _, _ = fixture(questions=('What?', 'What is visible? A road. A car.'))
    tokens = content_ids(model)
    generated = [tokens[:1] + [model.processor.tokenizer.bos_token_id] + tokens[1:], []]
    targets = [[[0, 1], [len(generated[0])-1]], []]
    baseline = m.collect_baseline(model, batch, generated, target_queries=targets,
                                  ranking_layers=(0, 1), layer_groups=((0,), (0, 1)))
    replay = baseline['replay']
    with torch.inference_mode():
        output = replay['text'](inputs_embeds=replay['fused'], attention_mask=replay['attention_mask'],
                                position_ids=replay['position_ids'], use_cache=False, output_attentions=True)
    expected_layers, expected_targets = [], []
    for attn in output.attentions:
        queries = replay['query_positions'][0, replay['query_valid'][0]]
        expected_layers.append(attn[0, :, queries].mean((0, 1))[replay['image_positions'][0]])
        expected_targets.append(torch.stack([
            attn[0, :, replay['query_positions'][0, [0]]].mean((0, 1))[replay['image_positions'][0]],
            attn[0, :, replay['query_positions'][0, [-1]]].mean((0, 1))[replay['image_positions'][0]],
        ]))
    torch.testing.assert_close(baseline['ranking'][0], torch.stack(expected_layers), atol=1e-6, rtol=1e-4)
    expected_targets = torch.stack(expected_targets, 1)
    expected_targets[:, 1] = expected_targets.mean(1)
    torch.testing.assert_close(baseline['target_maps'][0], expected_targets, atol=1e-6, rtol=1e-4)
    assert baseline['ranking'].dtype == torch.float32 and baseline['ranking'].device.type == 'cpu'
    assert torch.count_nonzero(baseline['ranking'][1]) == 0
    assert baseline['target_maps'][1].shape == (0, 2, model.visual_token_len)
    assert baseline['scores'][1] == dict(mean_logprob=None, token_logprobs=[], token_count=0)
    assert baseline['scores'] == m.score_replay(model, replay)
    assert all(not layer.self_attn._forward_pre_hooks for layer in replay['text'].layers)


def test_selection_random_and_probability_drop():
    m = core()
    attention = torch.tensor([1., 2., 2., 0., 0.])
    assert m.select_indices(attention, 3) == [1, 2, 0]
    assert m.select_indices(attention, 3, largest=False) == [3, 4, 0]
    random = m.random_indices('sample-a', k=4, visual_tokens=12)
    assert random == m.random_indices('sample-a', k=4, visual_tokens=12)
    assert random != m.random_indices('sample-b', k=4, visual_tokens=12)
    assert len(random) == 3 and all(len(set(indices)) == 4 for indices in random)
    assert m.relative_drop(-3, -2) == pytest.approx(1 - math.exp(1))
    assert m.relative_drop(-3, -4) == pytest.approx(1 - math.exp(-1))


def test_causal_runs_exact_condition_count_reuses_random_and_scores(monkeypatch):
    m = core()
    model, batch, _, _ = fixture()
    ids = [content_ids(model)]
    baseline = m.collect_baseline(model, batch, ids, ranking_layers=(0, 1), layer_groups=((0, 1),))
    calls = []
    real_score = m.score_replay
    def scored(*args, **kwargs):
        calls.append(kwargs['zero_indices'])
        return real_score(*args, **kwargs)
    monkeypatch.setattr(m, 'score_replay', scored)
    result = m.measure_causal(model, batch, ids, baseline, sample_ids=['one'], k=3, ranking_layers=(0, 1))[0]
    assert len(calls) == 2*2*2 + 3*2
    assert result['baseline_score'] == baseline['scores'][0]['mean_logprob']
    assert [entry['layer'] for entry in result['layers']] == [0, 1]
    for layer in result['layers']:
        for method in ('top', 'bottom'):
            condition = layer[method]
            assert len(condition['indices']) == 3
            assert condition['comp'] == pytest.approx(m.relative_drop(result['baseline_score'], condition['remove_score']))
    assert result['random']['comp'] == pytest.approx(sum(r['comp'] for r in result['random']['repeats'])/3)
    assert [r['indices'] for r in result['random']['repeats']] == m.random_indices('one', k=3, visual_tokens=24)


def test_special_only_and_hook_cleanup_on_failure(monkeypatch):
    m = core()
    model, batch, _, _ = fixture()
    special = [[model.processor.tokenizer.bos_token_id]]
    baseline = m.collect_baseline(model, batch, special, ranking_layers=(0,), layer_groups=((0, 1),))
    assert baseline['scores'][0]['mean_logprob'] is None
    assert torch.count_nonzero(baseline['ranking']) == 0
    def forbidden(*args, **kwargs):
        raise RuntimeError('injected failure')
    monkeypatch.setattr(model.native.model.text_model.layers[0].self_attn, 'forward', forbidden)
    with pytest.raises(RuntimeError, match='injected failure'):
        m.collect_baseline(model, batch, [content_ids(model)], ranking_layers=(0,), layer_groups=((0, 1),))
    assert all(not layer.self_attn._forward_pre_hooks for layer in model.native.model.text_model.layers)


def test_chunked_attention_padding_and_single_sample_parity():
    m = core()
    model, batch, items, collator = fixture(questions=('What?', 'What is visible? A road. A car.'))
    tokens = content_ids(model)
    generated = [tokens, tokens[:1]]
    targets = [[[0, len(tokens)-1]], [[0]]]
    options = dict(ranking_layers=(0, 1), layer_groups=((0,), (0, 1)))
    actual = m.collect_baseline(model, batch, generated, target_queries=targets, query_chunk_size=1, **options)
    unchunked = m.collect_baseline(model, batch, generated, target_queries=targets, query_chunk_size=100, **options)
    torch.testing.assert_close(actual['ranking'], unchunked['ranking'], atol=1e-6, rtol=1e-4)
    for row, item in enumerate(items):
        single = m.collect_baseline(model, collator([item]), [generated[row]], target_queries=[targets[row]], **options)
        torch.testing.assert_close(actual['ranking'][row], single['ranking'][0], atol=1e-6, rtol=1e-4)
        torch.testing.assert_close(actual['target_maps'][row], single['target_maps'][0], atol=1e-6, rtol=1e-4)
    batch['input_ids'] = torch.nn.functional.pad(batch['input_ids'], (0, 3), value=model.processor.tokenizer.unk_token_id)
    batch['attention_mask'] = torch.nn.functional.pad(batch['attention_mask'], (0, 3), value=0)
    right_padded = m.collect_baseline(model, batch, generated, target_queries=targets, **options)
    torch.testing.assert_close(actual['ranking'], right_padded['ranking'], atol=1e-6, rtol=1e-4)
    assert actual['scores'] == right_padded['scores']


def test_generated_attention_matches_native_cached_decode_including_last_token():
    m = core()
    model, batch, _, _ = fixture()
    tokens = content_ids(model)
    generated = tokens[:1] + [model.processor.tokenizer.bos_token_id] + tokens[1:]
    baseline = m.collect_baseline(model, batch, [generated], ranking_layers=(0, 1), layer_groups=((0, 1),))
    replay = baseline['replay']
    first = replay['query_positions'][0, 0].item()
    text = replay['text']
    layers = [[] for _ in text.layers]
    with torch.inference_mode():
        prefix = text(inputs_embeds=replay['fused'][:, :first], attention_mask=replay['attention_mask'][:, :first],
                      position_ids=replay['position_ids'][:, :first], use_cache=True)
        cache = prefix.past_key_values
        for index, position in enumerate(replay['query_positions'][0].tolist()):
            output = text(inputs_embeds=replay['fused'][:, position:position+1],
                          attention_mask=replay['attention_mask'][:, :position+1],
                          position_ids=replay['position_ids'][:, position:position+1],
                          past_key_values=cache, use_cache=True, output_attentions=True)
            cache = output.past_key_values
            if replay['query_valid'][0, index]:
                for layer, attention in zip(layers, output.attentions):
                    layer.append(attention[0, :, 0].mean(0)[replay['image_positions'][0]])
    expected = torch.stack([torch.stack(layer).mean(0) for layer in layers])
    torch.testing.assert_close(baseline['ranking'][0], expected, atol=1e-6, rtol=1e-4)


def test_zero_ablation_scores_match_native_oracle_and_nonfinite_stops(monkeypatch):
    m = core()
    model, batch, _, _ = fixture()
    replay = m.prepare_replay(model, batch, [content_ids(model)])
    changed = replay['fused'].clone()
    changed[0, replay['image_positions'][0, [0, 3]]] = 0
    with torch.inference_mode():
        output = model.native(input_ids=replay['input_ids'], inputs_embeds=changed,
                              attention_mask=replay['attention_mask'], position_ids=replay['position_ids'], use_cache=False)
        positions = replay['query_positions'][0, replay['query_valid'][0]]
        expected = output.logits[0, positions-1].log_softmax(-1).gather(1, replay['input_ids'][0, positions, None])[:, 0]
    actual = m.score_replay(model, replay, zero_indices=[[0, 3]])[0]
    assert actual['token_logprobs'] == pytest.approx(expected.tolist(), abs=1e-6)
    monkeypatch.setattr(model.native.lm_head, 'forward', lambda hidden: torch.full((*hidden.shape[:-1], model.native.config.text_config.vocab_size), float('nan')))
    with pytest.raises(ValueError, match='nonfinite'):
        m.score_replay(model, replay)
