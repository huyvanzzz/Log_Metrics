import pytest
import torch

from test_attention_collector import fixture


def test_generated_native_parity_and_padding():
    from smolvlm_attention.generated_collector import collect_generated_attention, reference_generated_attention
    model, batch, items, collator = fixture(questions=('What?', 'What is visible? A road. A car.'))
    ids = [model.processor.tokenizer.encode('A road.', add_special_tokens=False),
           model.processor.tokenizer.encode('A car. A road.', add_special_tokens=False)]
    actual = collect_generated_attention(model, batch, ids)
    expected = reference_generated_attention(model, batch, ids)
    assert actual['visual'].shape == (2, 2, 24)
    assert actual['groups'].shape == (2, 2, 3)
    assert actual['token_counts'] == list(map(len, ids))
    for key in ('visual', 'groups'):
        torch.testing.assert_close(actual[key], expected[key], atol=1e-6, rtol=1e-4)
        for i, item in enumerate(items):
            single = collect_generated_attention(model, collator([item]), [ids[i]])
            torch.testing.assert_close(actual[key][i], single[key][0], atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(actual['groups'].sum(-1), torch.ones(2, 2))
    torch.testing.assert_close(actual['visual'].sum(-1), actual['groups'][..., 0])
    assert (actual['groups'][..., 2] > 0).all()


def test_generated_empty_and_image_id_rejected():
    from smolvlm_attention.generated_collector import collect_generated_attention
    model, batch, _, _ = fixture(questions=('What?', 'What is visible?'))
    result = collect_generated_attention(model, batch, [[], [12]])
    assert result['token_counts'] == [0, 1]
    assert not result['visual'][0].any() and not result['groups'][0].any()
    assert collect_generated_attention(model, batch, [[], []])['groups'].sum() == 0
    with pytest.raises(ValueError, match='image'):
        collect_generated_attention(model, batch, [[model.native.config.image_token_id], []])
    with pytest.raises(ValueError):
        collect_generated_attention(model, batch, [[]])


def test_generated_no_future_attention_and_cleanup(monkeypatch):
    from smolvlm_attention.generated_collector import collect_generated_attention
    model, batch, _, _ = fixture()
    first = collect_generated_attention(model, batch, [[12]])
    second = collect_generated_attention(model, batch, [[12, 13]])
    # Native eager parity in the first test checks all query positions; this
    # additionally proves the last generated token is included in the mean.
    assert not torch.equal(first['groups'], second['groups'])
    text = model.native.model.text_model
    def fail(*args, **kwargs):
        raise RuntimeError('injected failure')
    monkeypatch.setattr(text.layers[0].self_attn, 'forward', fail)
    with pytest.raises(RuntimeError, match='injected'):
        collect_generated_attention(model, batch, [[12]])
    assert all(not layer.self_attn._forward_pre_hooks for layer in text.layers)


def test_replay_matches_native_kv_decode_including_last_content_token():
    from smolvlm_attention.collector import _prepare
    from smolvlm_attention.generated_collector import collect_generated_attention
    model, batch, _, _ = fixture()
    generated = [12, 13, 14]
    actual = collect_generated_attention(model, batch, [generated])
    with torch.inference_mode():
        text, fused, mask, positions, _, images = _prepare(model, batch)
        output = text(inputs_embeds=fused, attention_mask=mask, position_ids=positions,
                      use_cache=True, return_dict=True)
        cache = output.past_key_values
        prompt_len = fused.shape[1]
        visuals, groups = [], []
        for i, token in enumerate(generated):
            output = text(input_ids=torch.tensor([[token]]),
                          attention_mask=torch.ones(1, prompt_len+i+1, dtype=torch.long),
                          position_ids=torch.tensor([[prompt_len+i]]),
                          past_key_values=cache, use_cache=True,
                          output_attentions=True, return_dict=True)
            cache = output.past_key_values
            visual_layers, group_layers = [], []
            for attn in output.attentions:
                prob = attn[0, :, 0].mean(0)
                visual = prob[images[0]]
                visual_layers.append(visual)
                group_layers.append(torch.stack([visual.sum(), prob[:prompt_len].sum()-visual.sum(),
                                                 prob[prompt_len:].sum()]))
            visuals.append(torch.stack(visual_layers))
            groups.append(torch.stack(group_layers))
    torch.testing.assert_close(actual['visual'][0], torch.stack(visuals).mean(0), atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(actual['groups'][0], torch.stack(groups).mean(0), atol=1e-6, rtol=1e-4)


def test_collector_native_path_fuses_once_and_never_runs_vision_or_head(monkeypatch):
    from smolvlm_attention.generated_collector import collect_generated_attention
    model, batch, _, _ = fixture()
    calls, inference = [], []
    handle = model.native.model.connector.register_forward_hook(lambda *args: calls.append('connector'))
    merger = model.native.model.inputs_merger
    monkeypatch.setattr(model.native.model, 'inputs_merger',
                        lambda **kw: (calls.append('merger'), merger(**kw))[1])
    def forbidden(*args, **kwargs):
        raise AssertionError('vision and lm_head must not run')
    monkeypatch.setattr(model.native.model.vision_model, 'forward', forbidden)
    monkeypatch.setattr(model.native.lm_head, 'forward', forbidden)
    guard = model.native.model.text_model.layers[0].self_attn.register_forward_pre_hook(
        lambda *args: inference.append(torch.is_inference_mode_enabled()))
    try:
        collect_generated_attention(model, batch, [[12, 13]])
    finally:
        handle.remove()
        guard.remove()
    assert calls == ['connector', 'merger']
    assert inference == [True]
