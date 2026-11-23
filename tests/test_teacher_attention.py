import pytest
import torch

from test_smolvlm_pruned_decoder import fixture


def test_index22_matches_native_eager_and_preserves_rng_weights():
    from smolvlm_attention.teacher_targets.attention import collect_teacher_log_scores
    from smolvlm_attention.collector import reference_last_query_attention
    model, batch = fixture(24)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    actual = collect_teacher_log_scores(model, batch)
    native = reference_last_query_attention(model, batch)[:, 22]
    expected = (native / native.sum(-1, keepdim=True)).log()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
    assert actual.shape == (2, 24) and actual.dtype == torch.float32
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_target_only_qk_no_v_ffn_later_blocks_or_head(monkeypatch):
    from smolvlm_attention.teacher_targets.attention import collect_teacher_log_scores
    model, batch = fixture(24)
    layers = model.native.model.text_model.layers
    calls = []
    handles = [layer.register_forward_hook(lambda module, args, out, i=i: calls.append(i))
               for i, layer in enumerate(layers)]
    def forbidden(*args, **kwargs):
        raise AssertionError('forbidden downstream computation')
    for module in (layers[22], layers[22].self_attn.v_proj, layers[22].mlp,
                   layers[23], model.native.lm_head, model.native.model.vision_model):
        monkeypatch.setattr(module, 'forward', forbidden)
    try:
        actual = collect_teacher_log_scores(model, batch)
    finally:
        for handle in handles:
            handle.remove()
    assert calls == list(range(22))
    torch.testing.assert_close(actual.logsumexp(-1), torch.zeros(2), atol=1e-6, rtol=0)


def test_stable_log_reduction_matches_probability_reference_and_extremes():
    from smolvlm_attention.teacher_targets.attention import normalized_visual_log_probs
    logits = torch.randn(2, 9, 17, dtype=torch.float64)
    valid = torch.ones(2, 17, dtype=torch.bool)
    valid[0, :3] = False
    positions = torch.tensor([[3, 6, 8], [1, 5, 12]])
    probabilities = logits.masked_fill(~valid[:, None], -torch.inf).softmax(-1).mean(1)
    visual = probabilities.gather(1, positions)
    reference = (visual / visual.sum(-1, keepdim=True)).log()
    torch.testing.assert_close(normalized_visual_log_probs(logits, valid, positions), reference)
    logits *= 1e6
    actual = normalized_visual_log_probs(logits, valid, positions)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.logsumexp(-1), torch.zeros(2, dtype=torch.float64), atol=1e-8, rtol=0)


def test_padding_and_layer_boundaries():
    from smolvlm_attention.teacher_targets.attention import collect_teacher_log_scores
    model, batch = fixture(24)
    expected = collect_teacher_log_scores(model, batch)
    batch['input_ids'] = torch.nn.functional.pad(batch['input_ids'], (0, 4), value=0)
    batch['attention_mask'] = torch.nn.functional.pad(batch['attention_mask'], (0, 4))
    torch.testing.assert_close(collect_teacher_log_scores(model, batch), expected, atol=2e-6, rtol=1e-5)
    for index in (-1, 24, True):
        with pytest.raises(ValueError):
            collect_teacher_log_scores(model, batch, layer_index=index)


def test_full_384_visual_mapping_and_answers_are_not_inputs():
    from smolvlm_attention.teacher_targets.attention import collect_teacher_log_scores
    from smolvlm_attention.collector import reference_last_query_attention
    from smolvlm_inference.text import CachedBatchCollator
    model, _ = fixture(24)
    model.geometry['vision_tokens'] = 256
    model.geometry['visual_tokens_per_view'] = 64
    model.visual_token_len = 384
    collator = CachedBatchCollator(model.processor, image_seq_len=64, max_length=2048,
                                  max_new_tokens=0)
    features = torch.arange(6 * 256 * 8).reshape(6, 256, 8).float() / 1000
    item = dict(question='What is visible?\nA car?', answer='First answer.', visual_features=features)
    batch = collator([item])
    actual = collect_teacher_log_scores(model, batch)
    native = reference_last_query_attention(model, batch)[:, 22]
    torch.testing.assert_close(actual, (native / native.sum(-1, keepdim=True)).log(), atol=2e-6, rtol=1e-5)
    assert actual.shape == (1, 384)
    item['answer'] = 'Entirely different ground truth.'
    changed = collator([item])
    assert torch.equal(batch['input_ids'], changed['input_ids'])
    assert torch.equal(actual, collect_teacher_log_scores(model, changed))


def test_nonfinite_and_invalid_visual_keys_are_rejected():
    from smolvlm_attention.teacher_targets.attention import normalized_visual_log_probs
    logits = torch.zeros(1, 9, 4)
    positions = torch.tensor([[1, 2]])
    valid = torch.tensor([[1, 1, 0, 1]])
    with pytest.raises(ValueError, match='valid keys'):
        normalized_visual_log_probs(logits, valid, positions)
    logits[0, 0, 0] = torch.nan
    with pytest.raises(FloatingPointError):
        normalized_visual_log_probs(logits, torch.ones_like(valid), positions)
