import copy

import pytest
import torch

from smolvlm_inference.decoder import PrunedGreedyDecoder, question_attention_scores
from smolvlm_inference.pruning_text import PruningBatchCollator
from test_smolvlm_model import tiny_components
from smolvlm_inference.model import CachedSmolVLM


def fixture(layers=4):
    torch.manual_seed(27)
    native, processor = tiny_components()
    # Repeat actual native layers with distinct cache indices.
    native.model.text_model.layers = torch.nn.ModuleList([
        copy.deepcopy(native.model.text_model.layers[i % 2]) for i in range(layers)])
    native.config.text_config.num_hidden_layers = layers
    for i, layer in enumerate(native.model.text_model.layers):
        layer.self_attn.layer_idx = i
    native.config.text_config.max_position_embeddings = 2048
    model = CachedSmolVLM(native, processor).eval()
    items = [dict(question=q, answer='Unused.', sample_id=str(i), eval_id=i,
                  visual_features=torch.randn(6, 16, 8))
             for i, q in enumerate(('What is visible?', 'What changed? A person.'))]
    batch = PruningBatchCollator(processor, image_seq_len=4, max_length=2048)(items)
    return model, batch


@pytest.mark.parametrize('index,layers', [(2, 4), (15, 17)])
def test_identity_matches_native_greedy(index, layers):
    model, batch = fixture(layers)
    engine = PrunedGreedyDecoder(model, index, stage1_keep=24, stage2_keep=24)
    actual = engine.generate(batch, max_new_tokens=4, debug=True)
    expected = model.generate_from_features(
        **{k: batch[k] for k in ('input_ids', 'attention_mask', 'visual_features')},
        max_new_tokens=4, num_beams=1, do_sample=False)
    assert torch.equal(actual['generated_ids'], expected.cpu())
    assert actual['debug']['stage2_layer_index'] == index


@pytest.mark.parametrize('index,layers', [(2, 4), (15, 17)])
def test_pruned_cache_matches_recomputed_history_and_keeps_weights(index, layers):
    model, batch = fixture(layers)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    engine = PrunedGreedyDecoder(model, index, stage1_keep=18, stage2_keep=12)
    result = engine.generate(batch, max_new_tokens=4, debug=True)
    reference = engine.generate_reference(batch, max_new_tokens=4)
    assert torch.equal(result['generated_ids'], reference['generated_ids'])
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])
    assert result['debug']['visual_counts_by_layer'] == [18] * (index + 1) + [12] * (layers - index - 1)
    assert result['debug']['selection_calls'] == {'stage1': 1, 'stage2': 1}


def test_question_score_matches_native_eager_attention():
    model, batch = fixture()
    text = model.native.model.text_model
    x = text.embed_tokens(batch['input_ids'])
    positions = batch['source_position_ids']
    valid = batch['attention_mask'].bool()
    size = x.shape[1]
    causal = torch.arange(size)[None, :] <= torch.arange(size)[:, None]
    mask = torch.zeros(x.shape[0], 1, size, size).masked_fill(
        ~(causal[None, None] & valid[:, None, None]), torch.finfo(x.dtype).min)
    layer = text.layers[0]
    q, k = [], []
    hooks = [layer.self_attn.q_proj.register_forward_hook(lambda m, a, y: q.append(y)),
             layer.self_attn.k_proj.register_forward_hook(lambda m, a, y: k.append(y))]
    with torch.inference_mode():
        output = layer(x, attention_mask=mask, position_ids=positions,
                       position_embeddings=text.rotary_emb(x, positions), output_attentions=True)
        actual = question_attention_scores(q[0], k[0], layer.self_attn,
            text.rotary_emb(x, positions), mask, batch['question_mask'], batch['image_positions'])
    for h in hooks:
        h.remove()
    attn = output[1]
    mean = (attn * batch['question_mask'][:, None, :, None]).sum((1, 2))
    mean /= (batch['question_mask'].sum(1) * attn.shape[1])[:, None]
    expected = mean.gather(1, batch['image_positions'])
    torch.testing.assert_close(actual, expected)


def test_batch64_and_forced_stress_are_separate_from_eos_inference():
    model, batch = fixture()
    # A real zero LM head makes the native argmax deterministically token zero.
    model.native.lm_head.weight.zero_()
    model.native.generation_config.eos_token_id = 0
    batch = {key: value.repeat((32,) + (1,) * (value.ndim - 1))
             if isinstance(value, torch.Tensor) else value * 32
             for key, value in batch.items()}
    engine = PrunedGreedyDecoder(model, 2, stage1_keep=18, stage2_keep=12)
    normal = engine.generate(batch, max_new_tokens=4, debug=True)
    stress = engine.generate(batch, max_new_tokens=4, ignore_eos=True, debug=True)
    assert normal['generated_ids'].shape == (64, 1)
    assert normal['lengths'] == [1] * 64
    assert stress['generated_ids'].shape == (64, 4)
    assert stress['lengths'] == [4] * 64
    assert stress['debug']['selection_calls'] == {'stage1': 1, 'stage2': 1}


def test_context_overflow_and_nonfinite_logits_fail():
    model, batch = fixture()
    engine = PrunedGreedyDecoder(model, 2, stage1_keep=18, stage2_keep=12)
    with pytest.raises(ValueError, match='context'):
        engine.generate(batch, max_new_tokens=2048)
    model.native.lm_head.weight.fill_(float('nan'))
    with pytest.raises(FloatingPointError, match='logits'):
        engine.generate(batch, max_new_tokens=2)


def test_mixed_finished_and_active_samples_keep_independent_cache_masks():
    model, batch = fixture()
    engine = PrunedGreedyDecoder(model, 2, stage1_keep=18, stage2_keep=12)
    hidden = []
    handle = model.native.lm_head.register_forward_pre_hook(lambda m, args: hidden.append(args[0].clone()))
    engine.generate(batch, max_new_tokens=1)
    handle.remove()
    # Set real head weights to linearly separate the two actual prefill states.
    h0, h1 = hidden[0][:, 0]
    w = h0 / h0.norm() - h1 / h1.norm()
    scores = torch.stack((h0 @ w, h1 @ w))
    assert scores[0] > 0 and scores[1] < 0
    model.native.lm_head.weight.zero_()
    model.native.lm_head.weight[0] = w
    model.native.generation_config.eos_token_id = 0
    got = engine.generate(batch, max_new_tokens=4, debug=True)
    reference = engine.generate_reference(batch, max_new_tokens=4)
    assert got['lengths'][0] == 1 and got['lengths'][1] > 1
    assert torch.equal(got['generated_ids'], reference['generated_ids'])
    assert got['generated_ids'][0, 1:].eq(model.native.generation_config.pad_token_id).all()
