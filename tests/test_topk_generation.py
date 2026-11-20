import torch
import pytest
from test_attention_collector import fixture


def test_all_keep_matches_native():
    from smolvlm_topk_benchmark.generation import fuse_batch, compact_prompt, SharedCausalLM, greedy_config, generate_compacted
    model, batch, _, _ = fixture()
    fused = fuse_batch(model, batch)
    config = greedy_config(model, 4)
    expected = model.native.generate(input_ids=fused['input_ids'], inputs_embeds=fused['inputs_embeds'], attention_mask=fused['attention_mask'], generation_config=config)[:, fused['input_ids'].shape[1]:]
    adapter = SharedCausalLM(model.native)
    assert adapter.model is model.native.model.text_model
    assert adapter.lm_head is model.native.lm_head
    result = generate_compacted(adapter, compact_prompt(fused, [None]), model.processor.tokenizer, config)
    assert result[0]['generated_ids'] == expected[0].tolist()


def test_compaction_preserves_nonvisual_and_positions():
    from smolvlm_topk_benchmark.generation import fuse_batch, compact_prompt
    model, batch, _, _ = fixture()
    fused = fuse_batch(model, batch)
    compact = compact_prompt(fused, [[0, 5, 10, 23]])
    assert compact['input_ids'].shape[1] == fused['input_ids'].shape[1] - 20
    visual = fused['image_positions'][0]
    keep = torch.ones(fused['input_ids'].shape[1], dtype=torch.bool)
    keep[visual] = False
    keep[visual[[0, 5, 10, 23]]] = True
    assert torch.equal(compact['input_ids'][0], fused['input_ids'][0, keep])
    assert torch.equal(compact['prompt_positions'][0], fused['position_ids'][0, keep])
    with pytest.raises(ValueError):
        compact_prompt(fused, [[0, 0]])


def test_decode_eos_padding():
    from smolvlm_topk_benchmark.generation import decode_sequences
    class Tokenizer:
        def decode(self, ids, skip_special_tokens=True):
            return ' text ' if ids else ''
    result = decode_sequences(Tokenizer(), torch.tensor([[7, 2, 0, 0], [7, 8, 9, 10]]), [2], 4)
    assert result[0]['generated_ids'] == [7, 2]
    assert result[0]['stop_reason'] == 'eos'
    assert result[0]['prediction'] == 'text'
    assert result[1]['generated_token_count'] == 4


def test_pruned_cached_matches_full_prefix():
    from smolvlm_topk_benchmark.generation import fuse_batch, compact_prompt, SharedCausalLM, greedy_config, generate_compacted
    model, batch, _, _ = fixture()
    adapter = SharedCausalLM(model.native)
    compact = compact_prompt(fuse_batch(model, batch), [[0, 5, 10, 23]])
    config = greedy_config(model, 4)
    actual = generate_compacted(adapter, compact, model.processor.tokenizer, config)[0]['generated_ids']
    embeddings = compact['inputs_embeds']
    positions = compact['prompt_positions']
    expected = []
    with torch.inference_mode():
        for step in range(4):
            logits = adapter(inputs_embeds=embeddings, position_ids=positions, use_cache=False).logits
            token = logits[:, -1].argmax(-1)
            expected.append(token.item())
            if token.item() == config.eos_token_id:
                break
            embeddings = torch.cat([embeddings, adapter.get_input_embeddings()(token[:, None])], dim=1)
            positions = torch.cat([positions, compact['original_lengths'][:, None] + step], dim=1)
    assert actual == expected


def test_mixed_left_padding_matches_single_generation():
    from smolvlm_topk_benchmark.generation import fuse_batch, compact_prompt, SharedCausalLM, greedy_config, generate_compacted
    model, batch, items, collator = fixture(questions=('What?', 'What is visible? A road. A car.'))
    adapter = SharedCausalLM(model.native)
    config = greedy_config(model, 4)
    selections = [[0, 5, 10, 23], [0, 1]]
    actual = generate_compacted(adapter, compact_prompt(fuse_batch(model, batch), selections), model.processor.tokenizer, config)
    for index, item in enumerate(items):
        single = generate_compacted(adapter, compact_prompt(fuse_batch(model, collator([item])), [selections[index]]), model.processor.tokenizer, config)
        assert actual[index]['generated_ids'] == single[0]['generated_ids']


def test_32_of_384_physical_tokens():
    from smolvlm_topk_benchmark.generation import compact_prompt
    ids = torch.arange(400)[None]
    fused = dict(input_ids=ids, inputs_embeds=torch.randn(1, 400, 8),
                 attention_mask=torch.ones_like(ids), position_ids=ids.clone(),
                 original_lengths=torch.tensor([400]), image_positions=torch.arange(8, 392)[None])
    compact = compact_prompt(fused, [list(range(0, 384, 12))])
    assert compact['input_ids'].shape == (1, 48)
    assert compact['inputs_embeds'].shape == (1, 48, 8)
    assert compact['original_lengths'].item() == 400
    assert compact['prompt_positions'][0, -1].item() == 399


def test_original_context_limit_applies_even_when_physical_prompt_is_short():
    from types import SimpleNamespace
    from smolvlm_topk_benchmark.generation import generate_compacted
    adapter=SimpleNamespace(training=False,parameters=lambda:[],config=SimpleNamespace(max_position_embeddings=512))
    compact={'original_lengths':torch.tensor([510]),'input_ids':torch.ones(1,10,dtype=torch.long)}
    with pytest.raises(ValueError,match='original prompt'):
        generate_compacted(adapter,compact,None,SimpleNamespace(max_new_tokens=4))
