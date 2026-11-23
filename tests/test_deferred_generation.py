import pytest
import torch
from test_smolvlm_pruned_decoder import fixture
from smolvlm_topk_benchmark.generation import fuse_batch, compact_prompt, SharedCausalLM, greedy_config, generate_compacted


def test_full_512_token_budget_preserves_layer_local_cache_lengths():
    from smolvlm_topk_deferred.generation import DeferredGreedyDecoder
    model, batch = fixture()
    model.native.model.text_model.config.max_position_embeddings = 2048
    fused = fuse_batch(model, batch)
    keep = [[0, 5, 23], [1, 7, 12]]
    compact = compact_prompt(fused, keep)
    config = greedy_config(model, 512)
    config.eos_token_id = None
    decoder = DeferredGreedyDecoder(model)
    result = decoder.generate(fused, keep, config, debug=True)
    assert [r['generated_token_count'] for r in result] == [512, 512]
    assert all(r['stop_reason'] == 'max_new_tokens' for r in result)
    full, short = fused['input_ids'].shape[1], compact['input_ids'].shape[1]
    assert decoder.last_debug['final_cache_lengths'] == [full+511]*3 + [short+511]


def test_deferred_all_keep_and_reference():
    from smolvlm_topk_deferred.generation import DeferredGreedyDecoder
    model, batch = fixture()
    fused = fuse_batch(model, batch)
    config = greedy_config(model, 4)
    decoder = DeferredGreedyDecoder(model)
    actual = decoder.generate(fused, [None, None], config, debug=True)
    expected = generate_compacted(SharedCausalLM(model.native), compact_prompt(fused, [None, None]), model.processor.tokenizer, config)
    assert actual == expected
    assert decoder.last_debug['visual_counts_by_layer'] == [[24, 24]] * 4
    assert actual == decoder.generate_reference(fused, [None, None], config)


def test_deferred_fixed_selection_reference_and_layer_lengths():
    from smolvlm_topk_deferred.generation import DeferredGreedyDecoder
    model, batch = fixture()
    fused = fuse_batch(model, batch)
    config = greedy_config(model, 5)
    selection = [[0, 5, 10, 23], [0, 1, 6]]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    decoder = DeferredGreedyDecoder(model)
    actual = decoder.generate(fused, selection, config, debug=True)
    info = decoder.last_debug
    assert info['visual_counts_by_layer'] == [[24,24]] * 3 + [[4,3]]
    assert info['prefill_cache_lengths'][:3] == [fused['input_ids'].shape[1]] * 3
    assert info['prefill_cache_lengths'][3] < info['prefill_cache_lengths'][2]
    assert actual == decoder.generate_reference(fused, selection, config)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(v, before[k]) for k,v in model.state_dict().items())


@pytest.mark.parametrize('field,value', [('num_beams', 2), ('repetition_penalty', 1.1), ('do_sample', True), ('min_new_tokens', 2)])
def test_deferred_rejects_processors(field, value):
    from smolvlm_topk_deferred.generation import DeferredGreedyDecoder
    model, batch = fixture()
    config = greedy_config(model, 3)
    setattr(config, field, value)
    with pytest.raises(ValueError):
        DeferredGreedyDecoder(model).generate(fuse_batch(model,batch), [None,None], config)


def test_gather_is_after_complete_block_and_original_positions_preserved():
    from smolvlm_topk_deferred.generation import DeferredGreedyDecoder
    model, batch = fixture(30)
    fused = fuse_batch(model, batch)
    selections = [[0,5,23],[1,7,12]]
    captured = {}
    def after(module, args, output):
        captured['block2'] = output[0].detach().clone()
    def before(module, args, kwargs):
        if 'block3' not in captured:
            captured['block3'] = args[0].detach().clone()
            captured['positions'] = kwargs['position_ids'].clone()
    hooks = [model.native.model.text_model.layers[2].register_forward_hook(after),
             model.native.model.text_model.layers[3].register_forward_pre_hook(before, with_kwargs=True)]
    try:
        DeferredGreedyDecoder(model).generate(fused,selections,greedy_config(model,1))
    finally:
        for hook in hooks:
            hook.remove()
    expected = compact_prompt(dict(fused, inputs_embeds=captured['block2']), selections)
    assert torch.equal(captured['block3'], expected['inputs_embeds'])
    assert torch.equal(captured['positions'], expected['prompt_positions'])


def test_eos_stops_at_first_token_and_returns_old_schema():
    from smolvlm_topk_deferred.generation import DeferredGreedyDecoder
    model, batch = fixture()
    with torch.no_grad():
        model.native.lm_head.weight.zero_()
    model.native.generation_config.eos_token_id = [0]
    result = DeferredGreedyDecoder(model).generate(fuse_batch(model,batch),[None,None],greedy_config(model,5))
    assert all(r['generated_ids'] == [0] and r['stop_reason'] == 'eos' and r['generated_token_count'] == 1 for r in result)


def test_mixed_padding_matches_individual_rows_and_no_pruning_helpers(monkeypatch):
    from smolvlm_topk_deferred.generation import DeferredGreedyDecoder
    import smolvlm_inference.pruning as pruning
    def forbidden(*args, **kwargs):
        raise AssertionError('MVPruner must not run')
    monkeypatch.setattr(pruning,'stage1_indices',forbidden)
    monkeypatch.setattr(pruning,'stage2_indices',forbidden)
    model,batch = fixture()
    fused = fuse_batch(model,batch)
    selections = [[0,5,23],[1,7,12]]
    config = greedy_config(model,4)
    decoder = DeferredGreedyDecoder(model)
    expected = decoder.generate(fused,selections,config)
    for row in range(2):
        single = {k: v[row:row+1] if isinstance(v,torch.Tensor) else v for k,v in fused.items()}
        assert decoder.generate(single,[selections[row]],config)[0] == expected[row]
