import copy
import numpy as np
import pytest
import torch
from test_attention_collector import fixture


def components():
    model, _, items, collator = fixture(questions=('What?', 'What is visible? A road. A car.'))
    layers = model.native.model.text_model.layers
    model.native.model.text_model.layers = torch.nn.ModuleList([copy.deepcopy(layers[i % 2]) for i in range(4)])
    model.native.config.text_config.num_hidden_layers = 4
    for index, layer in enumerate(model.native.model.text_model.layers):
        layer.self_attn.layer_idx = index
    rows = [dict(item, row_index=index, sample_id=str(index),
                 input_ids=collator([item])['input_ids'][0].tolist())
            for index, item in enumerate(items)]
    return model, rows, collator, np.array([[0, 5, 10, 23], [1, 6, 11, 22]])


@pytest.mark.parametrize('name', ['score22_prune_input', 'score22_prune_after2'])
def test_numerical_preflight_real_model_and_unchanged_state(name):
    from smolvlm_topk_deferred.preflight import numerical_preflight
    model, rows, collator, selections = components()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    sizes = []
    def build_batch(selected):
        sizes.append(len(selected))
        return collator(selected)
    result = numerical_preflight(model, rows, build_batch, selections, name)
    assert result['passed']
    assert result['config_name'] == name
    assert result['probe_row_indices'] == [0, 1]
    assert result['max_new_tokens'] == 4
    assert 2 in sizes
    assert all(torch.equal(value, state[key]) for key, value in model.state_dict().items())
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('name', ['score22_prune_input', 'score22_prune_after2'])
def test_stress_runs_longest_fixed_batch_and_all_steps(name):
    from smolvlm_topk_deferred.preflight import stress
    model, rows, collator, selections = components()
    model.native.lm_head.weight.zero_()
    model.native.generation_config.eos_token_id = 0
    generation_before = model.native.generation_config.to_dict()
    seen = []
    def build_batch(selected):
        seen.append([row['row_index'] for row in selected])
        return collator(selected)
    result = stress(model, rows, build_batch, selections, name,
                    batch_size=2, max_new_tokens=3)
    assert result['passed']
    assert result['batch_size'] == 2
    assert result['max_new_tokens'] == 3
    assert result['generated_token_counts'] == [3, 3]
    assert seen == [[1, 0]]
    assert model.native.generation_config.to_dict() == generation_before
    assert result['eos_policy'] == 'disabled_for_stress_only'
    assert result['cuda_peak_allocated_bytes'] is None
    assert result['cuda_peak_reserved_bytes'] is None


def test_preflight_rejects_empty_or_unknown_config():
    from smolvlm_topk_deferred.preflight import numerical_preflight, stress
    model, rows, collator, selections = components()
    with pytest.raises(ValueError, match='row|sample'):
        numerical_preflight(model, [], collator, selections, 'score22_prune_after2')
    with pytest.raises(ValueError, match='config'):
        stress(model, rows, collator, selections, 'unknown', batch_size=2)
    with pytest.raises(ValueError, match='batch|sample'):
        stress(model, rows, collator, selections, 'score22_prune_after2', batch_size=3)

def test_deferred_decode_masks_and_cache_slots_are_layer_local():
    from smolvlm_topk_deferred.generation import DeferredGreedyDecoder
    from smolvlm_topk_benchmark.generation import fuse_batch, compact_prompt, greedy_config
    model, rows, collator, selections = components()
    fused = fuse_batch(model, collator(rows))
    chosen = selections.tolist()
    compact = compact_prompt(fused, chosen)
    config = greedy_config(model, 3)
    config.eos_token_id = None
    calls = {0: [], 3: []}
    handles = []
    for index in calls:
        def capture(module, args, kwargs, index=index):
            calls[index].append({key: kwargs[key].detach().clone()
                                 for key in ('attention_mask', 'position_ids', 'cache_position')})
        handles.append(model.native.model.text_model.layers[index].register_forward_pre_hook(capture, with_kwargs=True))
    try:
        decoder = DeferredGreedyDecoder(model)
        decoder.generate(fused, chosen, config, debug=True)
    finally:
        for handle in handles:
            handle.remove()
    for index, original in [(0, fused['attention_mask']), (3, compact['attention_mask'])]:
        width = original.shape[1]
        assert len(calls[index]) == 3
        for step in (1, 2):
            observed = calls[index][step]
            valid = torch.cat((original.bool(), torch.ones(original.shape[0], step, dtype=torch.bool)), 1)
            assert torch.equal(observed['attention_mask'][:, 0, 0].eq(0), valid)
            assert observed['cache_position'].tolist() == [width + step - 1]
            assert torch.equal(observed['position_ids'][:, 0], fused['original_lengths'] + step - 1)
    info = decoder.last_debug
    assert info['final_cache_lengths'] == [fused['attention_mask'].shape[1] + 2] * 3 + [compact['attention_mask'].shape[1] + 2]
