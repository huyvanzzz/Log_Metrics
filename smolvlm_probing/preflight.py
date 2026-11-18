"""Small native eager and cached-decode oracles for the loaded checkpoint.

This function produces a validation receipt only. Fallback text is used solely
when a generated answer cannot support a numerical check; no result is saved.
"""
import torch

from smolvlm_attention.collector import _prepare
from smolvlm_attention.generated_runtime import generation_config

from .config import LAYER_GROUPS, RANKING_LAYERS


def _answers(model, generated):
    tokenizer = model.processor.tokenizer
    native_eos = model.native.generation_config.eos_token_id
    eos = set(native_eos if isinstance(native_eos, (tuple, list)) else [native_eos])
    specials = set(tokenizer.all_special_ids)
    answers, fallback = [], []
    for index, raw in enumerate(generated):
        raw = raw.tolist()
        end = next((i for i, token in enumerate(raw) if token in eos), len(raw))
        answer = raw[:end]
        reason = ('invalid_multimodal_generation' if model.native.config.image_token_id in answer else
                  'empty' if not any(token not in specials for token in answer) else None)
        if reason:
            fallback.append(dict(sample_index=index, reason=reason))
            answer = tokenizer.encode('A car.', add_special_tokens=False)
            if (model.native.config.image_token_id in answer or not any(token not in specials for token in answer)):
                raise ValueError('Cannot construct nonempty regular fallback tokens for numeric preflight')
        answers.append(answer)
    return answers, fallback


def _single_prompt(batch):
    valid = batch['attention_mask'][0].bool()
    return dict(input_ids=batch['input_ids'][0, valid][None],
                attention_mask=batch['attention_mask'][0, valid][None],
                visual_features=batch['visual_features'][:1])


def _cached_oracle(model, batch, answer, baseline, ranking_layers):
    """Fresh prompt cache; score before each token, inspect attention after it."""
    text, fused, mask, positions, _, image_positions = _prepare(model, batch)
    output = text(inputs_embeds=fused, attention_mask=mask, position_ids=positions,
                  use_cache=True, output_attentions=False, return_dict=True)
    cache = output.past_key_values
    previous = output.last_hidden_state[:, -1]
    specials = set(model.processor.tokenizer.all_special_ids)
    attention = [[] for _ in ranking_layers]
    logprobs = []
    prompt_length = fused.shape[1]
    for index, token in enumerate(answer):
        if token not in specials:
            logprobs.append(model.native.lm_head(previous).float().log_softmax(-1)[0, token])
        output = text(input_ids=torch.tensor([[token]], device=fused.device),
                      attention_mask=mask.new_ones((1, prompt_length+index+1)),
                      position_ids=positions.new_tensor([[prompt_length+index]]),
                      past_key_values=cache, use_cache=True, output_attentions=True, return_dict=True)
        cache = output.past_key_values
        previous = output.last_hidden_state[:, -1]
        if token not in specials:
            for values, layer in zip(attention, ranking_layers):
                values.append(output.attentions[layer][0, :, 0].mean(0)[image_positions[0]].cpu())
    actual = torch.stack([torch.stack(values).mean(0) for values in attention])[None]
    torch.testing.assert_close(actual, baseline['ranking'], atol=2e-6, rtol=1e-4)
    expected = torch.tensor(baseline['scores'][0]['token_logprobs'], device=fused.device)
    torch.testing.assert_close(torch.stack(logprobs), expected, atol=2e-5, rtol=1e-4)


@torch.inference_mode()
def numerical_preflight(model, batch, *, ranking_layers=RANKING_LAYERS, layer_groups=LAYER_GROUPS):
    """Check original-ID replay against eager attention, LM logits, and KV decode.

    Defaults check production's five causal layers and six layer groups. Tiny
    fixtures may supply smaller layer tuples. Generated middle special IDs stay
    in every replay and cache, and only content IDs contribute to means.
    """
    from . import measurement

    generated = model.generate_from_features(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
        visual_features=batch['visual_features'], generation_config=generation_config(model.native, max_new_tokens=4))
    answers, fallback = _answers(model, generated.cpu())
    options = dict(ranking_layers=ranking_layers, layer_groups=layer_groups)
    baseline = measurement.collect_baseline(model, batch, answers, **options)
    replay = baseline['replay']
    output = replay['text'](inputs_embeds=replay['fused'], attention_mask=replay['attention_mask'],
        position_ids=replay['position_ids'], use_cache=False, output_attentions=True, return_dict=True)
    rankings = []
    weights = replay['query_valid'].float()
    weights /= weights.sum(-1, keepdim=True).clamp(min=1)
    for layer in ranking_layers:
        attention = output.attentions[layer]
        selected = attention.gather(2, replay['query_positions'][:, None, :, None].expand(
            -1, attention.shape[1], -1, attention.shape[-1])).mean(1)
        mean = (selected * weights[:, :, None]).sum(1)
        rankings.append(mean.gather(1, replay['image_positions']).cpu())
    torch.testing.assert_close(torch.stack(rankings, 1), baseline['ranking'], atol=2e-6, rtol=1e-4)
    del output, attention, selected, rankings
    no_ablation = measurement.score_replay(model, replay)
    for left, right in zip(baseline['scores'], no_ablation):
        torch.testing.assert_close(torch.tensor(left['token_logprobs']), torch.tensor(right['token_logprobs']),
                                   atol=2e-5, rtol=1e-4)
    for indices in (None, [[0, 1] for _ in answers]):
        actual = measurement.score_replay(model, replay, zero_indices=indices)
        fused = replay['fused'].clone()
        if indices is not None:
            for row, chosen in enumerate(indices):
                fused[row, replay['image_positions'][row, chosen]] = 0
        # No image_hidden_states: native fusion must not overwrite zero vectors.
        native = model.native(input_ids=replay['input_ids'], inputs_embeds=fused,
            attention_mask=replay['attention_mask'], position_ids=replay['position_ids'], use_cache=False, return_dict=True)
        for row in range(len(answers)):
            queries = replay['query_positions'][row, replay['query_valid'][row]]
            target = replay['input_ids'][row, queries]
            expected = native.logits[row, queries-1].float().log_softmax(-1).gather(1, target[:, None])[:, 0]
            torch.testing.assert_close(torch.tensor(actual[row]['token_logprobs'], device=expected.device),
                                       expected, atol=2e-5, rtol=1e-4)
        del native, fused
    single = _single_prompt(batch)
    single_baseline = measurement.collect_baseline(model, single, [answers[0]], **options)
    torch.testing.assert_close(single_baseline['ranking'][0], baseline['ranking'][0], atol=2e-6, rtol=1e-4)
    _cached_oracle(model, single, answers[0], single_baseline, ranking_layers)
    return dict(native_attention_parity=True, native_predecessor_scoring_parity=True,
                fused_zero_ablation_parity=True, cached_decode_attention_parity=True,
                cached_decode_scoring_parity=True, samples=len(answers),
                fallback_samples=fallback, fallback_policy='regular_A_car_tokens_for_numeric_check_only')
