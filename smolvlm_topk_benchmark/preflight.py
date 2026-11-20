"""Small mandatory numerical checks before running a full benchmark."""
import torch
from .generation import (SharedCausalLM, compact_prompt, decode_sequences,
                         fuse_batch, generate_compacted, greedy_config)


def _close(actual, expected, *, batched=False):
    # Different batch shapes can select different FP32 CUDA kernels. Keep the
    # native adapter check strict; token generation parity stays exact below.
    torch.testing.assert_close(actual, expected, atol=1e-5 if batched else 1e-6, rtol=1e-4)


def _last_logits(adapter, compact):
    return adapter(inputs_embeds=compact['inputs_embeds'],
        attention_mask=compact['attention_mask'], position_ids=compact['prompt_positions'],
        use_cache=False).logits[:, -1]


def _uncached(adapter, compact, tokenizer, config):
    embeddings = compact['inputs_embeds']
    mask = compact['attention_mask']
    positions = compact['prompt_positions']
    continuation = []
    for step in range(config.max_new_tokens):
        token = adapter(inputs_embeds=embeddings, attention_mask=mask,
            position_ids=positions, use_cache=False).logits[:, -1].argmax(-1)
        continuation.append(token)
        embeddings = torch.cat([embeddings, adapter.get_input_embeddings()(token[:, None])], 1)
        mask = torch.cat([mask, mask.new_ones((mask.shape[0], 1))], 1)
        positions = torch.cat([positions, compact['original_lengths'][:, None] + step], 1)
    return decode_sequences(tokenizer, torch.stack(continuation, 1), config.eos_token_id, config.max_new_tokens)


@torch.inference_mode()
def numerical_preflight(model, rows, build_batch, selection_rows, *, progress=None):
    if not rows:
        raise ValueError('preflight needs at least one row')
    def length(index):
        row = rows[index]
        if 'input_ids' in row:
            return len(row['input_ids'])
        if 'prompt_length' in row:
            return int(row['prompt_length'])
        raise ValueError('preflight rows need input_ids or prompt_length')
    probes = list(dict.fromkeys([min(range(len(rows)), key=length), max(range(len(rows)), key=length)]))
    row_indices = [int(rows[index].get('row_index', index)) for index in probes]
    active = next((name for name, values in selection_rows.items() if values is not None), None)
    adapter = SharedCausalLM(model.native)
    if adapter.model is not model.native.model.text_model or adapter.lm_head is not model.native.lm_head:
        raise AssertionError('adapter must share native text weights')
    parameters = [(parameter, parameter._version, parameter.data_ptr()) for parameter in model.parameters()]
    config = greedy_config(model, 4)
    tokenizer = model.processor.tokenizer
    singles = {}
    def selections(indices):
        if active is None:
            return [None] * len(indices)
        return [[int(value) for value in selection_rows[active][int(rows[index].get('row_index', index))]] for index in indices]
    for index in probes:
        batch = build_batch([rows[index]])
        fused = fuse_batch(model, batch)
        baseline = compact_prompt(fused, [None])
        actual = generate_compacted(adapter, baseline, tokenizer, config)
        device = fused['input_ids'].device
        native_ids = model.generate_from_features(input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device), visual_features=batch['visual_features'],
            generation_config=config)
        expected = decode_sequences(tokenizer, native_ids, config.eos_token_id, 4)
        if actual != expected:
            raise AssertionError('allkeep adapter differs from native generation')
        native_logits = model.native(input_ids=fused['input_ids'], inputs_embeds=fused['inputs_embeds'],
            attention_mask=fused['attention_mask'], position_ids=fused['position_ids'], use_cache=False).logits[:, -1]
        _close(_last_logits(adapter, baseline), native_logits)
        compact = compact_prompt(fused, selections([index]))
        cached = generate_compacted(adapter, compact, tokenizer, config)
        if cached != _uncached(adapter, compact, tokenizer, config):
            raise AssertionError('cached pruning differs from uncached full prefix')
        singles[index] = (_last_logits(adapter, compact), cached[0])
        if progress is not None:
            progress(f'preflight row {index} native and cached parity passed')
    fused = fuse_batch(model, build_batch([rows[index] for index in probes]))
    compact = compact_prompt(fused, selections(probes))
    logits = _last_logits(adapter, compact)
    records = generate_compacted(adapter, compact, tokenizer, config)
    for slot, index in enumerate(probes):
        _close(logits[slot:slot + 1], singles[index][0], batched=True)
        if records[slot] != singles[index][1]:
            raise AssertionError('batched generation differs from singleton generation')
    if any(parameter._version != version or parameter.data_ptr() != pointer for parameter, version, pointer in parameters):
        raise AssertionError('preflight changed native parameters')
    return dict(passed=True, probe_row_indices=row_indices, pruned_config=active,
                max_new_tokens=4, atol=1e-6, batch_atol=1e-5, rtol=1e-4)
