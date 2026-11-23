"""Mandatory numerical parity and fixed-size stress for paired pruning runs."""
import time

import torch

from smolvlm_topk_benchmark.generation import (
    SharedCausalLM, compact_prompt, decode_sequences, fuse_batch,
    generate_compacted, greedy_config,
)
from smolvlm_topk_benchmark.preflight import (
    numerical_preflight as input_numerical_preflight,
)

_NAMES = ('score22_prune_input', 'score22_prune_after2')


def _length(row):
    if 'input_ids' in row:
        return len(row['input_ids'])
    if 'prompt_length' in row:
        return int(row['prompt_length'])
    raise ValueError('preflight rows need input_ids or prompt_length')


def _validate(rows, config_name):
    if config_name not in _NAMES:
        raise ValueError('unknown paired benchmark config')
    if not rows:
        raise ValueError('preflight needs at least one sample row')


def _selections(rows, selections):
    return [None if selections is None else
            [int(value) for value in selections[int(row.get('row_index', index))]]
            for index, row in enumerate(rows)]


@torch.inference_mode()
def numerical_preflight(model, rows, build_batch, selections, config_name, *, progress=None):
    """Compare real cached decoding with native/all-keep and slow references."""
    _validate(rows, config_name)
    if config_name == _NAMES[0]:
        result = input_numerical_preflight(model, rows, build_batch,
            {config_name: selections}, progress=progress)
        return dict(result, config_name=config_name)
    from .generation import DeferredGreedyDecoder
    probes = list(dict.fromkeys([min(range(len(rows)), key=lambda i: _length(rows[i])),
                                max(range(len(rows)), key=lambda i: _length(rows[i]))]))
    versions = [(parameter, parameter._version, parameter.data_ptr())
                for parameter in model.parameters()]
    rng = torch.get_rng_state().clone()
    device = next(model.parameters()).device
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None
    config = greedy_config(model, 4)
    decoder = DeferredGreedyDecoder(model, prune_after_layer_index=2)
    singles = {}
    for index in probes:
        batch = build_batch([rows[index]])
        fused = fuse_batch(model, batch)
        all_keep = decoder.generate(fused, [None], config)
        native_ids = model.generate_from_features(input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            visual_features=batch['visual_features'], generation_config=config)
        native = decode_sequences(model.processor.tokenizer, native_ids,
                                  config.eos_token_id, config.max_new_tokens)
        if all_keep != native:
            raise AssertionError('deferred all-keep differs from native greedy')
        selection = _selections([rows[index]], selections)
        actual = decoder.generate(fused, selection, config)
        reference = decoder.generate_reference(fused, selection, config)
        if actual != reference:
            raise AssertionError('deferred cached pruning differs from full-history reference')
        singles[index] = actual[0]
        if progress is not None:
            progress(f'preflight row {index}: native/all-keep and full-history parity passed')
    selected_rows = [rows[index] for index in probes]
    fused = fuse_batch(model, build_batch(selected_rows))
    records = decoder.generate(fused, _selections(selected_rows, selections), config)
    if records != [singles[index] for index in probes]:
        raise AssertionError('deferred batched generation differs from singleton generation')
    if any(parameter._version != version or parameter.data_ptr() != pointer
           for parameter, version, pointer in versions):
        raise AssertionError('preflight changed native parameters')
    if not torch.equal(rng, torch.get_rng_state()) or (cuda_rng is not None and
            not torch.equal(cuda_rng, torch.cuda.get_rng_state(device))):
        raise AssertionError('preflight changed RNG state')
    return dict(passed=True, config_name=config_name,
                probe_row_indices=[int(rows[index].get('row_index', index)) for index in probes],
                max_new_tokens=4, checks=['native_all_keep', 'cached_full_history',
                                        'singleton_batched', 'weights_rng_unchanged'])


@torch.inference_mode()
def stress(model, rows, build_batch, selections, config_name, *, batch_size=64,
           max_new_tokens=512):
    """Force the full decode budget on the longest fixed-size prompt batch.

    EOS is disabled only in this local stress GenerationConfig. Production
    predictions continue to use native EOS semantics and the unchanged model.
    """
    _validate(rows, config_name)
    if not isinstance(batch_size, int) or batch_size <= 0 or len(rows) < batch_size:
        raise ValueError('stress requires the complete requested batch size of samples')
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError('stress needs a positive generation budget')
    ranked = sorted(range(len(rows)), key=lambda index: (-_length(rows[index]), index))[:batch_size]
    selected_rows = [rows[index] for index in ranked]
    longest = max(_length(row) for row in selected_rows)
    if longest + max_new_tokens > model.native.config.text_config.max_position_embeddings:
        raise ValueError('stress prompt plus generation exceeds native context')
    config = greedy_config(model, max_new_tokens)
    config.eos_token_id = None
    config.forced_eos_token_id = None
    device = next(model.parameters()).device
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    fused = fuse_batch(model, build_batch(selected_rows))
    keep = _selections(selected_rows, selections)
    if config_name == _NAMES[0]:
        records = generate_compacted(SharedCausalLM(model.native), compact_prompt(fused, keep),
                                     model.processor.tokenizer, config)
    else:
        from .generation import DeferredGreedyDecoder
        records = DeferredGreedyDecoder(model, prune_after_layer_index=2).generate(fused, keep, config)
    if len(records) != batch_size:
        raise AssertionError('stress output batch size differs')
    counts = [len(record['generated_ids']) for record in records]
    if counts != [max_new_tokens] * batch_size:
        raise AssertionError('stress did not generate the full forced continuation budget')
    return dict(passed=True, config_name=config_name, batch_size=batch_size,
                probe_row_indices=[int(row.get('row_index', index)) for index, row in zip(ranked, selected_rows)],
                max_prompt_tokens=longest, max_new_tokens=max_new_tokens,
                eos_policy='disabled_for_stress_only', generated_token_counts=counts,
                elapsed_seconds=time.perf_counter() - started,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None)
