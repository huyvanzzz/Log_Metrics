"""Fixed-answer replay, selective native attention, and embedding perturbations.

Generated sequences are the original IDs before EOS, including intervening
special tokens. Queries inspect the generated token itself (self key included),
while scoring uses its predecessor. No perturbation reruns vision or fusion.
"""
import hashlib
import json
import math
from numbers import Integral

import numpy as np
import torch
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from smolvlm_attention.collector import _prepare


RANKING_LAYERS = (1, 7, 13, 19, 25)
LAYER_GROUPS = tuple(tuple(range(start, start + 5)) for start in range(0, 30, 5))


@torch.inference_mode()
def prepare_replay(model, batch, generated_ids):
    """Build one left-padded fused replay without deleting generated specials.

    ``query_positions`` and ``content_mask`` retain original generated indexing;
    ``query_valid`` equals content_mask and excludes every tokenizer special ID.
    Padded query slots are invalid. An empty sample has one invalid slot.
    """
    ids, mask = batch['input_ids'], batch['attention_mask']
    if ids.ndim != 2 or ids.shape != mask.shape or not ids.shape[0] or not ids.shape[1]:
        raise ValueError('input_ids and attention_mask must be nonempty [B,T]')
    if len(generated_ids) != ids.shape[0]:
        raise ValueError('generated_ids must have one sequence per batch row')
    if not ((mask == 0) | (mask == 1)).all() or not mask.bool().any(-1).all():
        raise ValueError('prompt mask must contain valid binary entries')
    image_id = model.native.config.image_token_id
    vocab_size = model.native.config.text_config.vocab_size
    specials = set(model.processor.tokenizer.all_special_ids)
    sequences, counts, prompt_lengths, content = [], [], [], []
    for row, valid, generated in zip(ids, mask.bool(), generated_ids):
        generated = list(generated)
        if any(isinstance(token, bool) or not isinstance(token, Integral) or
               token < 0 or token >= vocab_size for token in generated):
            raise ValueError('generated token IDs must be valid integer vocabulary IDs')
        if image_id in generated:
            raise ValueError('generated content contains reserved image token ID')
        prompt = row[valid].tolist()
        prompt_lengths.append(len(prompt))
        counts.append(len(generated))
        sequences.append(prompt + generated)
        content.append([token not in specials for token in generated])
    length = max(map(len, sequences))
    joint_ids = ids.new_full((len(sequences), length), model.processor.tokenizer.pad_token_id)
    joint_mask = mask.new_zeros(joint_ids.shape)
    for i, sequence in enumerate(sequences):
        joint_ids[i, -len(sequence):] = torch.tensor(sequence, device=ids.device)
        joint_mask[i, -len(sequence):] = 1
    text, fused, mask, position_ids, _, image_positions = _prepare(
        model, dict(batch, input_ids=joint_ids, attention_mask=joint_mask))
    device = mask.device
    slots = max(1, max(counts))
    starts = torch.tensor([length-len(seq)+prompt for seq, prompt in zip(sequences, prompt_lengths)], device=device)
    query_positions = (starts[:, None] + torch.arange(slots, device=device)[None]).clamp(max=length-1)
    content_mask = torch.zeros((len(sequences), slots), dtype=torch.bool, device=device)
    for i, values in enumerate(content):
        content_mask[i, :len(values)] = torch.tensor(values, dtype=torch.bool, device=device)
    return dict(text=text, fused=fused, input_ids=joint_ids.to(device), attention_mask=mask,
                position_ids=position_ids, image_positions=image_positions,
                query_positions=query_positions, query_valid=content_mask,
                content_mask=content_mask, generated_lengths=counts)


def _score_hidden(model, replay, hidden, logit_chunk_size):
    if isinstance(logit_chunk_size, bool) or not isinstance(logit_chunk_size, Integral) or logit_chunk_size <= 0:
        raise ValueError('logit_chunk_size must be a positive integer')
    sample, slot = replay['query_valid'].nonzero(as_tuple=True)
    queries = replay['query_positions'][sample, slot]
    if (queries <= 0).any():
        raise ValueError('generated scoring positions must have predecessors')
    values = []
    for start in range(0, len(sample), logit_chunk_size):
        rows, positions = sample[start:start+logit_chunk_size], queries[start:start+logit_chunk_size]
        logits = model.native.lm_head(hidden[rows, positions-1]).float()
        tokens = replay['input_ids'][rows, positions]
        values.append(logits.log_softmax(-1).gather(1, tokens[:, None])[:, 0].cpu())
    values = torch.cat(values) if values else torch.empty(0, dtype=torch.float32)
    if not torch.isfinite(values).all():
        raise ValueError('nonfinite fixed-answer log probabilities')
    result, offset = [], 0
    for count in replay['query_valid'].sum(-1).tolist():
        token_values = values[offset:offset+count]
        result.append(dict(mean_logprob=token_values.double().mean().item() if count else None,
                           token_logprobs=token_values.tolist(), token_count=count))
        offset += count
    return result


def _forward(replay, fused):
    return replay['text'](inputs_embeds=fused, attention_mask=replay['attention_mask'],
                          position_ids=replay['position_ids'], use_cache=False,
                          output_attentions=False, output_hidden_states=False,
                          return_dict=True)


def _indices(indices, visual_tokens):
    indices = list(indices)
    if any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 or i >= visual_tokens for i in indices):
        raise ValueError('visual indices must be integers within visual geometry')
    if len(set(indices)) != len(indices):
        raise ValueError('visual indices must be unique')
    return indices


@torch.inference_mode()
def score_replay(model, replay, *, zero_indices=None, logit_chunk_size=128):
    """Score content IDs from q-1, optionally zeroing per-sample visual indices.

    Returns a list of ``{mean_logprob, token_logprobs, token_count}``. Empty
    content has mean_logprob=None; token_logprobs follows content-token order.
    """
    fused = replay['fused'].clone()
    if zero_indices is not None:
        if len(zero_indices) != fused.shape[0]:
            raise ValueError('zero_indices must have one list per sample')
        for row, selected in enumerate(zero_indices):
            selected = _indices(selected, replay['image_positions'].shape[1])
            fused[row, replay['image_positions'][row, selected]] = 0
    output = _forward(replay, fused)
    return _score_hidden(model, replay, output.last_hidden_state, logit_chunk_size)


@torch.inference_mode()
def collect_baseline(model, batch, generated_ids, *, target_queries=None,
                     ranking_layers=RANKING_LAYERS, layer_groups=LAYER_GROUPS,
                     query_chunk_size=16, logit_chunk_size=128):
    """One decoder replay for scores, causal rankings, and target group maps.

    ``target_queries[b][target]`` holds original generated-token indices. Specials
    are excluded. Ranking is CPU FP32 [B,R,V]; target_maps[b] is [targets,G,V].
    The transient ``replay`` entry can be reused for causal scoring, then dropped
    when persisting the baseline. No all-layer quadratic tensors are retained.
    """
    replay = prepare_replay(model, batch, generated_ids)
    text, fused = replay['text'], replay['fused']
    size, length = fused.shape[:2]
    visual_tokens = replay['image_positions'].shape[1]
    ranking_layers = tuple(ranking_layers)
    layer_groups = tuple(tuple(group) for group in layer_groups)
    layers = set(ranking_layers).union(*(set(group) for group in layer_groups))
    if any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 or i >= len(text.layers) for i in layers):
        raise ValueError('requested attention layer outside native decoder')
    if len(set(ranking_layers)) != len(ranking_layers) or any(not group or len(set(group)) != len(group) for group in layer_groups):
        raise ValueError('ranking layers must be unique and layer groups nonempty and unique')
    if isinstance(query_chunk_size, bool) or not isinstance(query_chunk_size, Integral) or query_chunk_size <= 0:
        raise ValueError('query_chunk_size must be a positive integer')
    if target_queries is None:
        target_queries = [[] for _ in range(size)]
    if len(target_queries) != size:
        raise ValueError('target_queries must have one list per sample')
    valid = replay['query_valid']
    target_weights = []
    for row, targets in enumerate(target_queries):
        weights = torch.zeros((len(targets), valid.shape[1]), device=fused.device)
        for target, indices in enumerate(targets):
            indices = _indices(indices, replay['generated_lengths'][row])
            weights[target, indices] = valid[row, indices].float()
        weights /= weights.sum(-1, keepdim=True).clamp(min=1)
        target_weights.append(weights)
    ranking = torch.zeros((size, len(ranking_layers), visual_tokens), dtype=torch.float32)
    maps = [torch.zeros((len(targets), len(layer_groups), visual_tokens), dtype=torch.float32) for targets in target_queries]
    counts = valid.sum(-1).clamp(min=1)
    rows = torch.arange(size, device=fused.device)[:, None]
    all_positions = torch.arange(length, device=fused.device)[None, None, :]
    handles, observed = [], []

    def capture(layer_index, attn, args, kwargs):
        observed.append(layer_index)
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        keys = attn.k_proj(hidden).view(size, length, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        embeddings = kwargs.get('position_embeddings')
        if embeddings is None:
            embeddings = attn.rotary_emb(keys, replay['position_ids'])
        cos, sin = embeddings
        keys = repeat_kv(keys*cos[:, None] + rotate_half(keys)*sin[:, None], attn.num_key_value_groups)
        rank_layer = torch.zeros((size, visual_tokens), dtype=torch.float32, device=fused.device)
        target_layer = [torch.zeros((len(targets), visual_tokens), dtype=torch.float32, device=fused.device) for targets in target_queries]
        for start in range(0, valid.shape[1], query_chunk_size):
            stop = start + query_chunk_size
            positions = replay['query_positions'][:, start:stop]
            query = attn.q_proj(hidden[rows, positions]).view(size, positions.shape[1], attn.num_heads, attn.head_dim).transpose(1, 2)
            query = query*cos[rows, positions][:, None] + rotate_half(query)*sin[rows, positions][:, None]
            scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(attn.head_dim)
            allowed = replay['attention_mask'][:, None].bool() & (all_positions <= positions[..., None])
            scores.masked_fill_(~allowed[:, None], torch.finfo(scores.dtype).min)
            visual = scores.float().softmax(-1).mean(1).gather(
                2, replay['image_positions'][:, None].expand(-1, positions.shape[1], -1))
            rank_layer.add_((visual * valid[:, start:stop, None]).sum(1) / counts[:, None])
            for row in range(size):
                target_layer[row].add_(target_weights[row][:, start:stop] @ visual[row])
        if layer_index in ranking_layers:
            ranking[:, ranking_layers.index(layer_index)] = rank_layer.cpu()
        for group_index, group in enumerate(layer_groups):
            if layer_index in group:
                for row in range(size):
                    maps[row][:, group_index].add_(target_layer[row].cpu() / len(group))

    try:
        for index in sorted(layers):
            def hook(attn, args, kwargs, index=index):
                capture(index, attn, args, kwargs)
            handles.append(text.layers[index].self_attn.register_forward_pre_hook(hook, with_kwargs=True))
        output = _forward(replay, fused)
    finally:
        for handle in handles:
            handle.remove()
    if observed != sorted(layers):
        raise RuntimeError('baseline attention hooks did not run once in decoder order')
    for values in [ranking, *maps]:
        if not torch.isfinite(values).all() or (values < 0).any() or (values.sum(-1) > 1+1e-5).any():
            raise ValueError('invalid reduced baseline attention probabilities')
    return dict(scores=_score_hidden(model, replay, output.last_hidden_state, logit_chunk_size),
                ranking=ranking, target_maps=maps, replay=replay, ranking_layers=list(ranking_layers))


def select_indices(attention, k=32, largest=True):
    """Stable Top/Bottom selection; equal scores favor smaller visual indices."""
    attention = torch.as_tensor(attention).detach().cpu()
    if attention.ndim != 1 or not torch.isfinite(attention).all():
        raise ValueError('attention must be a finite one-dimensional vector')
    if isinstance(k, bool) or not isinstance(k, Integral) or not 0 <= k <= len(attention):
        raise ValueError('k must lie within visual geometry')
    return torch.argsort(attention, descending=largest, stable=True)[:k].tolist()


def random_indices(sample_id, k=32, visual_tokens=384, seed=42, repeats=3):
    """Independent SHA256([seed, sample_id, repeat]) seeds feeding NumPy PCG64."""
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (k, visual_tokens, seed, repeats)) or not 0 <= k <= visual_tokens or visual_tokens <= 0 or repeats <= 0:
        raise ValueError('invalid random selection geometry, seed, or repeats')
    result = []
    for repeat in range(repeats):
        payload = json.dumps([int(seed), str(sample_id), repeat], ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        derived = int.from_bytes(hashlib.sha256(payload).digest(), 'big')
        rng = np.random.Generator(np.random.PCG64(derived))
        result.append(rng.choice(visual_tokens, size=k, replace=False).tolist())
    return result


def relative_drop(base_logprob, changed_logprob):
    """Geometric-mean probability drop, preserving negative values."""
    if not math.isfinite(base_logprob) or not math.isfinite(changed_logprob):
        raise ValueError('relative drop requires finite log probabilities')
    try:
        result = -math.expm1(changed_logprob-base_logprob)
    except OverflowError as exc:
        raise ValueError('nonfinite relative probability drop') from exc
    if not math.isfinite(result):
        raise ValueError('nonfinite relative probability drop')
    return result


@torch.inference_mode()
def measure_causal(model, batch, generated_ids, baseline, *, sample_ids, k=32,
                   seed=42, ranking_layers=RANKING_LAYERS):
    """Run 26 conditions sequentially for the default five ranking layers.

    Each row contains baseline_score, token_count, layers[{layer,top,bottom}],
    and random{repeats,comp,sgap}. Each condition stores indices, remove_score,
    keep_score, comp, sgap. Random repeats are shared by all ranking layers.
    """
    replay = baseline.get('replay')
    if replay is None:
        replay = prepare_replay(model, batch, generated_ids)
    size, visual_tokens = replay['image_positions'].shape
    if len(sample_ids) != size or len(baseline['scores']) != size:
        raise ValueError('sample IDs and baseline scores must match replay batch')
    if any(score['mean_logprob'] is None or score['token_count'] <= 0 for score in baseline['scores']):
        raise ValueError('causal measurement requires nonempty content in every sample')
    available = list(baseline.get('ranking_layers', ranking_layers))
    if baseline['ranking'].shape != (size, len(available), visual_tokens) or any(layer not in available for layer in ranking_layers):
        raise ValueError('baseline ranking layers or geometry mismatch')
    random = [random_indices(sample, k=k, visual_tokens=visual_tokens, seed=seed) for sample in sample_ids]
    results = [dict(baseline_score=score['mean_logprob'], token_count=score['token_count'], layers=[], random={}) for score in baseline['scores']]

    def condition(selected):
        removed = score_replay(model, replay, zero_indices=selected)
        complements = [sorted(set(range(visual_tokens))-set(indices)) for indices in selected]
        kept = score_replay(model, replay, zero_indices=complements)
        return [dict(indices=indices, remove_score=remove['mean_logprob'], keep_score=keep['mean_logprob'],
                     comp=relative_drop(base['baseline_score'], remove['mean_logprob']),
                     sgap=relative_drop(base['baseline_score'], keep['mean_logprob']))
                for indices, remove, keep, base in zip(selected, removed, kept, results)]

    for layer in ranking_layers:
        entries = [dict(layer=layer) for _ in range(size)]
        for name, largest in [('top', True), ('bottom', False)]:
            selected = [select_indices(attention, k=k, largest=largest) for attention in baseline['ranking'][:, available.index(layer)]]
            for entry, scores in zip(entries, condition(selected)):
                entry[name] = scores
        for result, entry in zip(results, entries):
            result['layers'].append(entry)
    repeats = [[] for _ in range(size)]
    for repeat in range(3):
        for row, scores in enumerate(condition([indices[repeat] for indices in random])):
            repeats[row].append(scores)
    for result, values in zip(results, repeats):
        result['random'] = dict(repeats=values, comp=sum(v['comp'] for v in values)/3,
                                sgap=sum(v['sgap'] for v in values)/3)
    return results
