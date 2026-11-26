"""Same-layer direct plus nonvisual two-step attention, mean heads first."""
import math
import torch
from transformers.models.llama.modeling_llama import rotate_half, repeat_kv
from .collector import _prepare, _validate

LAYER_IDS = tuple(range(0, 29, 2))


def _layers(text, layer_ids, chunk=8):
    ids = tuple(layer_ids)
    if (not ids or len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < len(text.layers) for i in ids)
            or type(chunk) is not int or chunk < 1):
        raise ValueError('invalid selected layers or query chunk size')
    return ids


def _nonvisual(mask, last, image_positions):
    valid = mask.bool().clone()
    valid.scatter_(1, image_positions, False)
    valid.scatter_(1, last[:, None], False)
    return valid


def combine_attention(attention, mask, last, image_positions):
    """Tiny full-matrix oracle reducer; input [B,H,T,T], no renormalization."""
    mean = attention.float().mean(1)
    rows = torch.arange(mask.shape[0], device=mask.device)
    q = mean[rows, last]
    via = _nonvisual(mask, last, image_positions)
    columns = mean.gather(2, image_positions[:, None].expand(-1, mean.shape[1], -1))
    return q.gather(1, image_positions) + ((q * via)[:, :, None] * columns).sum(1)


@torch.inference_mode()
def collect_combined_visual_attention(model, batch, layer_ids=LAYER_IDS, query_chunk_size=8):
    text, fused, mask, position_ids, last, images = _prepare(model, batch)
    selected = _layers(text, layer_ids, query_chunk_size)
    rows = torch.arange(mask.shape[0], device=mask.device)
    nonvisual = _nonvisual(mask, last, images)
    intermediates = [torch.where(row)[0] for row in nonvisual]
    collected, handles = {}, []

    def capture(index, attn, args, kwargs):
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        b, length, _ = hidden.shape
        keys = attn.k_proj(hidden).view(b, length, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        embeddings = kwargs.get('position_embeddings')
        cos, sin = embeddings if embeddings is not None else attn.rotary_emb(keys, position_ids)
        keys = repeat_kv(keys * cos[:, None] + rotate_half(keys) * sin[:, None], attn.num_key_value_groups)
        key_positions = torch.arange(length, device=hidden.device)

        def probabilities(positions):
            c = positions.shape[1]
            query = attn.q_proj(hidden[rows[:, None], positions]).view(b, c, attn.num_heads, attn.head_dim).transpose(1, 2)
            qc, qs = cos[rows[:, None], positions][:, None], sin[rows[:, None], positions][:, None]
            query = query * qc + rotate_half(query) * qs
            scores = query @ keys.transpose(-1, -2) / math.sqrt(attn.head_dim)
            allowed = mask[:, None, :].bool() & (key_positions[None, None] <= positions[:, :, None])
            scores.masked_fill_(~allowed[:, None], torch.finfo(scores.dtype).min)
            return scores.softmax(-1, dtype=torch.float32).mean(1)

        q = probabilities(last[:, None])[:, 0]
        result = q.gather(1, images).clone()
        for start in range(0, max(map(len, intermediates)), query_chunk_size):
            size = min(query_chunk_size, max(map(len, intermediates)) - start)
            # Use each row's valid final query as a safe dummy; zero its weight.
            positions = last[:, None].expand(-1, size).clone()
            active = torch.zeros_like(positions, dtype=torch.bool)
            for r, indices in enumerate(intermediates):
                part = indices[start:start + size]
                positions[r, :len(part)] = part
                active[r, :len(part)] = True
            probs = probabilities(positions)
            visual = probs.gather(2, images[:, None].expand(-1, size, -1))
            weights = q.gather(1, positions) * active
            result += (weights[:, :, None] * visual).sum(1)
        if index in collected:
            raise RuntimeError('selected layer called more than once')
        collected[index] = result.cpu()

    try:
        for index in selected:
            handles.append(text.layers[index].self_attn.register_forward_pre_hook(
                lambda m, a, kw, index=index: capture(index, m, a, kw), with_kwargs=True))
        text(inputs_embeds=fused, attention_mask=mask, position_ids=position_ids,
             use_cache=False, output_attentions=False, output_hidden_states=False)
    finally:
        for handle in handles: handle.remove()
    if set(collected) != set(selected):
        raise RuntimeError('selected attention hooks did not all run')
    return _validate(torch.stack([collected[i] for i in selected], 1))


@torch.inference_mode()
def reference_combined_visual_attention(model, batch, layer_ids=LAYER_IDS):
    """Native eager oracle, consuming/discarding selected matrices per layer."""
    text, fused, mask, position_ids, last, images = _prepare(model, batch)
    selected = _layers(text, layer_ids)
    results, handles = {}, []

    def enable(module, args, kwargs):
        kwargs = dict(kwargs, output_attentions=True)
        return args, kwargs

    def disable(module, args, kwargs):
        return args, dict(kwargs, output_attentions=False)

    def consume(index, module, args, output):
        results[index] = combine_attention(output[1], mask, last, images).cpu()
        return (output[0], None, *output[2:])

    try:
        for index, layer in enumerate(text.layers):
            if index not in selected:
                handles.append(layer.self_attn.register_forward_pre_hook(disable, with_kwargs=True))
        for index in selected:
            attn = text.layers[index].self_attn
            handles.append(attn.register_forward_pre_hook(enable, with_kwargs=True))
            handles.append(attn.register_forward_hook(lambda m,a,o,index=index: consume(index,m,a,o)))
        text(inputs_embeds=fused, attention_mask=mask, position_ids=position_ids,
             use_cache=False, output_attentions=True, output_hidden_states=False)
    finally:
        for handle in handles: handle.remove()
    return _validate(torch.stack([results[i] for i in selected], 1))


@torch.inference_mode()
def numerical_preflight(model, rows, build_batch, layer_ids=LAYER_IDS, progress=None):
    """Real-checkpoint oracle on shortest row and mixed short/long batch parity."""
    if not rows: raise ValueError('preflight requires samples')
    indices = list(dict.fromkeys((min(range(len(rows)), key=lambda i: len(rows[i]['input_ids'])),
                                 max(range(len(rows)), key=lambda i: len(rows[i]['input_ids'])))))
    singles = []
    for index in indices:
        batch = build_batch([rows[index]])
        actual = collect_combined_visual_attention(model, batch, layer_ids)
        if not singles:
            torch.testing.assert_close(actual, reference_combined_visual_attention(model, batch, layer_ids), atol=1e-6, rtol=1e-4)
        singles.append(actual)
    mixed = collect_combined_visual_attention(model, build_batch([rows[i] for i in indices]), layer_ids)
    torch.testing.assert_close(mixed, torch.cat(singles), atol=1e-5, rtol=1e-4)
    if progress: progress('combined attention native oracle and batch/single parity passed')
    return dict(status='passed', samples=len(indices), layer_ids=list(layer_ids))
