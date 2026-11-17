"""Attention at generated-token queries, reduced before leaving each layer.

The caller supplies generated content IDs (without EOS/padding). Replay uses the
same causal text model as generation and includes the query's own key.
"""
import math

import torch
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from .collector import _prepare


def _prepare_generated(model, batch, generated_ids):
    ids, mask = batch['input_ids'], batch['attention_mask']
    if ids.ndim != 2 or ids.shape != mask.shape or len(generated_ids) != ids.shape[0]:
        raise ValueError('generated_ids must have one sequence per batch row')
    if not ((mask == 0) | (mask == 1)).all() or not mask.bool().any(-1).all():
        raise ValueError('prompt mask must contain valid binary entries')
    counts, prompts, sequences = [], [], []
    image_id = model.native.config.image_token_id
    vocab_size = model.native.config.text_config.vocab_size
    for row, valid, generated in zip(ids, mask.bool(), generated_ids):
        if any(isinstance(token, bool) or not isinstance(token, int) or
               token < 0 or token >= vocab_size for token in generated):
            raise ValueError('generated token IDs must be valid integer vocabulary IDs')
        if image_id in generated:
            raise ValueError('generated content contains reserved image token ID')
        prompt = row[valid].tolist()
        prompts.append(len(prompt))
        counts.append(len(generated))
        sequences.append(prompt + list(generated))
    length = max(map(len, sequences))
    joint_ids = ids.new_full((len(sequences), length), model.processor.tokenizer.pad_token_id)
    joint_mask = mask.new_zeros(joint_ids.shape)
    for i, sequence in enumerate(sequences):
        joint_ids[i, -len(sequence):] = torch.tensor(sequence, device=ids.device)
        joint_mask[i, -len(sequence):] = 1
    joint = dict(batch, input_ids=joint_ids, attention_mask=joint_mask)
    text, fused, mask, position_ids, _, image_positions = _prepare(model, joint)
    positions = torch.arange(length, device=mask.device)[None, :]
    starts = torch.tensor([length - len(s) for s in sequences], device=mask.device)
    boundaries = starts + torch.tensor(prompts, device=mask.device)
    image_mask = torch.zeros_like(mask, dtype=torch.bool).scatter_(1, image_positions, True)
    generated_mask = (positions >= boundaries[:, None]) & mask.bool()
    prompt_mask = mask.bool() & ~generated_mask & ~image_mask
    qlength = max(1, max(counts))
    query_positions = boundaries[:, None] + torch.arange(qlength, device=mask.device)[None]
    query_valid = torch.arange(qlength, device=mask.device)[None] < torch.tensor(counts, device=mask.device)[:, None]
    query_positions = query_positions.clamp(max=length - 1)
    return (text, fused, mask, position_ids, image_positions, query_positions,
            query_valid, (image_mask, prompt_mask, generated_mask), counts)


def _reduce(probs, image_positions, query_valid, group_masks):
    # probs is [B,H,Q,T]; each sample has an independent generated-token mean.
    mean_heads = probs.mean(1)
    weights = query_valid.to(mean_heads.dtype)
    weights = weights / weights.sum(-1, keepdim=True).clamp(min=1)
    mean = (mean_heads * weights[..., None]).sum(1)
    visual = mean.gather(1, image_positions)
    groups = torch.stack([(mean * group_mask).sum(-1) for group_mask in group_masks], -1)
    return visual.cpu(), groups.cpu()


def _result(visual, groups, counts):
    visual, groups = torch.stack(visual, 1).float(), torch.stack(groups, 1).float()
    if not torch.isfinite(visual).all() or not torch.isfinite(groups).all():
        raise ValueError('nonfinite generated attention')
    nonempty = torch.tensor(counts) > 0
    if (visual < 0).any() or (groups < 0).any() or not torch.allclose(
            groups[nonempty].sum(-1), torch.ones_like(groups[nonempty, :, 0]), atol=2e-6, rtol=1e-5):
        raise ValueError('invalid generated attention probabilities')
    return dict(visual=visual, groups=groups, token_counts=counts)


@torch.inference_mode()
def collect_generated_attention(model, batch, generated_ids):
    """Return CPU visual[B,L,V], groups[B,L,3], and valid token_counts.

    Groups are Image, Prompt text, Generated. Empty answers contribute zeros;
    callers must exclude them from the sample denominator.
    """
    (text, fused, mask, position_ids, image_positions, query_positions,
     query_valid, group_masks, counts) = _prepare_generated(model, batch, generated_ids)
    rows = torch.arange(mask.shape[0], device=mask.device)[:, None]
    visuals, groups, handles = [], [], []

    def capture(attn, args, kwargs):
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        size, length, _ = hidden.shape
        query = attn.q_proj(hidden[rows, query_positions]).view(
            size, query_positions.shape[1], attn.num_heads, attn.head_dim).transpose(1, 2)
        keys = attn.k_proj(hidden).view(size, length, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        embeddings = kwargs.get('position_embeddings')
        if embeddings is None:
            embeddings = attn.rotary_emb(keys, position_ids)
        cos, sin = embeddings
        qcos, qsin = cos[rows, query_positions][:, None], sin[rows, query_positions][:, None]
        query = query * qcos + rotate_half(query) * qsin
        keys = keys * cos[:, None] + rotate_half(keys) * sin[:, None]
        keys = repeat_kv(keys, attn.num_key_value_groups)
        scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(attn.head_dim)
        allowed = mask[:, None, :].bool() & (torch.arange(length, device=mask.device)[None, None] <= query_positions[..., None])
        scores.masked_fill_(~allowed[:, None], torch.finfo(scores.dtype).min)
        visual, group = _reduce(scores.softmax(-1), image_positions, query_valid, group_masks)
        visuals.append(visual)
        groups.append(group)

    try:
        for layer in text.layers:
            handles.append(layer.self_attn.register_forward_pre_hook(capture, with_kwargs=True))
        text(inputs_embeds=fused, attention_mask=mask, position_ids=position_ids,
             use_cache=False, output_attentions=False, output_hidden_states=False)
    finally:
        for handle in handles:
            handle.remove()
    if len(visuals) != len(text.layers):
        raise RuntimeError('generated attention hooks did not run once per layer')
    return _result(visuals, groups, counts)


@torch.inference_mode()
def reference_generated_attention(model, batch, generated_ids):
    """Tiny-fixture oracle; native eager attention allocates full T-by-T arrays."""
    (text, fused, mask, position_ids, image_positions, query_positions,
     query_valid, group_masks, counts) = _prepare_generated(model, batch, generated_ids)
    output = text(inputs_embeds=fused, attention_mask=mask, position_ids=position_ids,
                  use_cache=False, output_attentions=True, output_hidden_states=False,
                  return_dict=True)
    visuals, groups = [], []
    for attention in output.attentions:
        selected = attention.gather(2, query_positions[:, None, :, None].expand(
            -1, attention.shape[1], -1, attention.shape[-1]))
        visual, group = _reduce(selected, image_positions, query_valid, group_masks)
        visuals.append(visual)
        groups.append(group)
    return _result(visuals, groups, counts)
