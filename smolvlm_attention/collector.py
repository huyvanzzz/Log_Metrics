"""Final valid prompt query attention; public results are FP32 CPU [B,L,V].

The collector retains only [B,H,1,T] scores per layer. The reference deliberately
uses native quadratic eager attention and is intended only for tiny parity tests.
"""
import math

import torch
import transformers
from transformers.models.llama.modeling_llama import LlamaSdpaAttention, rotate_half, repeat_kv


def _prepare(model, batch):
    if transformers.__version__ != '4.46.3':
        raise ValueError('attention collection requires transformers==4.46.3')
    if any(module.training for module in model.modules()):
        raise ValueError('attention collection requires eval mode')
    if torch.is_autocast_enabled() or torch.is_autocast_enabled('cpu'):
        raise ValueError('attention collection requires FP32 without autocast')
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('attention collection requires FP32 model parameters')
    text = model.native.model.text_model
    if text.config._attn_implementation != 'sdpa' or text.config.pretraining_tp != 1:
        raise ValueError('requires native SDPA with pretraining_tp=1')
    for layer in text.layers:
        attn = layer.self_attn
        if type(attn) is not LlamaSdpaAttention or attn.num_heads % attn.num_key_value_heads:
            raise ValueError('requires native LlamaSdpaAttention with divisible GQA heads')
    device = next(text.parameters()).device
    ids, mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
    if ids.ndim != 2 or mask.shape != ids.shape or not ids.shape[0] or not ids.shape[1]:
        raise ValueError('input_ids and attention_mask must be nonempty [B,T]')
    if not ((mask == 0) | (mask == 1)).all() or not mask.bool().any(-1).all():
        raise ValueError('attention_mask must be binary with valid keys in every row')
    valid = mask.bool()
    visual = (ids == model.native.config.image_token_id) & valid
    if not (visual.sum(-1) == model.visual_token_len).all():
        raise ValueError('incorrect valid image token count')
    positions = torch.arange(ids.shape[1], device=device).expand_as(ids)
    last = positions.masked_fill(~valid, -1).max(-1).values
    image_positions = positions[visual].reshape(ids.shape[0], model.visual_token_len)
    position_ids = mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(~valid, 1)
    model._context(ids)
    projected = model._project(batch['visual_features'], ids.shape[0])
    embeddings = model.native.get_input_embeddings()(ids)
    fused = model.native.model.inputs_merger(input_ids=ids, inputs_embeds=embeddings,
                                            image_hidden_states=projected)
    return text, fused, mask, position_ids, last, image_positions


def _validate(result):
    if not torch.isfinite(result).all() or (result < 0).any() or (result.sum(-1) > 1 + 1e-6).any():
        raise ValueError('invalid visual attention probabilities')
    return result.float().cpu()


@torch.inference_mode()
def collect_last_query_attention(model, batch):
    """Run native SDPA prefill once and collect each layer's last-query visual mass."""
    text, fused, mask, position_ids, last, image_positions = _prepare(model, batch)
    rows = torch.arange(mask.shape[0], device=mask.device)
    collected, handles = [], []

    def capture(attn, args, kwargs):
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        batch_size, length, _ = hidden.shape
        # Project only the selected query; project all context keys, never V.
        query = attn.q_proj(hidden[rows, last]).view(batch_size, attn.num_heads, 1, attn.head_dim)
        keys = attn.k_proj(hidden).view(batch_size, length, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        embeddings = kwargs.get('position_embeddings')
        if embeddings is None:
            embeddings = attn.rotary_emb(keys, position_ids)
        cos, sin = embeddings
        qcos, qsin = cos[rows, last][:, None, None], sin[rows, last][:, None, None]
        query = query * qcos + rotate_half(query) * qsin
        keys = keys * cos[:, None] + rotate_half(keys) * sin[:, None]
        keys = repeat_kv(keys, attn.num_key_value_groups)
        scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(attn.head_dim)
        scores.masked_fill_(~mask[:, None, None].bool(), torch.finfo(scores.dtype).min)
        probs = scores.softmax(-1)[..., 0, :].mean(1)
        collected.append(probs.gather(1, image_positions).cpu())

    try:
        for layer in text.layers:
            handles.append(layer.self_attn.register_forward_pre_hook(capture, with_kwargs=True))
        text(inputs_embeds=fused, attention_mask=mask, position_ids=position_ids,
             use_cache=False, output_attentions=False, output_hidden_states=False)
    finally:
        for handle in handles:
            handle.remove()
    if len(collected) != len(text.layers):
        raise RuntimeError('attention hooks did not run exactly once per layer')
    return _validate(torch.stack(collected, dim=1))


@torch.inference_mode()
def reference_last_query_attention(model, batch):
    """Tiny-input oracle using native SDPA's output_attentions eager fallback."""
    text, fused, mask, position_ids, last, image_positions = _prepare(model, batch)
    output = text(inputs_embeds=fused, attention_mask=mask, position_ids=position_ids,
                  use_cache=False, output_attentions=True, output_hidden_states=False,
                  return_dict=True)
    rows = torch.arange(mask.shape[0], device=mask.device)
    result = [attention[rows, :, last, :].mean(1).gather(1, image_positions)
              for attention in output.attentions]
    return _validate(torch.stack(result, dim=1))
