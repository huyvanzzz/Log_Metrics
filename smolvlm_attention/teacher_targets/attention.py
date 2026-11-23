"""Last-prompt-query teacher targets; zero-based index 22 is layer 23.

Only prefix decoder blocks execute. At the target block we evaluate its input
normalization and Q/K projections, without V, FFN, later blocks, or LM head.
"""
import math

import torch
from transformers.models.llama.modeling_llama import rotate_half, repeat_kv

from smolvlm_attention.collector import _prepare
from smolvlm_inference.decoder import _mask


def _finite(name, value):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f'{name} contains NaN/Inf')


def normalized_visual_log_probs(logits, valid, image_positions):
    """All-key softmax, mean heads, then visual normalization, in log space.

    ``logits`` is [B,H,T], ``valid`` [B,T], positions [B,V]. FP64 is
    preserved for independent numeric tests; production inputs are FP32.
    """
    if logits.ndim != 3 or valid.shape != (logits.shape[0], logits.shape[2]):
        raise ValueError('logits/valid must be [B,H,T] and [B,T]')
    if logits.dtype not in (torch.float32, torch.float64):
        raise ValueError('teacher score computation requires FP32 or FP64')
    if image_positions.ndim != 2 or image_positions.shape[0] != logits.shape[0] or not image_positions.shape[1]:
        raise ValueError('image_positions must be nonempty [B,V]')
    if not bool(((valid == 0) | (valid == 1)).all()) or not bool(valid.bool().any(-1).all()):
        raise ValueError('each row requires binary valid keys')
    if bool((image_positions < 0).any()) or bool((image_positions >= logits.shape[-1]).any()):
        raise ValueError('visual positions outside context')
    if not bool(valid.bool().gather(1, image_positions).all()):
        raise ValueError('visual positions must be valid keys')
    _finite('attention logits', logits)
    masked = logits.masked_fill(~valid[:, None].bool(), -torch.inf)
    log_attention = masked - masked.logsumexp(-1, keepdim=True)
    visual = log_attention.gather(2, image_positions[:, None].expand(-1, logits.shape[1], -1))
    mean_log = visual.logsumexp(1) - math.log(logits.shape[1])
    result = mean_log - mean_log.logsumexp(-1, keepdim=True)
    _finite('teacher log probabilities', result)
    return result


@torch.inference_mode()
def collect_teacher_log_scores(model, batch, layer_index=22):
    """Return FP32 CPU [B,V] log probabilities using the target layer's Q/K."""
    layers = model.native.model.text_model.layers
    if type(layer_index) is not int or not 0 <= layer_index < len(layers):
        raise ValueError('layer_index must identify an existing zero-based decoder block')
    text, hidden, mask, positions, last, images = _prepare(model, batch)
    _finite('fused inputs', hidden)
    causal_mask = _mask(mask.bool(), hidden.shape[1], dtype=hidden.dtype)
    cache_positions = torch.arange(hidden.shape[1], device=hidden.device)
    rotary = text.rotary_emb(hidden, positions)
    for layer in layers[:layer_index]:
        hidden = layer(hidden, attention_mask=causal_mask, position_ids=positions,
                       past_key_value=None, use_cache=False, output_attentions=False,
                       cache_position=cache_positions, position_embeddings=rotary)[0]
    _finite('target layer input', hidden)
    target = layers[layer_index]
    normalized = target.input_layernorm(hidden)
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    attention = target.self_attn
    query = attention.q_proj(normalized[rows, last]).view(
        hidden.shape[0], attention.num_heads, 1, attention.head_dim)
    keys = attention.k_proj(normalized).view(
        hidden.shape[0], hidden.shape[1], attention.num_key_value_heads, attention.head_dim).transpose(1, 2)
    _finite('target Q', query)
    _finite('target K', keys)
    cos, sin = rotary
    query = query * cos[rows, last][:, None, None] + rotate_half(query) * sin[rows, last][:, None, None]
    keys = keys * cos[:, None] + rotate_half(keys) * sin[:, None]
    keys = repeat_kv(keys, attention.num_key_value_groups)
    logits = (query @ keys.transpose(-2, -1)).squeeze(2) / math.sqrt(attention.head_dim)
    # Last valid query sees all valid keys, including text and image tokens.
    result = normalized_visual_log_probs(logits, mask, images)
    if not bool(torch.allclose(result.logsumexp(-1), torch.zeros_like(result[:, 0]), atol=2e-5, rtol=0)):
        raise FloatingPointError('teacher log probabilities are not normalized')
    return result.float().cpu()
