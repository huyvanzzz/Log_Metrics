"""Measure visual-token pair similarity and test single-token removal."""

from contextlib import ExitStack
import math

import numpy as np
import torch
from torch.nn.functional import normalize
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from smolvlm_attention.collector import _prepare
from .decoder import _PrunePrompt, _gather_rows
from .runtime import GENERATION


LAYERS = (3, 15, 22, 26)
DROP_LAYER = 22
QUERY_LIMIT = 16
GROUPS = ('high_hidden_high_attention', 'high_hidden_low_attention',
          'low_hidden_high_attention', 'low_hidden_low_attention')


def _incoming(attn, kwargs, mask, queries, query_valid, visual):
    hidden = kwargs['hidden_states']
    size, width, _ = hidden.shape
    q = attn.q_proj(_gather_rows(hidden, queries))
    q = q.view(size, -1, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(hidden).view(size, width, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    cos, sin = (value.expand(size, -1, -1) for value in kwargs['position_embeddings'])
    q = q * _gather_rows(cos, queries)[:, None] + rotate_half(q) * _gather_rows(sin, queries)[:, None]
    k = k * cos[:, None] + rotate_half(k) * sin[:, None]
    scores = q @ repeat_kv(k, attn.num_key_value_groups).transpose(-2, -1) / math.sqrt(attn.head_dim)
    causal = torch.arange(width, device=hidden.device) <= queries[..., None]
    scores.masked_fill_(~(causal & mask[:, None].bool())[:, None], -torch.inf)
    probs = scores.softmax(-1)
    columns = visual[:, None, None, :].expand(-1, attn.num_heads, queries.shape[1], -1)
    incoming = probs.gather(-1, columns) * query_valid[:, None, :, None]
    return incoming.flatten(1, 2).transpose(1, 2), incoming[:, :, -1, :].mean(1)


@torch.inference_mode()
def measure_pairs(model, inputs, layers=LAYERS, query_limit=QUERY_LIMIT):
    """Return upper-triangle cosines for both representations, in common pair order."""
    text, fused, mask, positions, last, visual = _prepare(model, inputs)
    if max(layers) > len(text.layers) or query_limit < 1 or not bool((last > visual[:, -1]).all()):
        raise ValueError('Requested layers and post-image text queries must exist')
    queries = last[:, None] - torch.arange(query_limit - 1, -1, -1, device=last.device)
    query_valid = queries > visual[:, -1:]
    queries = torch.where(query_valid, queries, last[:, None])
    first, second = torch.triu_indices(visual.shape[1], visual.shape[1], offset=1, device=last.device)
    pairs, patterns, masses = {}, {}, {}

    def capture_attention(attn, args, kwargs):
        patterns[attn.layer_idx + 1], masses[attn.layer_idx + 1] = _incoming(
            attn, kwargs, mask, queries, query_valid, visual)

    def capture_states(number):
        def capture(layer, args, output):
            states = normalize(_gather_rows(output[0], visual), dim=-1)
            attention = normalize(patterns.pop(number), dim=-1)
            vectors = [states, attention]
            pairs[number] = torch.stack([(v @ v.transpose(-2, -1))[:, first, second] for v in vectors], dim=1)
        return capture

    with ExitStack() as hooks:
        for number in layers:
            layer = text.layers[number - 1]
            hooks.callback(layer.self_attn.register_forward_pre_hook(capture_attention, with_kwargs=True).remove)
            hooks.callback(layer.register_forward_hook(capture_states(number)).remove)
        text(inputs_embeds=fused, attention_mask=mask, position_ids=positions, use_cache=False)
    return {number: (pairs[number].cpu().numpy(), masses[number].cpu().numpy()) for number in layers}


def _correlation(x, y):
    x, y = x - x.mean(dtype=np.float64), y - y.mean(dtype=np.float64)
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.clip(np.dot(x, y) / norm, -1, 1)) if norm else None


def _ranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def correlations(hidden, attention):
    return dict(pearson=_correlation(hidden, attention), spearman=_correlation(_ranks(hidden), _ranks(attention)))


def choose_pairs(similarities, mass, seed):
    """Sample four strata with distinct dropped tokens near a common attention level."""
    hidden, attention = similarities
    low_h, high_h = np.quantile(hidden, [.25, .75])
    low_a, high_a = np.quantile(attention, [.25, .75])
    if low_h >= high_h or low_a >= high_a:
        return {}
    first, second = np.triu_indices(len(mass), k=1)
    rng = np.random.default_rng(seed)
    target = float(np.quantile(mass, rng.uniform(.25, .75)))
    chosen, removed = {}, set()
    for group in rng.permutation(GROUPS):
        allowed_h = hidden >= high_h if group.startswith('high_hidden') else hidden <= low_h
        allowed_a = attention >= high_a if group.endswith('high_attention') else attention <= low_a
        candidates = np.flatnonzero(allowed_h & allowed_a)
        pairs = np.tile(candidates, 2)
        drop = np.concatenate((first[candidates], second[candidates]))
        keep = np.concatenate((second[candidates], first[candidates]))
        available = np.flatnonzero(~np.isin(drop, list(removed)))
        if not len(available):
            return {}
        closest = available[np.argsort(np.abs(mass[drop[available]] - target), kind='stable')[:32]]
        index = rng.choice(closest)
        pair, dropped, kept = int(pairs[index]), int(drop[index]), int(keep[index])
        removed.add(dropped)
        chosen[str(group)] = dict(dropped=dropped, kept=kept, hidden_similarity=float(hidden[pair]),
            attention_similarity=float(attention[pair]), dropped_raw_attention=float(mass[dropped]), target_attention=target)
    return chosen


class _RemoveOne(_PrunePrompt):
    def __init__(self, model, inputs, drop):
        visual_count = model.visual_token_len
        super().__init__(inputs, model.native.config.image_token_id, visual_count, None,
                         visual_count - 1, 'raw', .9, model.geometry['visual_tokens_per_view'])
        self.drop = drop

    def capture(self, attn, args, kwargs):
        pass

    def prune(self, layer, args, outputs):
        if self.kept is not None:
            return
        keep = torch.ones_like(self.valid)
        keep.scatter_(1, self.visual.gather(1, self.drop), False)
        self.kept = keep.nonzero()[:, 1].view(len(keep), -1)
        return (_gather_rows(outputs[0], self.kept), *outputs[1:])


@torch.inference_mode()
def generate_pair_outputs(model, inputs, drop=None, *, layer=DROP_LAYER, generation=None):
    first = []

    def remember(input_ids, scores):
        if not first:
            first.append(scores.detach().softmax(dim=-1).cpu())
        return scores

    with ExitStack() as hooks:
        if drop is not None:
            state = _RemoveOne(model, inputs, drop)
            state.install(model.native.model.text_model.layers, layer, hooks)
        tokens = model.generate_from_features(**inputs, **(GENERATION if generation is None else generation),
                                              logits_processor=[remember])
    return tokens.cpu(), first[0]


def js_divergence(p, q):
    p, q = p.double(), q.double()
    midpoint = (p + q) / 2
    def relative(x):
        return (torch.special.xlogy(x, x) - torch.special.xlogy(x, midpoint)).sum(-1)
    return ((relative(p) + relative(q)) / 2).clamp_min(0)
