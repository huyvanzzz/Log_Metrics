"""Measure final-prompt attention and compact later layers during native generation."""

from contextlib import ExitStack
import math
from time import perf_counter

import torch
from torch.nn.functional import normalize
from transformers import DynamicCache, StoppingCriteria
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from smolvlm_attention.collector import _prepare


def _visual_attention(attn, kwargs, valid, last, visual):
    hidden = kwargs['hidden_states'][:len(valid)]
    batch, length, _ = hidden.shape
    rows = torch.arange(batch, device=hidden.device)
    query = attn.q_proj(hidden[rows, last]).view(batch, attn.num_heads, 1, attn.head_dim)
    keys = attn.k_proj(hidden).view(batch, length, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    cos, sin = (value[:batch].expand(batch, -1, -1) for value in kwargs['position_embeddings'])
    query = query * cos[rows, last][:, None, None] + rotate_half(query) * sin[rows, last][:, None, None]
    keys = keys * cos[:, None] + rotate_half(keys) * sin[:, None]
    scores = (query @ repeat_kv(keys, attn.num_key_value_groups).transpose(-2, -1)) / math.sqrt(attn.head_dim)
    scores = scores.masked_fill(~valid[:, None, None, :].bool(), -torch.inf)
    return scores.softmax(-1).squeeze(-2).mean(1).gather(1, visual)


@torch.inference_mode()
def attention_difference(model, batch, shallow, deep):
    """Return unpruned shallow-minus-deep attention in prompt visual order."""
    text, fused, mask, positions, last, visual = _prepare(model, batch)
    captured = []

    def capture(attn, args, kwargs):
        captured.append(_visual_attention(attn, kwargs, mask, last, visual))

    with ExitStack() as hooks:
        for number in (shallow, deep):
            handle = text.layers[number - 1].self_attn.register_forward_pre_hook(capture, with_kwargs=True)
            hooks.callback(handle.remove)
        text(inputs_embeds=fused, attention_mask=mask, position_ids=positions,
             use_cache=False, output_attentions=False, output_hidden_states=False)
    return (captured[0] - captured[1]).cpu().numpy()


def topk(scores, count):
    return scores.argsort(descending=True, stable=True)[:count]


def mmr(relevance, states, count, weight):
    """Select by relevance and maximum cosine similarity to earlier picks."""
    unit = normalize(states, dim=-1)
    similarity = unit @ unit.T
    selected = torch.zeros_like(relevance, dtype=torch.bool)
    redundancy = torch.full_like(relevance, -torch.inf)
    scores = relevance
    picks, penalties = [], []
    for step in range(count):
        pick = scores.masked_fill(selected, -torch.inf).argmax().view(1)
        picks.append(pick)
        penalties.append(redundancy.index_select(0, pick) if step else relevance.new_zeros(1))
        selected.scatter_(0, pick, True)
        redundancy = torch.maximum(redundancy, similarity.index_select(0, pick).squeeze(0))
        scores = weight * relevance - (1 - weight) * redundancy
    return torch.cat(picks), torch.cat(penalties)


class _PrunePrompt:
    def __init__(self, inputs, image_id, curve, count, method, weight):
        ids, self.valid = inputs['input_ids'], inputs['attention_mask']
        if ids.shape[0] != 1 or not bool(self.valid.all()):
            raise ValueError('Pruned generation requires one unpadded prompt per call')
        self.width = ids.shape[1]
        self.last = ids.new_tensor([self.width - 1])
        self.visual = (ids[0] == image_id).nonzero().flatten()
        if len(self.visual) != len(curve) or not 1 <= count <= len(curve):
            raise ValueError('Invalid visual positions, curve, or keep count')
        self.curve, self.count, self.method, self.weight = curve, count, method, weight
        self.raw = self.corrected = self.picks = self.penalties = self.kept = None
        self.prefill_mask = None

    def capture(self, attn, args, kwargs):
        if self.raw is None:
            # The pinned generation path duplicates one prompt across all beams.
            self.raw = _visual_attention(attn, kwargs, self.valid, self.last, self.visual[None])[0]
            self.corrected = self.raw - self.curve

    def prune(self, layer, args, outputs):
        if self.kept is not None:
            return
        hidden = outputs[0]
        if self.method == 'mmr':
            states = hidden[0].index_select(0, self.visual)
            self.picks, self.penalties = mmr(self.corrected, states, self.count, self.weight)
        else:
            relevance = self.raw if self.method == 'raw' else self.corrected
            self.picks = topk(relevance, self.count)
        keep = torch.ones(self.width, device=hidden.device, dtype=torch.bool)
        keep[self.visual] = False
        keep[self.visual.index_select(0, self.picks)] = True
        self.kept = keep.nonzero().flatten()
        return (hidden.index_select(1, self.kept), *outputs[1:])

    def adjust(self, layer, args, kwargs):
        # Lower layers retain full KV history. DynamicCache and HF beam reordering
        # allow later layers to store only compact history, with original RoPE IDs.
        if not isinstance(kwargs['past_key_value'], DynamicCache):
            raise RuntimeError('PosPrune requires native DynamicCache during beam search')
        mask = kwargs['attention_mask']
        positions = kwargs['cache_position']
        if positions.numel() == self.width:
            kwargs['cache_position'] = positions.index_select(0, self.kept)
            kwargs['position_ids'] = kwargs['position_ids'].index_select(1, self.kept)
            kwargs['position_embeddings'] = tuple(value.index_select(1, self.kept)
                                                   for value in kwargs['position_embeddings'])
            if mask is not None:
                if self.prefill_mask is None:
                    self.prefill_mask = mask.index_select(-2, self.kept).index_select(-1, self.kept)
                kwargs['attention_mask'] = self.prefill_mask
        elif mask is not None:
            key_indices = torch.cat((self.kept, torch.arange(self.width, mask.shape[-1], device=mask.device)))
            kwargs['attention_mask'] = mask.index_select(-1, key_indices)
        return args, kwargs

    def install(self, layers, shallow, hooks):
        layer = layers[shallow - 1]
        handles = [layer.self_attn.register_forward_pre_hook(self.capture, with_kwargs=True),
                   layer.register_forward_hook(self.prune)]
        handles.extend(layer.register_forward_pre_hook(self.adjust, with_kwargs=True) for layer in layers[shallow:])
        for handle in handles:
            hooks.callback(handle.remove)

    def details(self):
        return dict(picks=self.picks.tolist(), raw=self.raw.tolist(), corrected=self.corrected.tolist(),
                    penalties=None if self.penalties is None else self.penalties.tolist())


class _FirstBeamStep(StoppingCriteria):
    """Timestamp the first beam expansion without changing when generation stops."""

    def __init__(self, device):
        self.device = device
        self.first = self.never_stop = None

    def mark(self):
        if self.device.type == 'cuda':
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
            return event
        return perf_counter()

    def __call__(self, input_ids, scores, **kwargs):
        if self.first is None:
            self.first = self.mark()
            self.never_stop = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        return self.never_stop


@torch.inference_mode()
def generate(model, inputs, generation, shallow, *, method=None, curve=None, count=None, weight=0.9):
    """Run the stock cached generation path, optionally pruning after layer `shallow`."""
    if method not in (None, 'raw', 'corrected', 'mmr'):
        raise ValueError(f'Unknown pruning method {method!r}')
    device = inputs['input_ids'].device
    timer = _FirstBeamStep(device)
    state = None
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    start = timer.mark()
    with ExitStack() as hooks:
        if method is not None:
            state = _PrunePrompt(inputs, model.native.config.image_token_id, curve, count, method, weight)
            state.install(model.native.model.text_model.layers, shallow, hooks)
        tokens = model.generate_from_features(**inputs, **generation, stopping_criteria=[timer])
        end = timer.mark()
    if device.type == 'cuda':
        end.synchronize()
        elapsed = start.elapsed_time(end)
        first = None if timer.first is None else start.elapsed_time(timer.first)
    else:
        elapsed = (end - start) * 1000
        first = None if timer.first is None else (timer.first - start) * 1000
    return tokens, dict(latency_ms=elapsed, first_beam_step_ms=first,
                        generated_tokens=tokens.shape[1]), None if state is None else state.details()
