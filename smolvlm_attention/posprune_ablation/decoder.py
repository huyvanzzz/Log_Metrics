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
    return scores.argsort(descending=True, stable=True)[..., :count]


def mmr(relevance, states, count, weight):
    """Select by relevance and maximum cosine similarity to earlier picks."""
    single = relevance.ndim == 1
    if single:
        relevance, states = relevance[None], states[None]
    unit = normalize(states, dim=-1)
    similarity = unit @ unit.transpose(1, 2)
    selected = torch.zeros_like(relevance, dtype=torch.bool)
    redundancy = torch.full_like(relevance, -torch.inf)
    scores = relevance
    picks, penalties = [], []
    for step in range(count):
        pick = scores.masked_fill(selected, -torch.inf).argmax(dim=1, keepdim=True)
        picks.append(pick)
        penalties.append(redundancy.gather(1, pick) if step else relevance.new_zeros(pick.shape))
        selected.scatter_(1, pick, True)
        candidate = similarity.gather(1, pick[..., None].expand(-1, -1, relevance.shape[1])).squeeze(1)
        redundancy = torch.maximum(redundancy, candidate)
        scores = weight * relevance - (1 - weight) * redundancy
    indices, penalties = torch.cat(picks, dim=1), torch.cat(penalties, dim=1)
    return (indices[0], penalties[0]) if single else (indices, penalties)


def _gather_rows(tensor, positions):
    """Gather sequence columns independently for each batch row."""
    tensor = tensor.expand(len(positions), *tensor.shape[1:])
    shape = (*positions.shape, *([1] * (tensor.ndim - 2)))
    return tensor.gather(1, positions.reshape(shape).expand(*positions.shape, *tensor.shape[2:]))


def _gather_mask_keys(mask, positions):
    return mask.gather(-1, positions[:, None, None, :].expand(-1, *mask.shape[1:3], -1))


class _PrunePrompt:
    def __init__(self, inputs, image_id, curve, count, method, weight):
        ids, mask = inputs['input_ids'], inputs['attention_mask']
        self.valid = mask.bool()
        if not bool(self.valid[:, -1].all()) or bool((self.valid[:, :-1] & ~self.valid[:, 1:]).any()):
            raise ValueError('Pruned generation requires nonempty left-padded prompts')
        self.width = ids.shape[1]
        self.last = ids.new_full((len(ids),), self.width - 1)
        visual = (ids == image_id) & self.valid
        if not 1 <= count <= len(curve) or not bool((visual.sum(1) == len(curve)).all()):
            raise ValueError('Invalid visual positions, curve, or keep count')
        self.visual = visual.nonzero()[:, 1].view(len(ids), len(curve))
        self.curve, self.count, self.method, self.weight = curve, count, method, weight
        self.raw = self.corrected = self.picks = self.penalties = self.kept = None
        self.prefill_mask = None

    def capture(self, attn, args, kwargs):
        if self.raw is None:
            # If beam search is used, every beam starts from the same prompt.
            self.raw = _visual_attention(attn, kwargs, self.valid, self.last, self.visual)
            self.corrected = self.raw - self.curve

    def prune(self, layer, args, outputs):
        if self.kept is not None:
            return
        hidden = outputs[0]
        if self.method == 'mmr':
            states = _gather_rows(hidden[:len(self.visual)], self.visual)
            self.picks, self.penalties = mmr(self.corrected, states, self.count, self.weight)
        else:
            relevance = self.raw if self.method == 'raw' else self.corrected
            self.picks = topk(relevance, self.count)
        keep = torch.ones_like(self.valid)
        keep.scatter_(1, self.visual, False)
        keep.scatter_(1, self.visual.gather(1, self.picks), True)
        self.kept = keep.nonzero()[:, 1].view(len(self.visual), -1).expand(len(hidden), -1)
        return (_gather_rows(hidden, self.kept), *outputs[1:])

    def adjust(self, layer, args, kwargs):
        # Lower layers retain full KV history. DynamicCache and HF beam reordering
        # allow later layers to store only compact history, with original RoPE IDs.
        if not isinstance(kwargs['past_key_value'], DynamicCache):
            raise RuntimeError('PosPrune requires native DynamicCache during generation')
        mask = kwargs['attention_mask']
        if mask is not None:
            mask = mask.expand(len(self.kept), *mask.shape[1:])
        positions = kwargs['cache_position']
        if positions.numel() == self.width:
            # DynamicCache appends by physical length. Original per-row positions
            # still drive RoPE and the gathered attention mask.
            kwargs['cache_position'] = positions[:self.kept.shape[1]]
            kwargs['position_ids'] = _gather_rows(kwargs['position_ids'], self.kept)
            kwargs['position_embeddings'] = tuple(_gather_rows(value, self.kept)
                                                   for value in kwargs['position_embeddings'])
            if mask is not None:
                if self.prefill_mask is None:
                    queries = self.kept[:, None, :, None].expand(-1, mask.shape[1], -1, mask.shape[-1])
                    self.prefill_mask = _gather_mask_keys(mask.gather(-2, queries), self.kept)
                kwargs['attention_mask'] = self.prefill_mask
        elif mask is not None:
            generated = torch.arange(self.width, mask.shape[-1], device=mask.device).expand(len(self.kept), -1)
            kwargs['attention_mask'] = _gather_mask_keys(mask, torch.cat((self.kept, generated), dim=1))
        return args, kwargs

    def install(self, layers, shallow, hooks):
        layer = layers[shallow - 1]
        handles = [layer.self_attn.register_forward_pre_hook(self.capture, with_kwargs=True),
                   layer.register_forward_hook(self.prune)]
        handles.extend(layer.register_forward_pre_hook(self.adjust, with_kwargs=True) for layer in layers[shallow:])
        for handle in handles:
            hooks.callback(handle.remove)

    def details(self):
        penalties = [None] * len(self.picks) if self.penalties is None else self.penalties.tolist()
        return [dict(picks=picks, raw=raw, corrected=corrected, penalties=penalty)
                for picks, raw, corrected, penalty in
                zip(self.picks.tolist(), self.raw.tolist(), self.corrected.tolist(), penalties)]


class _FirstTokenStep(StoppingCriteria):
    """Timestamp the first token selection without changing when generation stops."""

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
    if method is not None and len(inputs['input_ids']) > 1:
        config = generation.get('generation_config', model.native.generation_config)
        if generation.get('num_beams', config.num_beams) != 1 or generation.get('do_sample', config.do_sample):
            raise ValueError('Batched pruning supports greedy generation')
    device = inputs['input_ids'].device
    timer = _FirstTokenStep(device)
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
    return tokens, dict(latency_ms=elapsed, first_token_step_ms=first), None if state is None else state.details()
