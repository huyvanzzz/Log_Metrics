"""Training-free prefill pruning with native layers and layer-local KV masks."""
from contextlib import contextmanager

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from .pruning import stage1_indices, stage2_indices


def _finite(name, value):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f'{name} contains NaN/Inf')


def _mask(valid, query_count, past=0, dtype=torch.float32):
    """Physical causal order is independent of the retained original RoPE IDs."""
    keys = valid.shape[1]
    causal = torch.arange(keys, device=valid.device)[None, :] <= (
        past + torch.arange(query_count, device=valid.device)[:, None])
    allowed = causal[None, None] & valid[:, None, None, :]
    # Native SDPA also unmasks fully masked padded query rows. Their outputs
    # are never used as valid keys or selected for generation.
    allowed = allowed | ~allowed.any(-1, keepdim=True)
    return torch.zeros(allowed.shape, device=valid.device, dtype=dtype).masked_fill(
        ~allowed, torch.finfo(dtype).min)


def _positions(mask):
    counts = mask.sum(1)
    if not bool((counts == counts[0]).all()):
        raise ValueError('Every sample must have the same visual budget')
    return mask.nonzero(as_tuple=False)[:, 1].reshape(mask.shape[0], int(counts[0]))


def _gather(value, indices):
    if value.ndim == 3:
        return value.gather(1, indices[..., None].expand(-1, -1, value.shape[-1]))
    return value.gather(1, indices)


def _pack(state, keep):
    """Stable source-order gather followed by left padding, for ragged text."""
    keep = keep & state['valid']
    counts = keep.sum(1)
    width = int(counts.max())
    if not bool((counts > 0).all()):
        raise ValueError('Cannot prune every token of a sample')
    slots = torch.arange(width, device=keep.device)[None, :]
    valid = slots >= width - counts[:, None]
    index = torch.zeros_like(valid, dtype=torch.long)
    index[valid] = keep.nonzero(as_tuple=False)[:, 1]
    result = {key: _gather(value, index) for key, value in state.items()}
    result['valid'] = valid
    result['question'] &= valid
    result['image'] = result['image'].masked_fill(~valid, -1)
    result['camera'] = result['camera'].masked_fill(~valid, -1)
    result['pos'] = result['pos'].masked_fill(~valid, 0)
    result['hidden'] = result['hidden'].masked_fill(~valid[..., None], 0)
    return result


def _retain_images(state, image_ids, total):
    selected = torch.zeros(state['hidden'].shape[0], total, dtype=torch.bool,
                           device=state['hidden'].device).scatter_(1, image_ids, True)
    keep = (state['image'] < 0) | selected.gather(1, state['image'].clamp_min(0))
    return _pack(state, keep)


def question_attention_scores(q_projection, k_projection, attention, rotary,
                              attention_mask, question_mask, image_positions,
                              chunk_size=16):
    """Exact question rows of native pre-dropout attention, without full S×S A."""
    batch, size, _ = q_projection.shape
    heads = attention.config.num_attention_heads
    kv_heads = attention.config.num_key_value_heads
    dim = attention.head_dim
    q = q_projection.view(batch, size, heads, dim).transpose(1, 2)
    k = k_projection.view(batch, size, kv_heads, dim).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, *rotary)
    k = repeat_kv(k, heads // kv_heads)
    counts = question_mask.sum(1)
    if not bool((counts > 0).all()):
        raise ValueError('Question attention needs nonempty question spans')
    width = int(counts.max())
    q_indices = torch.zeros(batch, width, dtype=torch.long, device=q.device)
    q_valid = torch.arange(width, device=q.device)[None, :] < counts[:, None]
    q_indices[q_valid] = question_mask.nonzero(as_tuple=False)[:, 1]
    scores = q.new_zeros(batch, size)
    for start in range(0, width, chunk_size):
        index = q_indices[:, start:start + chunk_size]
        query = q.gather(2, index[:, None, :, None].expand(-1, heads, -1, dim))
        mask = attention_mask.gather(2, index[:, None, :, None].expand(-1, 1, -1, size))
        logits = torch.matmul(query, k.transpose(-1, -2)) / dim ** 0.5
        probabilities = torch.softmax(logits + mask, dim=-1)
        probabilities *= q_valid[:, None, start:start + chunk_size, None]
        scores += probabilities.sum((1, 2))
    scores /= (counts * heads)[:, None]
    result = scores.gather(1, image_positions)
    _finite('Stage 2 attention scores', result)
    return result


@contextmanager
def _capture_qk(attention):
    projections = {}
    hooks = [attention.q_proj.register_forward_hook(lambda m, a, y: projections.update(q=y)),
             attention.k_proj.register_forward_hook(lambda m, a, y: projections.update(k=y))]
    try:
        yield projections
    finally:
        for hook in hooks:
            hook.remove()


def _json(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _json(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    return value


class PrunedGreedyDecoder:
    def __init__(self, model, stage2_layer_index, stage1_keep=196, stage2_keep=32):
        self.model = model
        self.text = model.native.model.text_model
        if type(stage2_layer_index) is not int or not 0 <= stage2_layer_index < len(self.text.layers):
            raise ValueError('Invalid Stage 2 decoder block index')
        total = model.visual_token_len
        if not 6 <= stage2_keep <= stage1_keep <= total:
            raise ValueError('Invalid pruning budgets')
        if getattr(self.text.config, 'pretraining_tp', 1) != 1:
            raise ValueError('Scoring hooks require native pretraining_tp=1')
        self.index = stage2_layer_index
        self.keep1, self.keep2 = stage1_keep, stage2_keep

    def _inputs(self, batch):
        ids = batch['input_ids']
        valid = batch['attention_mask'].bool()
        visual = self.model._project(batch['visual_features'], ids.shape[0])
        visual = visual.reshape(ids.shape[0], self.model.visual_token_len, -1)
        embeddings = self.model.native.get_input_embeddings()(ids)
        hidden = self.model.native.model.inputs_merger(
            input_ids=ids, inputs_embeds=embeddings,
            image_hidden_states=visual.flatten(0, 1).reshape(-1, self.model.geometry['visual_tokens_per_view'], visual.shape[-1]))
        images = batch['image_positions']
        camera = torch.full_like(ids, -1).scatter_(1, images, batch['image_camera_ids'])
        image_ids = torch.full_like(ids, -1).scatter_(1, images,
            torch.arange(images.shape[1], device=ids.device)[None, :].expand(ids.shape[0], -1))
        selected, _, first = stage1_indices(visual, embeddings, batch['question_mask'], keep=self.keep1)
        state = dict(hidden=hidden, valid=valid, pos=batch['source_position_ids'],
                     question=batch['question_mask'].bool(), camera=camera, image=image_ids)
        return state, selected, first

    def _prefill(self, source, selected1, cache, frozen2=None):
        state = _retain_images(source, selected1, self.model.visual_token_len)
        prefixes, counts, cache_lengths = [], [], []
        second, selected2, scores = {}, frozen2, None
        states_finite = torch.ones((), dtype=torch.bool, device=state['hidden'].device)
        for i, layer in enumerate(self.text.layers):
            valid, hidden = state['valid'], state['hidden']
            counts.append(int((state['image'] >= 0).sum(1)[0]))
            prefixes.append(valid.clone())
            mask = _mask(valid, hidden.shape[1], dtype=hidden.dtype)
            rotary = self.text.rotary_emb(hidden, state['pos'])
            kwargs = dict(attention_mask=mask, position_ids=state['pos'],
                past_key_value=cache, use_cache=cache is not None,
                cache_position=torch.arange(hidden.shape[1], device=hidden.device),
                position_embeddings=rotary)
            if i == self.index and frozen2 is None:
                with _capture_qk(layer.self_attn) as projections:
                    state['hidden'] = layer(hidden, **kwargs)[0]
                visual_positions = _positions(state['image'] >= 0)
                scores = question_attention_scores(projections['q'], projections['k'],
                    layer.self_attn, rotary, mask, state['question'], visual_positions)
                local, _, second = stage2_indices(scores, _gather(state['camera'], visual_positions), keep=self.keep2)
                selected2 = _gather(_gather(state['image'], visual_positions), local)
                del projections
            else:
                state['hidden'] = layer(hidden, **kwargs)[0]
            if self.check_states:
                states_finite &= torch.isfinite(state['hidden']).all()
            cache_lengths.append(hidden.shape[1])
            if i == self.index:
                state = _retain_images(state, selected2, self.model.visual_token_len)
        logits = self.model.native.lm_head(self.text.norm(state['hidden'][:, -1:, :]))[:, 0].float()
        if self.check_states and not bool(states_finite):
            raise FloatingPointError('Prefill hidden states contain NaN/Inf')
        _finite('Prefill logits', logits)
        debug = dict(stage2_layer_index=self.index, visual_counts_by_layer=counts,
                     prefill_cache_lengths=cache_lengths, stage2=second,
                     stage2_source_indices=selected2,
                     stage2_score_range=[scores.min(), scores.max()] if scores is not None else None)
        return logits, prefixes, selected2, debug

    def _step(self, token, active, positions, cache, prefixes):
        hidden = self.model.native.get_input_embeddings()(token[:, None])
        states_finite = torch.ones((), dtype=torch.bool, device=hidden.device)
        for i, layer in enumerate(self.text.layers):
            past = cache.get_seq_length(i)
            prefixes[i] = torch.cat((prefixes[i], active[:, None]), 1)
            mask = _mask(prefixes[i], 1, past, hidden.dtype)
            rotary = self.text.rotary_emb(hidden, positions[:, None])
            hidden = layer(hidden, attention_mask=mask, position_ids=positions[:, None],
                past_key_value=cache, use_cache=True,
                cache_position=torch.arange(past, past + 1, device=hidden.device),
                position_embeddings=rotary)[0]
            if self.check_states:
                states_finite &= torch.isfinite(hidden).all()
        logits = self.model.native.lm_head(self.text.norm(hidden))[:, 0].float()
        if self.check_states and not bool(states_finite):
            raise FloatingPointError('Decode hidden states contain NaN/Inf')
        _finite('Decode logits', logits)
        return logits

    @torch.inference_mode()
    def generate(self, batch, max_new_tokens=512, ignore_eos=False, debug=False):
        return self._generate(batch, max_new_tokens, ignore_eos, debug, reference=False)

    @torch.inference_mode()
    def generate_reference(self, batch, max_new_tokens=512):
        """Slow test oracle: recompute history with prefill selections frozen."""
        return self._generate(batch, max_new_tokens, False, False, reference=True)

    def _generate(self, batch, max_new_tokens, ignore_eos, debug, reference):
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError('max_new_tokens must be positive')
        self.check_states = debug or reference
        source, selected1, first = self._inputs(batch)
        original_lengths = source['valid'].sum(1)
        if int(original_lengths.max()) + max_new_tokens > self.text.config.max_position_embeddings:
            raise ValueError('Original prompt plus generation exceeds native context budget')
        cache = None if reference else DynamicCache()
        logits, prefixes, selected2, info = self._prefill(source, selected1, cache)
        batch_size = logits.shape[0]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=logits.device)
        lengths = torch.full((batch_size,), max_new_tokens, dtype=torch.long, device=logits.device)
        config = self.model.native.generation_config
        eos = config.eos_token_id
        eos = [] if eos is None else eos if isinstance(eos, (list, tuple)) else [eos]
        eos = torch.tensor(eos, device=logits.device)
        if config.pad_token_id is None:
            raise ValueError('Native generation requires PAD token ID')
        output = []
        for step in range(max_new_tokens):
            if step:
                active = ~finished
                positions = original_lengths + step - 1
                if reference:
                    token_hidden = self.model.native.get_input_embeddings()(output[-1][:, None])
                    source = dict(hidden=torch.cat((source['hidden'], token_hidden), 1),
                        valid=torch.cat((source['valid'], active[:, None]), 1),
                        pos=torch.cat((source['pos'], positions[:, None]), 1),
                        question=torch.cat((source['question'], torch.zeros_like(active[:, None])), 1),
                        camera=torch.cat((source['camera'], torch.full_like(positions[:, None], -1)), 1),
                        image=torch.cat((source['image'], torch.full_like(positions[:, None], -1)), 1))
                    logits, _, _, _ = self._prefill(source, selected1, None, frozen2=selected2)
                else:
                    logits = self._step(output[-1], active, positions, cache, prefixes)
            token = logits.argmax(-1).masked_fill(finished, config.pad_token_id)
            output.append(token)
            if not ignore_eos:
                ended = (token[:, None] == eos[None, :]).any(1) & ~finished
                lengths = torch.where(ended, step + 1, lengths)
                finished |= ended
                if bool(finished.all()):
                    break
        if debug:
            info.update(stage1=first, stage1_source_indices=selected1,
                        selection_calls={'stage1': 1, 'stage2': 1},
                        final_cache_lengths=[cache.get_seq_length(i) for i in range(len(self.text.layers))])
        return dict(generated_ids=torch.stack(output, 1).cpu(), lengths=lengths.cpu().tolist(),
                    debug=_json(info) if debug else {})
