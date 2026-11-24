"""SmolVLM decoder interventions and greedy generation with layer-local KV lengths."""

import math

import torch
from transformers import DynamicCache, GenerationConfig
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from .design import RANK_LAYERS


MASS_EPS = 1e-8


def visual_weights(probs, scores, keep, mode, preserve_mass=False, recipient=None):
    """Edit visual columns of one post-softmax head/query row."""
    mass = probs.sum(-1, keepdim=True)
    if mode == 'merge':
        return torch.zeros_like(probs).scatter(-1, recipient, mass), torch.zeros_like(mass, dtype=torch.bool)
    masked = probs.masked_fill(~keep, 0)
    fallback = torch.zeros_like(mass, dtype=torch.bool)
    if preserve_mass:
        remaining = masked.sum(-1, keepdim=True)
        fallback = (remaining <= MASS_EPS) & (mass > 0)
        scaled = (masked / remaining.clamp_min(MASS_EPS)) * mass
        # Conditional softmax on the surviving visual logits stays stable even
        # when their original probabilities underflow. Zero visual mass stays zero.
        conditional = scores.masked_fill(~keep, -torch.inf).softmax(-1).nan_to_num()
        masked = torch.where(fallback, conditional * mass, scaled)
    return masked, fallback


def _keep_positions(mode, order, visual_count, device):
    if mode == 'mask_all':
        return torch.zeros((1, visual_count), dtype=torch.bool, device=device)
    ten = max(1, round(visual_count * 0.10))
    five = max(1, round(visual_count * 0.05))
    if mode in ('keep5', 'keep10'):
        count = five if mode == 'keep5' else ten
        return torch.zeros_like(order, dtype=torch.bool).scatter(1, order[:, :count], True)
    if mode == 'top5':
        indices = order[:, :five]
    else:
        indices = order[:, :ten] if mode == 'top10' else order[:, -ten:]
    return torch.ones_like(order, dtype=torch.bool).scatter(1, indices, False)


def _qkv(attn, hidden, embeddings, values=False):
    batch, length, _ = hidden.shape
    q = attn.q_proj(hidden).view(batch, length, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(hidden).view(batch, length, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, *embeddings)
    v = (attn.v_proj(hidden).view(batch, length, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
         if values else None)
    return q, k, v


def _scores(attn, query, key, allowed):
    return ((query @ repeat_kv(key, attn.num_key_value_groups).transpose(-2, -1))
            / math.sqrt(attn.head_dim)).masked_fill(~allowed, -torch.inf)


def _rank(attn, hidden, embeddings, allowed, question, visual_indices):
    q, k, _ = _qkv(attn, hidden, embeddings)
    batch = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
    width = int(question.sum(1).max())
    positions = torch.arange(hidden.shape[1], device=hidden.device)
    selected = torch.where(question, positions, hidden.shape[1]).sort(1).values[:, :width]
    valid = selected < hidden.shape[1]
    selected = torch.where(valid, selected, selected[:, :1])
    probs = _scores(attn, q.transpose(1, 2)[batch, selected].transpose(1, 2), k,
                    allowed[:, 0][batch, selected][:, None]).softmax(-1)
    index = visual_indices[:, None, :].expand(-1, width, -1)
    mean = probs.mean(1).gather(-1, index)
    return (mean * valid[..., None]).sum(1).argsort(dim=-1, descending=True, stable=True)


def _intervened_attention(attn, variant, visual_indices, affected, order, recipient, diagnostics):
    def forward(hidden_states, attention_mask, past_key_value, position_embeddings, **kwargs):
        q, k, v = _qkv(attn, hidden_states, position_embeddings, values=True)
        key, value = past_key_value.update(k, v, attn.layer_idx)
        scores = _scores(attn, q, key, attention_mask)
        probs = scores.softmax(-1).nan_to_num()
        index = visual_indices[:, None, None, :].expand(-1, q.shape[1], q.shape[2], -1)
        original = probs.gather(-1, index)
        if variant.mode == 'merge':
            keep = None
            sink = torch.full_like(index[..., :1], recipient)
            removed = original.sum(-1) - original.gather(-1, sink).squeeze(-1)
        else:
            keep = _keep_positions(variant.mode, order, index.shape[-1], q.device)[:, None, None]
            sink = None
            removed = original.masked_fill(keep, 0).sum(-1)
        replacement, fallback = visual_weights(original, scores.gather(-1, index), keep, variant.mode,
                                               variant.preserve_mass, sink)
        probs.scatter_(-1, index, torch.where(affected[:, None, :, None], replacement, original))
        mask = affected[:, None]
        diagnostics.append(torch.stack(((removed * mask).sum((1, 2)),
                                         (original.sum(-1) * mask).sum((1, 2)),
                                         (fallback.squeeze(-1) & mask).sum((1, 2)),
                                         affected.sum(1) * q.shape[1]), dim=-1))
        result = probs @ repeat_kv(value, attn.num_key_value_groups)
        result = result.transpose(1, 2).reshape_as(hidden_states)
        return attn.o_proj(result), None, past_key_value
    return forward


def _drop_images(tensor, keep):
    return tensor[keep].reshape(tensor.shape[0], -1, *tensor.shape[2:])


class Decoder:
    def __init__(self, model, encoder):
        self.model = model
        self.encoder = encoder
        self.text = model.native.model.text_model
        self.device = next(self.text.parameters()).device

    def _inputs(self, encoded, projected):
        width = max(len(ids) for ids, _ in encoded)
        input_ids = torch.full((len(encoded), width), self.encoder.tokenizer.pad_token_id, dtype=torch.long)
        valid = torch.zeros_like(input_ids, dtype=torch.bool)
        question = torch.zeros_like(valid)
        for row, (ids, query) in enumerate(encoded):
            start = width - len(ids)
            input_ids[row, start:] = torch.tensor(ids)
            valid[row, start:] = True
            question[row, torch.tensor(query) + start] = True
        input_ids, valid, question = (value.to(self.device) for value in (input_ids, valid, question))
        visual = input_ids.eq(self.encoder.image_id) & valid
        visual_indices = torch.arange(width, device=self.device).expand_as(input_ids)[visual].reshape(len(encoded), -1)
        native = self.model.native
        fused = native.model.inputs_merger(
            input_ids=input_ids, inputs_embeds=native.get_input_embeddings()(input_ids),
            image_hidden_states=projected.repeat(len(encoded), 1, 1),
        )
        positions = (valid.cumsum(1) - 1).masked_fill(~valid, 1)
        return input_ids, fused, valid, question, visual, visual_indices, positions

    def _layer(self, layer, hidden, valid_keys, positions, cache, *, prefill):
        allowed = valid_keys[:, None, None]
        if prefill:
            causal = torch.ones((hidden.shape[1], hidden.shape[1]), device=self.device, dtype=torch.bool).tril()
            allowed = allowed & causal
        return layer(hidden, attention_mask=allowed, position_ids=positions, past_key_value=cache,
                     position_embeddings=self.text.rotary_emb(hidden, positions), use_cache=True)[0]

    def _rank_layer(self, layer, hidden, valid, positions, cache, question, visual_indices):
        def capture(attn, _, kwargs):
            ranks.append(_rank(attn, kwargs['hidden_states'], kwargs['position_embeddings'],
                               kwargs['attention_mask'], question, visual_indices))

        ranks = []
        handle = layer.self_attn.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            hidden = self._layer(layer, hidden, valid, positions, cache, prefill=True)
        finally:
            handle.remove()
        return hidden, ranks[0]

    @torch.inference_mode()
    def rank_tokens(self, encoded, projected, rank_layers):
        """Collect clean prefill rankings without decoding an answer."""
        _, hidden, valid, question, _, visual_indices, positions = self._inputs(encoded, projected)
        ranks = {}
        for index, layer in enumerate(self.text.layers[:max(rank_layers)], 1):
            if index in rank_layers:
                hidden, order = self._rank_layer(layer, hidden, valid, positions, None, question, visual_indices)
                ranks[index] = order.cpu()
            else:
                hidden = self._layer(layer, hidden, valid, positions, None, prefill=True)
        return ranks

    @torch.inference_mode()
    def generate(self, encoded, projected, variant, limit, *, rankings=None, recipient=None, rank_layers=RANK_LAYERS):
        _, hidden, valid, question, visual, visual_indices, positions = self._inputs(encoded, projected)
        lengths = valid.sum(1, keepdim=True)
        self.model._context(hidden[..., 0], limit)
        compact, keys, diagnostics, ranks = ~visual, [], [], {}
        columns = torch.arange(hidden.shape[1], device=self.device)
        start = torch.where(question, columns, hidden.shape[1]).min(dim=1).values
        affected = valid & (columns >= start[:, None])
        cache = DynamicCache()

        for index, layer in enumerate(self.text.layers, 1):
            if variant.mode == 'discard' and index == variant.layers[0]:
                hidden, valid, positions = (_drop_images(value, compact) for value in (hidden, valid, positions))
            if variant.mode == 'skip' and index == variant.layers[0]:
                trimmed_valid = _drop_images(valid, compact)
                trimmed = self._layer(layer, _drop_images(hidden, compact), trimmed_valid,
                                      _drop_images(positions, compact), cache, prefill=True)
                hidden = hidden.clone()
                hidden[compact] = trimmed.reshape(-1, hidden.shape[-1])
                keys.append(trimmed_valid)
                continue
            keys.append(valid)
            if variant.mode == 'baseline' and index in rank_layers:
                hidden, ranks[index] = self._rank_layer(layer, hidden, valid, positions, cache, question, visual_indices)
            elif variant.mode in ('top5', 'top10', 'bottom10', 'keep5', 'keep10', 'mask_all', 'merge') and index in variant.layers:
                attention = layer.self_attn
                order = rankings[index] if rankings is not None else None
                attention.forward = _intervened_attention(attention, variant, visual_indices, affected,
                                                          order, recipient, diagnostics)
                try:
                    hidden = self._layer(layer, hidden, valid, positions, cache, prefill=True)
                finally:
                    del attention.forward
            else:
                hidden = self._layer(layer, hidden, valid, positions, cache, prefill=True)

        head = self.model.native.lm_head
        logits = head(self.text.norm(hidden[:, -1:]))[:, 0]
        eos = self.model.native.generation_config.eos_token_id
        eos_ids = torch.tensor(eos if isinstance(eos, list) else [eos], device=self.device)
        finished = torch.zeros(hidden.shape[0], device=self.device, dtype=torch.bool)
        generated = []
        for step in range(limit):
            tokens = logits.argmax(1)
            tokens = torch.where(finished, self.encoder.tokenizer.pad_token_id, tokens)
            generated.append(tokens)
            finished |= torch.isin(tokens, eos_ids)
            if finished.all() or step + 1 == limit:
                break
            hidden = self.model.native.get_input_embeddings()(tokens[:, None])
            for layer_index, layer in enumerate(self.text.layers):
                keys[layer_index] = torch.cat((keys[layer_index], torch.ones_like(finished[:, None])), dim=1)
                hidden = self._layer(layer, hidden, keys[layer_index], lengths + step, cache, prefill=False)
            logits = head(self.text.norm(hidden))[:, 0]

        tokens = torch.stack(generated, 1).cpu()
        ranks = {layer: value.cpu() for layer, value in ranks.items()}
        mass = torch.stack(diagnostics).sum(0).cpu() if diagnostics else torch.zeros((len(encoded), 4))
        truncated = ~finished.cpu()
        return tokens, ranks, mass, truncated

    @torch.inference_mode()
    def verify_native(self, encoded, projected, baseline):
        ids, fused, valid, *_ = self._inputs(encoded, projected)
        native = self.model.native
        previous = native.generation_config
        settings = GenerationConfig(bos_token_id=previous.bos_token_id, eos_token_id=previous.eos_token_id,
                                    pad_token_id=previous.pad_token_id, max_new_tokens=3,
                                    do_sample=False, num_beams=1, use_cache=True)
        expected = native.generate(input_ids=ids, inputs_embeds=fused, attention_mask=valid.long(),
                                   generation_config=settings)[:, ids.shape[1]:].cpu()
        actual, *_ = self.generate(encoded, projected, baseline, 3, rank_layers=())
        torch.testing.assert_close(actual, expected, msg='Baseline greedy tokens differ from the native decoder')
