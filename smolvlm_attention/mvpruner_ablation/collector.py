"""Measure question-to-image probabilities during native SmolVLM prefill."""

import math

import torch
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from smolvlm_attention.collector import _prepare
from smolvlm_inference.text import CachedBatchCollator, chat_messages


class QuestionBatch:
    def __init__(self, model):
        self.processor = model.processor
        self.tokenizer = self.processor.tokenizer
        self.order = model.prompt_camera_order
        self.visual_tokens = model.visual_token_len
        self.collator = CachedBatchCollator(
            self.processor, image_seq_len=model.geometry['visual_tokens_per_view'],
            max_length=model.native.config.text_config.max_position_embeddings,
            max_new_tokens=0, prompt_camera_order=self.order,
        )
        marker = 'MVPRUNER_QUESTION_SPAN_218754'
        prefix, suffix = self._render(marker).split(marker)
        self.prefix, self.suffix = prefix, suffix
        self.image_token = str(self.processor.image_token)
        self.image_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

    def _render(self, question):
        prompt = self.processor.apply_chat_template(
            chat_messages(question, prompt_camera_order=self.order),
            tokenize=False, add_generation_prompt=True,
        )
        return self.collator._expand(prompt)

    def encode(self, question):
        if self.image_token in question:
            raise ValueError('Question contains a reserved image placeholder')
        prompt = self._render(question)
        if prompt != self.prefix + question + self.suffix:
            raise ValueError('Chat template changed the question span')
        start, stop = len(self.prefix), len(self.prefix) + len(question)
        encoded = self.tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
        ids = encoded['input_ids']
        selected = [i for i, (left, right) in enumerate(encoded['offset_mapping'])
                    if right > left and right > start and left < stop]
        image_positions = [i for i, token in enumerate(ids) if token == self.image_id]
        if (len(ids) > self.collator.max_length or len(image_positions) != self.visual_tokens
                or not selected or selected[0] <= image_positions[-1]):
            raise ValueError('Prompt must fit context and place the question after all image tokens')
        return ids, selected

    def batch(self, encoded, features):
        width = max(len(ids) for ids, _ in encoded)
        pad = self.tokenizer.pad_token_id
        if pad is None:
            raise ValueError('Tokenizer must define a padding token')
        input_ids = torch.full((len(encoded), width), pad, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        question_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for row, (ids, selected) in enumerate(encoded):
            offset = width - len(ids)
            input_ids[row, offset:] = torch.tensor(ids)
            attention_mask[row, offset:] = 1
            question_mask[row, torch.tensor(selected) + offset] = True
        return dict(input_ids=input_ids, attention_mask=attention_mask,
                    question_mask=question_mask,
                    visual_features=features.unsqueeze(0).expand(len(encoded), -1, -1, -1))


def normalized_entropy(visual):
    """Entropy of positive-mass attention restricted to the last (image key) axis."""
    p = visual / visual.sum(dim=-1, keepdim=True)
    return -torch.special.xlogy(p, p).sum(-1) / math.log(visual.shape[-1])


def reduce_visual(visual, valid_queries):
    """Reduce [B,Q,V] without letting padded queries affect either statistic."""
    counts = valid_queries.sum(1)
    mean_visual = visual.masked_fill(~valid_queries[..., None], 0).sum(1) / counts[:, None]
    token_entropy = normalized_entropy(visual).masked_fill(~valid_queries, 0)
    return mean_visual, token_entropy.sum(1) / counts, normalized_entropy(mean_visual)


def _prepare_queries(model, batch):
    text, fused, mask, position_ids, _, images = _prepare(model, batch)
    question = batch['question_mask'].to(mask.device, dtype=torch.bool)
    if question.shape != mask.shape or (question & ~mask.bool()).any() or not question.any(1).all():
        raise ValueError('Question mask must select valid tokens in every sample')
    positions = torch.arange(mask.shape[1], device=mask.device)
    width = int(question.sum(1).max())
    queries = torch.where(question, positions, mask.shape[1]).sort(1).values[:, :width]
    valid = queries < mask.shape[1]
    # Reuse the first real query for padding so softmax always has valid keys.
    queries = torch.where(valid, queries, queries[:, :1])
    if (queries[:, 0] <= images[:, -1]).any():
        raise ValueError('Question queries must follow all image keys')
    allowed = mask[:, None, :].bool() & (positions <= queries[..., None])
    return text, fused, mask, position_ids, images, queries, valid, allowed


def _reduce_probs(probs, images, valid):
    # Softmax has already been computed across all causal keys for every head.
    visual = probs.mean(1).gather(-1, images[:, None].expand(-1, probs.shape[2], -1))
    return reduce_visual(visual, valid)


def _stack(results):
    stacked = [torch.stack(series, dim=1) for series in zip(*results)]
    visual, token, question = stacked
    if (not torch.isfinite(visual).all() or (visual < 0).any()
            or not torch.isfinite(token).all() or not torch.isfinite(question).all()
            or (token < -1e-6).any() or (token > 1 + 1e-6).any()
            or (question < -1e-6).any() or (question > 1 + 1e-6).any()):
        raise ValueError('Invalid visual attention or normalized entropy')
    return dict(visual=visual.cpu(), token_entropy=token.clamp(0, 1).cpu(),
                question_entropy=question.clamp(0, 1).cpu())


@torch.inference_mode()
def collect(model, batch):
    text, fused, mask, positions, images, queries, valid, allowed = _prepare_queries(model, batch)
    rows = torch.arange(mask.shape[0], device=mask.device)[:, None]
    results, handles = [], []

    def capture(attn, args, kwargs):
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        size, length, _ = hidden.shape
        q = attn.q_proj(hidden[rows, queries]).reshape(
            size, queries.shape[1], attn.num_heads, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(hidden).reshape(
            size, length, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        embeddings = kwargs.get('position_embeddings')
        cos, sin = embeddings if embeddings is not None else attn.rotary_emb(k, positions)
        q = q * cos[rows, queries][:, None] + rotate_half(q) * sin[rows, queries][:, None]
        k = k * cos[:, None] + rotate_half(k) * sin[:, None]
        k = repeat_kv(k, attn.num_key_value_groups)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(attn.head_dim)
        scores.masked_fill_(~allowed[:, None], torch.finfo(scores.dtype).min)
        results.append(_reduce_probs(scores.softmax(-1), images, valid))

    try:
        for layer in text.layers:
            handles.append(layer.self_attn.register_forward_pre_hook(capture, with_kwargs=True))
        text(inputs_embeds=fused, attention_mask=mask, position_ids=positions,
             use_cache=False, output_attentions=False, output_hidden_states=False)
    finally:
        for handle in handles:
            handle.remove()
    if len(results) != len(text.layers):
        raise RuntimeError('Expected one attention capture per decoder layer')
    return _stack(results)


@torch.inference_mode()
def reference(model, batch):
    """Native eager check for a small preflight, with unrestricted T-by-T buffers."""
    text, fused, mask, positions, images, queries, valid, _ = _prepare_queries(model, batch)
    output = text(inputs_embeds=fused, attention_mask=mask, position_ids=positions,
                  use_cache=False, output_attentions=True, output_hidden_states=False,
                  return_dict=True)
    results = []
    for attention in output.attentions:
        selected = attention.gather(2, queries[:, None, :, None].expand(
            -1, attention.shape[1], -1, attention.shape[-1]))
        results.append(_reduce_probs(selected, images, valid))
    return _stack(results)
