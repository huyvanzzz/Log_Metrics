"""Frozen native fusion and exactly three complete decoder blocks.

run_prefix returns raw hidden states, validity and original RoPE positions, image
columns, last valid physical query columns, original lengths, layer masks/widths.
No final normalization or suffix layer is applied.
"""
import torch
from smolvlm_topk_deferred.generation import _mask
from smolvlm_map_predictor.model import visual_features, gather_questions


@torch.no_grad()
def fuse_inputs(backbone, batch):
    backbone.eval()
    if any(p.requires_grad for p in backbone.parameters()):
        raise ValueError('native backbone must be frozen')
    device = next(backbone.parameters()).device
    ids, valid = batch['input_ids'].to(device), batch['attention_mask'].to(device)
    if not valid.bool().any(1).all():
        raise ValueError('empty native prompt')
    visual = visual_features(backbone, batch, device).reshape(ids.shape[0] * backbone.geometry['num_views'], -1, backbone.geometry['text_dim'])
    embedded = backbone.native.model.inputs_merger(input_ids=ids,
        inputs_embeds=backbone.native.get_input_embeddings()(ids), image_hidden_states=visual)
    pos = batch.get('position_ids')
    pos = ((valid.long().cumsum(-1)-1).masked_fill(~valid.bool(), 0) if pos is None
           else pos.to(device=device, non_blocking=True))
    if 'projected_visual_features' in batch and 'image_positions' in batch:
        image = batch['image_positions'].to(device=device, non_blocking=True)
    else:
        image_mask = (ids == backbone.native.config.image_token_id) & valid.bool()
        if not (image_mask.sum(1) == backbone.visual_token_len).all():
            raise ValueError('unexpected visual token count')
        image = image_mask.nonzero()[:, 1].reshape(ids.shape[0], backbone.visual_token_len)
    if image.shape != (ids.shape[0], backbone.visual_token_len):
        raise ValueError('unexpected visual token count')
    return dict(input_ids=ids, inputs_embeds=embedded, attention_mask=valid,
                position_ids=pos, image_positions=image,
                last_query_columns=batch.get('last_query_columns'),
                original_lengths=valid.sum(-1).long(), pad_token_id=backbone.processor.tokenizer.pad_token_id)


@torch.no_grad()
def run_prefix(backbone, fused, cache=None):
    backbone.eval()
    if any(p.requires_grad for p in backbone.parameters()):
        raise ValueError('native backbone must be frozen')
    text = backbone.native.model.text_model
    if len(text.layers) < 3:
        raise ValueError('three decoder blocks required')
    if cache is not None and any(cache.get_seq_length(i) for i in range(3)):
        raise ValueError('prefix requires an empty cache')
    hidden,valid,pos = fused['inputs_embeds'],fused['attention_mask'].bool(),fused['position_ids']
    masks,lengths=[],[]
    causal_mask = _mask(valid, hidden.shape[1], dtype=hidden.dtype)
    rotary = text.rotary_emb(hidden, pos)
    slots = torch.arange(hidden.shape[1], device=hidden.device)
    for layer in text.layers[:3]:
        masks.append(valid); lengths.append(hidden.shape[1])
        hidden=layer(hidden,attention_mask=causal_mask,
            position_ids=pos,position_embeddings=rotary,past_key_value=cache,
            use_cache=cache is not None,cache_position=slots)[0]
    columns=fused.get('last_query_columns')
    if columns is None:
        columns=torch.where(valid,torch.arange(valid.shape[1],device=valid.device)[None],-1).max(-1).values
    else:
        columns=columns.to(device=valid.device,non_blocking=True)
    return dict(hidden=hidden,valid=valid,pos=pos,image_positions=fused['image_positions'],
        last_query_columns=columns,original_lengths=fused['original_lengths'],prefix_masks=masks,lengths=lengths)


def predictor_inputs(prefix, batch):
    hidden = prefix['hidden']
    image = prefix['image_positions']
    question, question_mask, question_positions = gather_questions(
        hidden, prefix['pos'], batch
    )
    return dict(
        visual=hidden.gather(1, image[:, :, None].expand(-1, -1, hidden.shape[-1])),
        question=question,
        question_mask=question_mask,
        visual_positions=prefix['pos'].gather(1, image),
        question_positions=question_positions,
    )


@torch.no_grad()
def frozen_features(backbone, batch):
    return predictor_inputs(run_prefix(backbone, fuse_inputs(backbone, batch)), batch)