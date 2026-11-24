"""Frozen native fusion and exactly three complete decoder blocks.

run_prefix returns raw hidden states, validity and original RoPE positions, image
columns, last valid physical query columns, original lengths, layer masks/widths.
No final normalization or suffix layer is applied.
"""
import torch
from smolvlm_topk_deferred.generation import _mask


@torch.no_grad()
def fuse_inputs(backbone, batch):
    backbone.eval()
    if any(p.requires_grad for p in backbone.parameters()):
        raise ValueError('native backbone must be frozen')
    device = next(backbone.parameters()).device
    ids, valid = batch['input_ids'].to(device), batch['attention_mask'].to(device)
    if not valid.bool().any(1).all():
        raise ValueError('empty native prompt')
    visual = backbone._project(batch['visual_features'], ids.shape[0])
    embedded = backbone.native.model.inputs_merger(input_ids=ids,
        inputs_embeds=backbone.native.get_input_embeddings()(ids), image_hidden_states=visual)
    pos = (valid.long().cumsum(-1)-1).masked_fill(~valid.bool(), 0)
    image = [torch.where((row == backbone.native.config.image_token_id) & mask.bool())[0]
             for row,mask in zip(ids,valid)]
    if any(len(x) != backbone.visual_token_len for x in image):
        raise ValueError('unexpected visual token count')
    return dict(input_ids=ids, inputs_embeds=embedded, attention_mask=valid,
                position_ids=pos, image_positions=torch.stack(image),
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
    for layer in text.layers[:3]:
        masks.append(valid.clone()); lengths.append(hidden.shape[1])
        hidden=layer(hidden,attention_mask=_mask(valid,hidden.shape[1],dtype=hidden.dtype),
            position_ids=pos,position_embeddings=text.rotary_emb(hidden,pos),past_key_value=cache,
            use_cache=cache is not None,cache_position=torch.arange(hidden.shape[1],device=hidden.device))[0]
    columns=torch.where(valid,torch.arange(valid.shape[1],device=valid.device)[None],-1).max(-1).values
    return dict(hidden=hidden,valid=valid,pos=pos,image_positions=fused['image_positions'],
        last_query_columns=columns,original_lengths=fused['original_lengths'],prefix_masks=masks,lengths=lengths)


def predictor_inputs(prefix):
    hidden=prefix['hidden']; image=prefix['image_positions']; query=prefix['last_query_columns'][:,None]
    return dict(visual=hidden.gather(1,image[:,:,None].expand(-1,-1,hidden.shape[-1])),
        last_query=hidden.gather(1,query[:,:,None].expand(-1,-1,hidden.shape[-1])),
        visual_positions=prefix['pos'].gather(1,image),query_positions=prefix['pos'].gather(1,query))


@torch.no_grad()
def frozen_features(backbone,batch):
    return predictor_inputs(run_prefix(backbone,fuse_inputs(backbone,batch)))
