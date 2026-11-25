"""Post-softmax visual-mass intervention; full keys, bounded query chunks."""
import math
from contextlib import contextmanager
import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

MASS_EPS = 1e-8


def preserve_visual_mass(probs, logits, keep, allowed=None):
    keep=keep.to(device=probs.device,dtype=torch.bool)
    if allowed is not None:
        keep=keep & allowed
    mass=probs.sum(-1,keepdim=True)
    if bool(((~keep.any(-1,keepdim=True)) & (mass>0)).any()):
        raise ValueError('no surviving causal visual key in affected attention row')
    remaining=probs.masked_fill(~keep,0).sum(-1,keepdim=True)
    fallback=(remaining<=MASS_EPS)&(mass>0)
    scaled=probs.masked_fill(~keep,0)/remaining.clamp_min(MASS_EPS)*mass
    conditional=logits.masked_fill(~keep,-torch.inf).softmax(-1).nan_to_num()
    return torch.where(fallback,conditional*mass,scaled),fallback


@contextmanager
def intervention(attn, image_positions, selected, affected, diagnostics, query_chunk=64):
    """Temporarily replace only this native attention, restoring even on failure."""
    if getattr(attn.config, 'pretraining_tp', 1) != 1:
        raise ValueError('mass attention requires pretraining_tp=1')
    if not 0<query_chunk<=64:
        raise ValueError('query_chunk must be in 1..64')
    had_override='forward' in attn.__dict__
    old_override=attn.__dict__.get('forward')
    def forward(hidden_states,attention_mask=None,position_ids=None,past_key_value=None,
                output_attentions=False,use_cache=False,cache_position=None,position_embeddings=None,**kwargs):
        if output_attentions:
            raise ValueError('full attention export is not supported by mass decoder')
        b,l,_=hidden_states.shape
        q=attn.q_proj(hidden_states).view(b,l,attn.num_heads,attn.head_dim).transpose(1,2)
        k=attn.k_proj(hidden_states).view(b,l,attn.num_key_value_heads,attn.head_dim).transpose(1,2)
        v=attn.v_proj(hidden_states).view(b,l,attn.num_key_value_heads,attn.head_dim).transpose(1,2)
        if position_embeddings is None:
            position_embeddings=attn.rotary_emb(v,position_ids)
        q,k=apply_rotary_pos_emb(q,k,*position_embeddings)
        if past_key_value is not None:
            k,v=past_key_value.update(k,v,attn.layer_idx,dict(sin=position_embeddings[1],cos=position_embeddings[0],cache_position=cache_position))
        k=repeat_kv(k,attn.num_key_value_groups)
        v=repeat_kv(v,attn.num_key_value_groups)
        output=torch.empty_like(q)
        keep=torch.zeros((b,image_positions.shape[1]),dtype=torch.bool,device=q.device).scatter(1,selected,True)
        summaries=[]
        for start in range(0,l,query_chunk):
            stop=min(start+query_chunk,l)
            scores=q[:,:,start:stop]@k.transpose(-1,-2)/math.sqrt(attn.head_dim)
            if attention_mask is None:
                raise ValueError('explicit native causal/padding mask required')
            mask=attention_mask[:,:,start:stop,:k.shape[-2]]
            scores=scores+mask
            probs=scores.softmax(-1,dtype=torch.float32).to(q.dtype)
            index=image_positions[:,None,None,:].expand(b,q.shape[1],stop-start,-1)
            original=probs.gather(-1,index)
            raw=scores.gather(-1,index)
            active=affected[:,None,start:stop,None]
            allowed=(mask.expand(b,q.shape[1],stop-start,-1).gather(-1,index)==0)
            # Inactive rows may precede every visual key: do not require survivors.
            replacement,fallback=preserve_visual_mass(original.masked_fill(~active,0),raw,keep[:,None,None],allowed)
            replacement=torch.where(active,replacement,original)
            probs.scatter_(-1,index,replacement)
            output[:,:,start:stop]=probs@v
            if diagnostics is not None:
                summaries.append(torch.stack(((original.sum(-1)*active.squeeze(-1)).sum(),
                    (replacement.sum(-1)*active.squeeze(-1)).sum(),(fallback & active).sum())))
        if diagnostics is not None:
            values=torch.stack(summaries).sum(0).detach().cpu().tolist()
            diagnostics.update(visual_mass_before=values[0],visual_mass_after=values[1],fallback_count=int(values[2]),query_chunk_size=query_chunk)
        result=output.transpose(1,2).contiguous().reshape(b,l,-1)
        return attn.o_proj(result),None,past_key_value
    attn.forward=forward
    try:
        yield
    finally:
        if had_override:
            attn.forward=old_override
        else:
            del attn.forward
