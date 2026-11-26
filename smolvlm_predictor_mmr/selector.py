"""Predictor relevance with Nguyet-style maximal marginal relevance selection."""
import math
import torch
from torch.nn.functional import normalize


class MMRSelector:
    def __init__(self, weight=.9):
        if isinstance(weight,bool) or not isinstance(weight,(int,float)) or not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError('MMR weight must be finite in [0,1]')
        self.weight=float(weight)
        self.last_debug={}

    @torch.no_grad()
    def __call__(self,log_scores,visual_states,keep_tokens):
        if (log_scores.ndim!=2 or visual_states.ndim!=3 or visual_states.shape[:2]!=log_scores.shape or
            log_scores.shape[0]<1 or visual_states.shape[-1]<1 or log_scores.device!=visual_states.device or
            log_scores.dtype!=torch.float32 or visual_states.dtype!=torch.float32):
            raise ValueError('MMR requires matching nonempty FP32 scores [B,N] and states [B,N,D]')
        if type(keep_tokens) is not int or not 0<keep_tokens<=log_scores.shape[1]:
            raise ValueError('Invalid MMR keep_tokens')
        if not bool(torch.isfinite(log_scores).all() & torch.isfinite(visual_states).all()):
            raise ValueError('MMR inputs must be finite')
        p=log_scores.exp()
        if not bool(torch.isfinite(p).all()): raise ValueError('MMR probabilities overflowed')
        shifted=p-p.amin(-1,keepdim=True)
        span=shifted.amax(-1,keepdim=True)
        relevance=shifted/span.masked_fill(span==0,1)
        unit=normalize(visual_states,dim=-1,eps=1e-12)
        similarity=unit@unit.transpose(1,2)
        selected=torch.zeros_like(relevance,dtype=torch.bool)
        redundancy=torch.full_like(relevance,-torch.inf)
        penalty=torch.zeros_like(relevance)
        scores=relevance
        picks=[]; penalties=[]
        for _ in range(keep_tokens):
            pick=scores.masked_fill(selected,-torch.inf).argmax(1,keepdim=True)
            picks.append(pick); penalties.append(penalty.gather(1,pick))
            selected.scatter_(1,pick,True)
            candidate=similarity.gather(1,pick[...,None].expand(-1,-1,relevance.shape[1])).squeeze(1)
            redundancy=torch.maximum(redundancy,candidate)
            penalty=redundancy
            scores=self.weight*relevance-(1-self.weight)*penalty
        pick_order=torch.cat(picks,1); indices=pick_order.sort(1).values
        self.last_debug=dict(relevance=relevance.detach(),relevance_min=relevance.amin(-1),
            relevance_max=relevance.amax(-1),zero_probability_count=(p==0).sum(-1),
            flat_rows=(span[:,0]==0),pick_order=pick_order.detach(),
            penalties=torch.cat(penalties,1).detach(),selected_source_indices=indices.detach())
        return indices
