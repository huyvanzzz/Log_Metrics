"""Deterministic, parameter-free camera token selection."""
import torch
import torch.nn.functional as F

def _require_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        detached = value.detach()
        raise FloatingPointError(
            f"{name} contains NaN/Inf: shape={tuple(value.shape)} "
            f"nan={int(torch.isnan(detached).sum())} inf={int(torch.isinf(detached).sum())}"
        )


@torch.no_grad()
def allocate_bounded_budget(
    weights: torch.Tensor,
    capacity: torch.Tensor,
    *,
    total: int,
    minimum: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Allocate an exact integer budget independently for every batch row."""
    if weights.ndim != 2 or capacity.shape != weights.shape:
        raise ValueError("weights and capacity must have shape [B,V]")
    _require_finite("allocation weights", weights)
    if minimum * weights.shape[1] > total:
        raise ValueError("minimum allocation exceeds total")
    if bool((capacity < minimum).any()) or bool((capacity.sum(dim=1) < total).any()):
        raise ValueError("capacity cannot satisfy requested allocation")

    safe_weights = weights.clamp_min(0)
    sums = safe_weights.sum(dim=1, keepdim=True)
    fallback = capacity.to(weights.dtype)
    safe_weights = torch.where(sums <= eps, fallback, safe_weights)
    quotas = safe_weights / safe_weights.sum(dim=1, keepdim=True).clamp_min(eps) * float(total)
    allocation = torch.floor(quotas).to(torch.long)
    allocation = torch.maximum(allocation, torch.full_like(allocation, minimum))
    allocation = torch.minimum(allocation, capacity.to(torch.long))

    # Each camera's next marginal priority drops by one per repaired unit.
    # Stable ranking of all possible units is equivalent to sequential argmax,
    # including its lowest-camera-index tie break, without O(total) launches.
    units = torch.arange(total, device=weights.device)[None, None, :]
    for removing in (True, False):
        count = (allocation.sum(1) - total if removing else total - allocation.sum(1)).clamp_min(0)
        room = allocation - minimum if removing else capacity - allocation
        levels = allocation.unsqueeze(-1) - units if removing else allocation.unsqueeze(-1) + units
        if removing:
            marginal = levels.to(quotas.dtype) - quotas.unsqueeze(-1)
        else:
            marginal = quotas.unsqueeze(-1) - levels.to(quotas.dtype)
        marginal = marginal.masked_fill(units >= room.unsqueeze(-1), -torch.inf)
        ranked = marginal.flatten(1).argsort(dim=1, descending=True, stable=True)
        chosen = torch.arange(ranked.shape[1], device=weights.device)[None, :] < count[:, None]
        changes = torch.zeros_like(allocation).scatter_add_(1, ranked // total, chosen.long())
        allocation = allocation - changes if removing else allocation + changes
    if bool((allocation.sum(dim=1) != total).any()):
        raise RuntimeError("bounded allocation failed to reach exact total")
    return allocation


@torch.no_grad()
def _ccts_selection(
    distances: torch.Tensor, relevance: torch.Tensor, budgets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Algorithm 1, batched over samples/views; one scalar sync per call."""
    selected = torch.zeros_like(relevance, dtype=torch.bool)
    seed = relevance.argmax(-1)
    selected.scatter_(-1, seed.unsqueeze(-1), (budgets > 0).unsqueeze(-1))
    def distance_from(indices):
        rows = indices[..., None, None].expand(*indices.shape, 1, relevance.shape[-1])
        return distances.gather(-2, rows).squeeze(-2)

    running = distance_from(seed)
    for step in range(1, int(budgets.max().item())):
        active = step < budgets
        scores = (running * relevance).masked_fill(selected | ~active.unsqueeze(-1), -torch.inf)
        chosen = scores.argmax(-1)
        addition = torch.zeros_like(selected).scatter_(-1, chosen.unsqueeze(-1), active.unsqueeze(-1))
        selected |= addition
        running = torch.minimum(running, distance_from(chosen))
    return selected, seed



@torch.no_grad()
def stage1_indices(visual, question, question_mask, *, keep=196, num_views=6, eps=1e-8):
    if visual.ndim != 3 or visual.shape[1] % num_views:
        raise ValueError('visual must be [B,V*N,D]')
    batch,total,dim=visual.shape
    n=total//num_views
    if question.ndim!=3 or question.shape[0]!=batch or question.shape[2]!=dim:
        raise ValueError('question must be [B,Q,D]')
    if question_mask.shape!=question.shape[:2] or not question_mask.bool().any(1).all():
        raise ValueError('question needs valid tokens in every row')
    if not num_views<=keep<=total:
        raise ValueError('invalid stage1 budget')
    _require_finite('visual',visual)
    _require_finite('question',question)
    base=torch.arange(total,device=visual.device).expand(batch,-1)
    per_view=visual.float().reshape(batch,num_views,n,dim)
    dots=torch.einsum('bvnd,bqd->bvnq',per_view,question.float())
    mask=question_mask.bool()[:,None,None,:]
    relevance=(dots.softmax(2)*mask).sum(-1)/mask.sum(-1)
    normalized=F.normalize(per_view,dim=-1,eps=eps)
    distances=1-normalized@normalized.transpose(-1,-2)
    distances.masked_fill_(torch.eye(n,device=visual.device,dtype=torch.bool)[None,None],float('inf'))
    diversity=distances.min(-1).values.mean(-1) if n>1 else per_view.new_zeros(batch,num_views)
    high,low=diversity.max(1,keepdim=True).values,diversity.min(1,keepdim=True).values
    weights=((diversity-high)/(high-low+1e-5)).softmax(1)
    capacity=torch.full((batch,num_views),n,device=visual.device,dtype=torch.long)
    budgets=allocate_bounded_budget(weights,capacity,total=keep,eps=eps)
    selected,seeds=_ccts_selection(distances,relevance,budgets)
    indices=base.masked_fill(~selected.reshape(batch,total),total).sort(1).values[:,:keep]
    debug=dict(relevance=relevance,raw_relevance=relevance.reshape(batch,total),stage1_budgets=budgets,
               stage1_indices=indices,stage1_diversity=diversity,stage1_dra_weights=weights,stage1_seed_indices=seeds)
    return indices,indices//n,debug


@torch.no_grad()
def stage2_indices(scores,camera_ids,*,keep=32,num_views=6,eps=1e-8):
    if scores.ndim!=2 or camera_ids.shape!=scores.shape:
        raise ValueError('scores and camera_ids must be [B,N]')
    if camera_ids.dtype!=torch.long or ((camera_ids<0)|(camera_ids>=num_views)).any():
        raise ValueError('camera IDs must be integers within camera range')
    _require_finite('scores',scores)
    scores=scores.float()
    batch,count=scores.shape
    counts=torch.stack([(camera_ids==c).sum(1) for c in range(num_views)],1)
    relevance=torch.stack([(scores*(camera_ids==c)).sum(1)/counts[:,c].clamp_min(1) for c in range(num_views)],1)
    low,high=relevance.min(1,keepdim=True).values,relevance.max(1,keepdim=True).values
    shares=((relevance-low)/(high-low).clamp_min(eps)).softmax(1)
    weights=counts*(1+shares-counts/counts.sum(1,keepdim=True))
    budgets=allocate_bounded_budget(weights,counts,total=keep,eps=eps)
    selected=torch.zeros_like(camera_ids,dtype=torch.bool)
    for c in range(num_views):
        ranked=scores.masked_fill(camera_ids!=c,-torch.inf).argsort(dim=1,descending=True,stable=True)
        active=torch.arange(count,device=scores.device)[None,:]<budgets[:,c,None]
        selected.scatter_(1,ranked,selected.gather(1,ranked)|active)
    base=torch.arange(count,device=scores.device).expand(batch,-1)
    indices=base.masked_fill(~selected,count).sort(1).values[:,:keep]
    return indices,camera_ids.gather(1,indices),dict(stage2_indices=indices,stage2_budgets=budgets,camera_relevance=relevance,weights=weights)
