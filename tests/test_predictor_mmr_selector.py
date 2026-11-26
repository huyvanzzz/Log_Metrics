import torch
import pytest
from smolvlm_predictor_mmr.selector import MMRSelector


def reference(logp, states, count, weight=.9):
    p=logp.exp(); shifted=p-p.min(); span=shifted.max()
    r=shifted/(span if span else 1)
    unit=torch.nn.functional.normalize(states,dim=-1,eps=1e-12)
    sim=unit@unit.T
    chosen=[]; penalties=[]
    for step in range(count):
        penalty=torch.zeros_like(r) if not chosen else sim[:,chosen].max(-1).values
        score=r if not chosen else weight*r-(1-weight)*penalty
        score[chosen]=-torch.inf
        pick=int(score.argmax()); chosen.append(pick); penalties.append(penalty[pick])
    return chosen,torch.stack(penalties)


def test_reference_exact_and_nonmutation():
    torch.manual_seed(42)
    logp=torch.randn(3,24).log_softmax(-1); states=torch.randn(3,24,8)
    before=states.clone(); rng=torch.get_rng_state().clone()
    selector=MMRSelector(); selected=selector(logp,states,12)
    for row in range(3):
        picks,penalties=reference(logp[row],states[row],12)
        assert selected[row].tolist()==sorted(picks)
        assert selector.last_debug['pick_order'][row].tolist()==picks
        torch.testing.assert_close(selector.last_debug['penalties'][row],penalties)
    assert torch.equal(states,before) and torch.equal(rng,torch.get_rng_state())


def test_flat_zero_negative_underflow_ties_and_allkeep():
    selector=MMRSelector()
    states=torch.tensor([[[1.,0.],[1.,0.],[-1.,0.],[0.,0.]]])
    logp=torch.full((1,4),-1000.)
    result=selector(logp,states,4)
    assert result.tolist()==[[0,1,2,3]]
    assert selector.last_debug['pick_order'][0,:2].tolist()==[0,2]
    assert selector.last_debug['zero_probability_count'].tolist()==[4]
    assert selector.last_debug['flat_rows'].tolist()==[True]
    assert selector.last_debug['penalties'][0,1].item()==-1.


def test_weight_one_matches_stable_probability_topk_and_probability_not_log_scale():
    states=torch.randn(2,10,3); scores=torch.tensor([[0.,-1.,-2.,-3.,-4.,-5.,-6.,-7.,-8.,-9.]]).expand(2,-1)
    selector=MMRSelector(weight=1.)
    assert selector(scores,states,4).tolist()==[[0,1,2,3]]*2
    p=scores.exp(); shifted=p-p.amin(-1,keepdim=True)
    torch.testing.assert_close(selector.last_debug['relevance'],shifted/shifted.amax(-1,keepdim=True))


@pytest.mark.parametrize('count',[0,25,True])
def test_invalid_count(count):
    with pytest.raises(ValueError): MMRSelector()(torch.zeros(1,24),torch.zeros(1,24,3),count)


def test_invalid_inputs():
    with pytest.raises(ValueError): MMRSelector(weight=1.1)
    with pytest.raises(ValueError): MMRSelector()(torch.full((1,2),float('nan')),torch.zeros(1,2,3),1)
    with pytest.raises(ValueError): MMRSelector()(torch.zeros(1,2,dtype=torch.float64),torch.zeros(1,2,3),1)

# Literal default-method MMR loop from Nguyet commit
# 5061b949299504c821994591b8bc1b65530bad04, posprune_ablation/decoder.py.
def nguyet_mmr(relevance,states,count,weight):
    unit=torch.nn.functional.normalize(states,dim=-1)
    similarity=unit@unit.transpose(1,2)
    selected=torch.zeros_like(relevance,dtype=torch.bool)
    redundancy=torch.full_like(relevance,-torch.inf)
    penalty=torch.zeros_like(relevance)
    scores=relevance
    picks,penalties=[],[]
    for _ in range(count):
        pick=scores.masked_fill(selected,-torch.inf).argmax(dim=1,keepdim=True)
        picks.append(pick); penalties.append(penalty.gather(1,pick))
        selected.scatter_(1,pick,True)
        candidate=similarity.gather(1,pick[...,None].expand(-1,-1,relevance.shape[1])).squeeze(1)
        redundancy=torch.maximum(redundancy,candidate)
        penalty=redundancy
        scores=weight*relevance-(1-weight)*penalty
    return torch.cat(picks,dim=1),torch.cat(penalties,dim=1)


def test_pinned_nguyet_exact_pick_and_penalty_parity():
    generator=torch.Generator().manual_seed(123)
    logp=torch.randn(3,24,generator=generator).log_softmax(-1)
    states=torch.randn(3,24,8,generator=generator)
    states[1,2]=states[1,3]; states[2]=0
    selector=MMRSelector(); selector(logp,states,12)
    p=logp.exp(); shifted=p-p.amin(-1,keepdim=True)
    relevance=shifted/shifted.amax(-1,keepdim=True).masked_fill(shifted.amax(-1,keepdim=True)==0,1)
    picks,penalties=nguyet_mmr(relevance,states,12,.9)
    assert torch.equal(selector.last_debug['pick_order'],picks)
    assert torch.equal(selector.last_debug['penalties'],penalties)
