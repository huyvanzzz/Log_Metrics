import torch
import pytest
from test_smolvlm_model import tiny_components
from smolvlm_inference.pruning import stage1_indices, stage2_indices
from smolvlm_inference.pruning_text import PruningBatchCollator


def test_stage1_reference_and_identity():
    torch.manual_seed(8)
    v, q = torch.randn(2,12,5), torch.randn(2,4,5)
    mask = torch.tensor([[1,1,0,0],[1,1,1,1]],dtype=torch.bool)
    indices,cameras,debug=stage1_indices(v,q,mask,keep=8,num_views=3)
    dots=torch.einsum('bvnd,bqd->bvnq',v.reshape(2,3,4,5),q)
    expected=(dots.softmax(2)*mask[:,None,None,:]).sum(-1)/mask.sum(-1)[:,None,None]
    assert torch.allclose(debug['relevance'],expected)
    assert indices.shape==(2,8)
    assert torch.equal(cameras,indices//4)
    assert torch.equal(stage1_indices(v,q,mask,keep=12,num_views=3)[0],torch.arange(12).expand(2,-1))
    assert not any(t.requires_grad for t in debug.values())


def test_stage2_stable_raw_scores():
    scores=torch.tensor([[-3.,-3.,-4.,-2.,-2.,-5.]])
    cams=torch.tensor([[0,0,0,1,1,1]])
    i,c,d=stage2_indices(scores,cams,keep=4,num_views=2)
    assert i.tolist()==[[0,1,3,4]]
    r=torch.tensor([[-10/3,-3.]])
    z=(r-r.min(1,keepdim=True).values)/(r.max(1,keepdim=True).values-r.min(1,keepdim=True).values).clamp_min(1e-8)
    assert torch.allclose(d['camera_relevance'],r)
    assert torch.allclose(d['weights'],3*(1+z.softmax(1)-.5))


def test_question_offsets_padding_and_answer_invariance():
    _,p=tiny_components()
    collate=PruningBatchCollator(p,image_seq_len=4,max_length=2048)
    q='Front:\nWhat is visible?\nAssistant: What is visible?'
    items=[dict(question=q,answer='<image> answer',visual_features=torch.zeros(6,16,8)),dict(question='A car.',visual_features=torch.zeros(6,16,8))]
    batch=collate(items)
    other=collate([{**x,'answer':'other'} for x in items])
    assert torch.equal(batch['input_ids'],other['input_ids'])
    assert torch.equal(batch['question_mask'],other['question_mask'])
    assert batch['question_mask'].any(1).all()
    assert not (batch['question_mask'] & ~batch['attention_mask'].bool()).any()
    assert batch['image_positions'].shape==(2,24)
    assert batch['answers'][0]=='<image> answer'
    assert torch.equal(batch['source_position_ids'],(batch['attention_mask'].cumsum(1)-1).clamp_min(0))
    with pytest.raises(ValueError,match='question'):
        collate([{**items[0],'question':''}])

def _allocation(weights,capacities,total):
    quotas=weights/weights.sum()*total
    result=[min(cap,max(1,int(q))) for q,cap in zip(quotas,capacities)]
    while sum(result)>total:
        c=max((c for c in range(len(result)) if result[c]>1),key=lambda c:(result[c]-float(quotas[c]),-c))
        result[c]-=1
    while sum(result)<total:
        c=max((c for c in range(len(result)) if result[c]<capacities[c]),key=lambda c:(float(quotas[c])-result[c],-c))
        result[c]+=1
    return result


def test_stage1_independent_ccts_and_diversity_reference():
    torch.manual_seed(22)
    visual=torch.randn(1,18,7)
    q=torch.randn(1,3,7)
    indices,_,debug=stage1_indices(visual,q,torch.ones(1,3,dtype=torch.bool),keep=11,num_views=3)
    distances=[]
    relevances=[]
    for c in range(3):
        tokens=visual[0,c*6:(c+1)*6]
        unit=tokens/tokens.norm(dim=-1,keepdim=True)
        d=1-unit@unit.T
        d.fill_diagonal_(float('inf'))
        distances.append(d)
        relevances.append((tokens@q[0].T).softmax(0).mean(1))
    diversity=torch.tensor([d.min(1).values.mean() for d in distances])
    weights=((diversity-diversity.max())/(diversity.max()-diversity.min()+1e-5)).softmax(0)
    budgets=_allocation(weights,[6]*3,11)
    expected=[]
    for c,(d,r,k) in enumerate(zip(distances,relevances,budgets)):
        chosen=[int(r.argmax())]
        while len(chosen)<k:
            priority=d[chosen].min(0).values*r
            priority[chosen]=-torch.inf
            chosen.append(int(priority.argmax()))
        expected.extend(c*6+i for i in chosen)
    assert indices.tolist()==[sorted(expected)]
    assert debug['stage1_budgets'].tolist()==[budgets]
    assert torch.allclose(debug['stage1_diversity'][0],diversity)


def test_equal_geometry_stable_lower_index_and_singleton_identity():
    v=torch.ones(1,12,4)
    q=torch.ones(1,2,4)
    i,_,_=stage1_indices(v,q,torch.ones(1,2,dtype=torch.bool),keep=6,num_views=3)
    assert i.tolist()==[[0,1,4,5,8,9]]
    one,_,debug=stage1_indices(v[:,:3],q,torch.ones(1,2,dtype=torch.bool),keep=3,num_views=3)
    assert one.tolist()==[[0,1,2]]
    assert torch.isfinite(debug['stage1_diversity']).all()


def test_collator_question_character_overlap_exact():
    _,p=tiny_components()
    collate=PruningBatchCollator(p,image_seq_len=4,max_length=2048,max_new_tokens=0)
    question='Front:\nBack-Right:\nAssistant: A car.'
    batch=collate([dict(question=question,visual_features=torch.zeros(6,16,8))])
    from smolvlm_inference.text import chat_messages
    rendered=collate._expand(p.apply_chat_template(chat_messages(question),tokenize=False,add_generation_prompt=True))
    # Known native fixture suffix makes this independent of sentinel implementation.
    end=len(rendered)-len('<end_of_utterance>\nAssistant:')
    start=end-len(question)
    offsets=p.tokenizer(rendered,add_special_tokens=True,return_offsets_mapping=True)['offset_mapping']
    expected=torch.tensor([[b>start and a<end and b>a for a,b in offsets]])
    assert torch.equal(batch['question_mask'],expected)

@pytest.mark.parametrize('capacities,total', [([1,3,8],8),([8,1,3],7),([3,3,3],3),([2,2,2],6)])
def test_bounded_budget_matches_sequential_reference(capacities,total):
    from smolvlm_inference.pruning import allocate_bounded_budget
    weights=torch.tensor([[0.1,0.3,0.6]])
    actual=allocate_bounded_budget(weights,torch.tensor([capacities]),total=total)
    assert actual.tolist()==[_allocation(weights[0],capacities,total)]


def test_stage2_unequal_camera_counts_and_identity():
    scores=torch.tensor([[.4,.1,.8,.2,.7,.9,.3,.6]])
    cameras=torch.tensor([[0,1,1,1,2,2,2,2]])
    indices,selected,debug=stage2_indices(scores,cameras,keep=6,num_views=3)
    r=torch.tensor([.4,(.1+.8+.2)/3,(.7+.9+.3+.6)/4])
    z=(r-r.min())/(r.max()-r.min()).clamp_min(1e-8)
    counts=torch.tensor([1,3,4])
    weights=counts*(1+z.softmax(0)-counts/8)
    budgets=_allocation(weights,counts.tolist(),6)
    expected=[]
    for c,k in enumerate(budgets):
        candidates=[i for i in range(8) if cameras[0,i]==c]
        expected.extend(sorted(candidates,key=lambda i:(-float(scores[0,i]),i))[:k])
    assert indices.tolist()==[sorted(expected)]
    assert debug['stage2_budgets'].tolist()==[budgets]
    identity,_,_=stage2_indices(scores,cameras,keep=8,num_views=3)
    assert identity.tolist()==[list(range(8))]

@pytest.mark.parametrize('question',['What is visible?','Front:\nWhat is visible?\nAssistant: What is visible?','__PRUNING_QUESTION_SPAN__ A car.'])
def test_prompt_length_without_features_equals_collated_length(question):
    _,p=tiny_components()
    collate=PruningBatchCollator(p,image_seq_len=4,max_length=2048)
    row=dict(question=question,answer='<image> unused reference')
    expected=collate([{**row,'visual_features':torch.zeros(6,16,8)}])['attention_mask'].sum().item()
    assert collate.prompt_length(row)==expected
    with pytest.raises(ValueError,match='question'):
        collate.prompt_length({'question':''})
    with pytest.raises(ValueError,match='placeholder'):
        collate.prompt_length({'question':'<image>'})
    short=PruningBatchCollator(p,image_seq_len=4,max_length=8)
    with pytest.raises(ValueError,match='context budget'):
        short.prompt_length(row)
