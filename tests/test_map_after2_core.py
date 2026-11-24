import math
import pytest
import torch
from test_smolvlm_pruned_decoder import fixture


def test_after2_predictor_math_and_signature():
    from transformers import LlamaConfig
    from smolvlm_map_predictor_after2.model import MapPredictor, METHOD_ID
    from transformers.models.llama.modeling_llama import rotate_half
    p = MapPredictor(LlamaConfig(hidden_size=576,num_attention_heads=9))
    assert sum(x.numel() for x in p.parameters()) == 197120
    assert p.architecture_signature['method'] == METHOD_ID
    visual, query = torch.randn(2,384,576), torch.randn(2,1,576)
    vp, qp = torch.arange(384)[None].expand(2,-1), torch.tensor([[400],[420]])
    q,k = p.Wq(p.question_mlp(query)),p.Wk(p.visual_mlp(visual))
    def rope(x,pos):
        c,s=p.rotary(x,pos)
        return x*c+rotate_half(x)*s
    expected=(rope(q,qp)@rope(k,vp).transpose(-1,-2)/8).log_softmax(-1)[:,0]
    torch.testing.assert_close(p(visual,query,vp,qp),expected)
    with pytest.raises(ValueError): p(visual,query.expand(-1,2,-1),vp,qp.expand(-1,2))


def test_after2_prefix_native_boundary_grad_and_cache():
    from smolvlm_map_predictor_after2.prefix import fuse_inputs,run_prefix,predictor_inputs,frozen_features
    from smolvlm_map_predictor_after2.model import MapPredictor
    from transformers.cache_utils import DynamicCache
    model,batch=fixture()
    rng=torch.get_rng_state().clone()
    weights={k:v.clone() for k,v in model.state_dict().items()}
    captured=[]
    h=model.native.model.text_model.layers[2].register_forward_hook(lambda m,a,y:captured.append(y[0].detach().clone()))
    with torch.no_grad():
        model.native(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],
            image_hidden_states=model._project(batch['visual_features'],2),use_cache=False)
    h.remove()
    calls=[]
    handles=[layer.register_forward_hook(lambda m,a,y,i=i:calls.append(i)) for i,layer in enumerate(model.native.model.text_model.layers)]
    forbidden=[]
    def forbid(module,args):
        forbidden.append(type(module).__name__)
        raise AssertionError('forbidden native path')
    guards=[module.register_forward_pre_hook(forbid) for module in (model.native,model.native.model.text_model,model.native.model.vision_model,model.native.model.text_model.norm,model.native.lm_head)]
    cache_flags=[]
    checks=[layer.register_forward_pre_hook(lambda m,a,k:cache_flags.append((k['use_cache'],k['past_key_value'])),with_kwargs=True) for layer in model.native.model.text_model.layers[:3]]
    fused=fuse_inputs(model,batch)
    prefix=run_prefix(model,fused)
    for handle in guards+checks: handle.remove()
    assert not forbidden and cache_flags == [(False,None)]*3
    assert calls == [0,1,2]
    for handle in handles: handle.remove()
    torch.testing.assert_close(prefix['hidden'][prefix['valid']],captured[0][prefix['valid']],atol=1e-6,rtol=1e-5)
    cache=DynamicCache()
    cached=run_prefix(model,fused,cache)
    torch.testing.assert_close(prefix['hidden'],cached['hidden'])
    assert [cache.get_seq_length(i) for i in range(3)] == [fused['input_ids'].shape[1]]*3
    features=frozen_features(model,batch)
    assert not features['visual'].is_inference() and not features['visual'].requires_grad
    assert features['last_query'].shape == (2,1,model.geometry['text_dim'])
    expected=torch.where(prefix['valid'],torch.arange(prefix['valid'].shape[1]),-1).max(-1).values
    assert torch.equal(prefix['last_query_columns'],expected)
    assert torch.equal(features['query_positions'],prefix['pos'].gather(1,expected[:,None]))
    assert torch.equal(rng,torch.get_rng_state())
    p=MapPredictor(model.native.config.text_config,hidden_dim=8,projection_dim=4,visual_tokens=24)
    p(**features).square().sum().backward()
    assert all(x.grad is None for x in model.parameters())
    assert any(x.grad is not None for x in p.parameters())
    for k,v in model.state_dict().items(): assert torch.equal(v,weights[k])


def test_last_query_uses_physical_valid_column_with_right_and_left_padding():
    from smolvlm_map_predictor_after2.prefix import fuse_inputs,run_prefix,predictor_inputs
    model,batch=fixture()
    # Move the first native prompt to right padding without changing its valid tokens.
    batch=dict(batch)
    for key in ('input_ids','attention_mask'):
        batch[key]=batch[key].clone()
        valid=batch['attention_mask'][0].bool() if key == 'input_ids' else batch[key][0].bool()
        row=batch[key][0].clone()
        batch[key][0]=torch.cat((row[valid],row[~valid]))
    fused=fuse_inputs(model,batch)
    prefix=run_prefix(model,fused)
    features=predictor_inputs(prefix)
    for i in range(2):
        column=torch.where(fused['attention_mask'][i].bool())[0][-1]
        assert prefix['last_query_columns'][i] == column
        torch.testing.assert_close(features['last_query'][i,0],prefix['hidden'][i,column])
        assert features['query_positions'][i,0] == fused['original_lengths'][i]-1

def test_single_query_gradcheck_and_extreme_logits():
    from transformers import LlamaConfig
    from smolvlm_map_predictor_after2.model import MapPredictor
    predictor=MapPredictor(LlamaConfig(hidden_size=4,num_attention_heads=1),hidden_dim=4,projection_dim=2,visual_tokens=3).double()
    visual=torch.randn(1,3,4,dtype=torch.double,requires_grad=True)
    query=torch.randn(1,1,4,dtype=torch.double,requires_grad=True)
    vp,qp=torch.tensor([[1,4,7]]),torch.tensor([[12]])
    assert torch.autograd.gradcheck(lambda v,q:predictor(v,q,vp,qp),(visual,query))
    predictor=predictor.float()
    result=predictor(visual.detach().float()*1e3,query.detach().float()*1e3,vp,qp)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result.logsumexp(-1),torch.zeros(1),atol=1e-5,rtol=0)
