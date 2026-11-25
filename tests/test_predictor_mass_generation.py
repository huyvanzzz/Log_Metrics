import pytest
import torch
from test_smolvlm_pruned_decoder import fixture


def test_visual_mass_reference_and_fallback():
    from smolvlm_predictor_mass.attention import preserve_visual_mass
    p=torch.tensor([[[[.3,.15,.05]]]])
    keep=torch.tensor([False,True,True])
    out,fb=preserve_visual_mass(p,p.log(),keep)
    torch.testing.assert_close(out,torch.tensor([[[[0.,.375,.125]]]]))
    p=torch.tensor([[[[.5,0.,0.]]]])
    out,fb=preserve_visual_mass(p,torch.tensor([[[[0.,-1000.,-1001.]]]]),keep)
    assert fb.all() and torch.isfinite(out).all()
    torch.testing.assert_close(out.sum(-1),p.sum(-1))


def test_zero_mass_and_missing_causal_survivor():
    from smolvlm_predictor_mass.attention import preserve_visual_mass
    zero=torch.zeros(2,3)
    out,fallback=preserve_visual_mass(zero,zero,torch.tensor([True,False,False]))
    assert torch.equal(out,zero) and not fallback.any()
    with pytest.raises(ValueError,match='surviving causal'):
        preserve_visual_mass(torch.tensor([[.2,.3,.1]]),zero[:1],
                             torch.tensor([False,True,False]),
                             allowed=torch.tensor([[True,False,True]]))


@pytest.mark.parametrize('variant',['before','after2'])
def test_mass_generation_allkeep_and_fixed_reference(variant):
    from smolvlm_predictor_mass.generation import Decoder
    from smolvlm_map_predictor.model import MapPredictor as Before
    from smolvlm_map_predictor_after2.model import MapPredictor as After
    model,batch=fixture(layers=5)
    model.requires_grad_(False)
    batch['position_ids']=batch['source_position_ids']
    P=Before if variant=='before' else After
    predictor=P(model.native.config.text_config,hidden_dim=8,projection_dim=4,visual_tokens=24).eval()
    engine=Decoder(model,predictor,variant,24)
    actual=engine.generate(batch,4,True)
    native=model.generate_from_features(**{k:batch[k] for k in ('input_ids','attention_mask','visual_features')},max_new_tokens=4,num_beams=1,do_sample=False)
    assert [r['generated_ids'] for r in actual]==native.tolist()
    engine=Decoder(model,predictor,variant,12)
    rng=torch.get_rng_state().clone()
    calls=[]
    hook=predictor.register_forward_hook(lambda *a:calls.append(1))
    actual=engine.generate(batch,4,True)
    hook.remove()
    assert calls==[1]
    assert torch.equal(rng,torch.get_rng_state())
    info=engine.last_debug
    boundary=0 if variant=='before' else 3
    assert info['visual_counts_by_layer']==[[24,24]]*(boundary+1)+[[12,12]]*(4-boundary)
    assert actual==engine.generate_reference(batch,4)
    assert not hasattr(model.native.model.text_model.layers[boundary].self_attn,'_mass_intervention')


def test_attention_chunks_match_independent_full_reference_and_scope():
    import math
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb,repeat_kv
    from smolvlm_predictor_mass.attention import intervention
    from smolvlm_topk_deferred.generation import _mask
    model,batch=fixture()
    attn=model.native.model.text_model.layers[0].self_attn
    hidden=model.native.get_input_embeddings()(batch['input_ids'])
    pos=batch['source_position_ids']; valid=batch['attention_mask'].bool()
    image=batch['image_positions']; selected=torch.arange(0,24,2)[None].expand(2,-1)
    start=torch.where(batch['question_mask'],torch.arange(hidden.shape[1])[None],hidden.shape[1]).min(-1).values
    affected=valid & (torch.arange(hidden.shape[1])[None]>=start[:,None])
    rotary=model.native.model.text_model.rotary_emb(hidden,pos)
    mask=_mask(valid,hidden.shape[1])
    b,l,d=hidden.shape
    q=attn.q_proj(hidden).view(b,l,attn.num_heads,attn.head_dim).transpose(1,2)
    k=attn.k_proj(hidden).view(b,l,attn.num_key_value_heads,attn.head_dim).transpose(1,2)
    v=attn.v_proj(hidden).view(b,l,attn.num_key_value_heads,attn.head_dim).transpose(1,2)
    q,k=apply_rotary_pos_emb(q,k,*rotary)
    logits=q@repeat_kv(k,attn.num_key_value_groups).transpose(-1,-2)/math.sqrt(attn.head_dim)+mask
    original=logits.softmax(-1)
    reference=original.clone()
    for row in range(b):
        for head in range(attn.num_heads):
            for query in torch.where(affected[row])[0]:
                visual=original[row,head,query,image[row]]
                mass=visual.sum()
                replacement=torch.zeros_like(visual)
                replacement[selected[row]]=visual[selected[row]]/visual[selected[row]].sum()*mass
                reference[row,head,query,image[row]]=replacement
    expected=attn.o_proj((reference@repeat_kv(v,attn.num_key_value_groups)).transpose(1,2).reshape(b,l,d))
    counts={'q':0,'k':0,'v':0}
    def count(name):
        def hook(*args):counts[name]+=1
        return hook
    hooks=[getattr(attn,name+'_proj').register_forward_hook(count(name)) for name in counts]
    diagnostics={}
    with intervention(attn,image,selected,affected,diagnostics,query_chunk=7):
        actual=attn(hidden_states=hidden,attention_mask=mask,position_ids=pos,position_embeddings=rotary)[0]
    for hook in hooks:hook.remove()
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    assert counts==dict(q=1,k=1,v=1)
    assert abs(diagnostics['visual_mass_before']-diagnostics['visual_mass_after'])<1e-4
    assert 'forward' not in attn.__dict__
    with pytest.raises(RuntimeError,match='forced'):
        with intervention(attn,image,selected,affected,None):
            raise RuntimeError('forced')
    assert 'forward' not in attn.__dict__


@pytest.mark.parametrize('variant',['before','after2'])
def test_projected_batch_parity_and_early_eos(variant):
    from smolvlm_predictor_mass.generation import Decoder
    from smolvlm_map_predictor.model import MapPredictor as Before
    from smolvlm_map_predictor_after2.model import MapPredictor as After
    model,batch=fixture(layers=5); model.requires_grad_(False)
    batch['position_ids']=batch['source_position_ids']
    P=Before if variant=='before' else After
    predictor=P(model.native.config.text_config,hidden_dim=8,projection_dim=4,visual_tokens=24).eval()
    engine=Decoder(model,predictor,variant,12)
    expected=engine.generate(batch,3)
    projected=model._project(batch['visual_features'],2).reshape(2,24,-1)
    compact=dict(batch,projected_visual_features=projected)
    compact.pop('visual_features')
    assert expected==engine.generate(compact,3)
    model.native.lm_head.weight.zero_(); model.native.generation_config.eos_token_id=0
    result=engine.generate(compact,3)
    assert all(row['generated_ids']==[0] for row in result)
    assert result==engine.generate_reference(compact,3)


@pytest.mark.parametrize('variant',['before','after2'])
def test_complete_block_gather_and_compact_queries(variant):
    from smolvlm_predictor_mass.generation import Decoder
    from smolvlm_map_predictor.model import MapPredictor as Before
    from smolvlm_map_predictor_after2.model import MapPredictor as After
    from smolvlm_map_predictor_after2.prefix import fuse_inputs
    from smolvlm_topk_benchmark.generation import compact_prompt
    model,batch=fixture(layers=5); model.requires_grad_(False)
    batch['position_ids']=batch['source_position_ids']
    P=Before if variant=='before' else After
    predictor=P(model.native.config.text_config,hidden_dim=8,projection_dim=4,visual_tokens=24).eval()
    boundary=0 if variant=='before' else 3
    engine=Decoder(model,predictor,variant,12)
    outputs=[]; suffix=[]; calls=[]
    hooks=[model.native.model.text_model.layers[boundary].register_forward_hook(lambda m,a,y:outputs.append(y[0].clone())),
        model.native.model.text_model.layers[boundary+1].register_forward_pre_hook(lambda m,a:suffix.append(a[0].clone())),
        predictor.register_forward_hook(lambda *a:calls.append(1))]
    result=engine.generate(batch,1,True)
    for h in hooks:h.remove()
    fused=fuse_inputs(model,batch)
    expected=compact_prompt(dict(fused,inputs_embeds=outputs[0]),[r['visual_indices'] for r in result])
    assert torch.equal(expected['inputs_embeds'],suffix[0])
    assert calls==[1]
    cols=[torch.where(row)[0] for row in batch['question_mask']]
    width=max(map(len,cols)); columns=torch.zeros((2,width),dtype=torch.long); valid=torch.zeros_like(columns,dtype=torch.bool)
    for i,c in enumerate(cols):columns[i,:len(c)]=c;valid[i,:len(c)]=True
    compact=dict(batch,question_columns=columns,question_valid=valid,question_mask=valid)
    assert result==engine.generate(compact,1)


@pytest.mark.parametrize('variant',['before','after2'])
def test_mixed_eos_reference_and_native_decode_positions(variant):
    from smolvlm_predictor_mass.generation import Decoder
    from smolvlm_map_predictor.model import MapPredictor as Before
    from smolvlm_map_predictor_after2.model import MapPredictor as After
    model,batch=fixture(layers=5);model.requires_grad_(False)
    batch['position_ids']=batch['source_position_ids']
    P=Before if variant=='before' else After
    predictor=P(model.native.config.text_config,hidden_dim=8,projection_dim=4,visual_tokens=24).eval()
    engine=Decoder(model,predictor,variant,12)
    states=[]
    h=model.native.lm_head.register_forward_pre_hook(lambda m,a:states.append(a[0].clone()))
    engine.generate(batch,1);h.remove()
    x,y=states[0][:,0];direction=x/x.norm()-y/y.norm()
    model.native.lm_head.weight.zero_();model.native.lm_head.weight[0]=direction;model.native.lm_head.weight[1]=-direction
    model.native.generation_config.eos_token_id=0
    actual=engine.generate(batch,4,True)
    assert actual==engine.generate_reference(batch,4)
    assert actual[0]['generated_ids']==[0]
    assert actual[1]['generated_token_count']>1
