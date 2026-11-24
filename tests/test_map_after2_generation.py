import torch
from test_smolvlm_pruned_decoder import fixture


def test_after2_generation_allkeep_and_fixed_pruned_reference():
    from smolvlm_map_predictor_after2.model import MapPredictor
    from smolvlm_map_predictor_after2.generation import PredictorGreedyDecoder
    model,batch=fixture()
    predictor=MapPredictor(model.native.config.text_config,hidden_dim=8,projection_dim=4,visual_tokens=24).eval()
    full=PredictorGreedyDecoder(model,predictor,keep_tokens=24)
    result=full.generate(batch,max_new_tokens=4,debug=True)
    native=model.generate_from_features(**{k:batch[k] for k in ('input_ids','attention_mask','visual_features')},max_new_tokens=4,num_beams=1,do_sample=False)
    assert [r['generated_ids'] for r in result] == native.tolist()
    assert full.last_debug['predictor_calls'] == 1
    engine=PredictorGreedyDecoder(model,predictor,keep_tokens=12)
    rng=torch.get_rng_state().clone()
    calls=[]
    h=predictor.register_forward_hook(lambda m,a,y:calls.append(1))
    result=engine.generate(batch,max_new_tokens=4,debug=True)
    h.remove()
    assert calls == [1]
    assert torch.equal(rng,torch.get_rng_state())
    assert engine.last_debug['visual_counts_by_layer'] == [[24,24]]*3+[[12,12]]
    reference=engine.generate_reference(batch,max_new_tokens=4)
    assert result == reference
    assert all(len(r['visual_indices'])==12 for r in result)

def test_generation_rejects_training_and_preserves_backbone_weights():
    import pytest
    from smolvlm_map_predictor_after2.model import MapPredictor
    from smolvlm_map_predictor_after2.generation import PredictorGreedyDecoder
    model,batch=fixture()
    predictor=MapPredictor(model.native.config.text_config,hidden_dim=8,projection_dim=4,visual_tokens=24).eval()
    engine=PredictorGreedyDecoder(model,predictor,12)
    before={k:v.clone() for k,v in model.state_dict().items()}
    model.train()
    with pytest.raises(ValueError,match='eval'): engine.generate(batch,1)
    model.eval()
    calls=[]
    hooks=[layer.register_forward_hook(lambda m,a,y,i=i:calls.append(i)) for i,layer in enumerate(model.native.model.text_model.layers)]
    engine.generate(batch,1,debug=True)
    for hook in hooks: hook.remove()
    assert calls == [0,1,2,3]
    info=engine.last_debug
    assert info['prefill_cache_lengths'][:3] == [batch['input_ids'].shape[1]]*3
    assert info['prefill_cache_lengths'][3] == batch['input_ids'].shape[1]-12
    for key,value in model.state_dict().items(): assert torch.equal(value,before[key])

def test_mixed_eos_masks_positions_and_budget_stop_with_real_head():
    from smolvlm_map_predictor_after2.model import MapPredictor
    from smolvlm_map_predictor_after2.generation import PredictorGreedyDecoder
    model,batch=fixture()
    predictor=MapPredictor(model.native.config.text_config,hidden_dim=8,projection_dim=4,visual_tokens=24).eval()
    engine=PredictorGreedyDecoder(model,predictor,12)
    states=[]
    handle=model.native.lm_head.register_forward_pre_hook(lambda m,a:states.append(a[0].detach().clone()))
    engine.generate(batch,1)
    handle.remove()
    h0,h1=states[0][:,0]
    direction=h0/h0.norm()-h1/h1.norm()
    assert h0@direction > 0 and h1@direction < 0
    model.native.lm_head.weight.zero_()
    model.native.lm_head.weight[0]=direction
    model.native.lm_head.weight[1]=-direction
    model.native.generation_config.eos_token_id=0
    calls=[]; steps=[]; fed=[]
    hooks=[predictor.register_forward_hook(lambda m,a,y:calls.append(1)),
        model.native.get_input_embeddings().register_forward_pre_hook(lambda m,a:fed.append(a[0].detach().clone()))]
    def inspect(module,args,kwargs):
        if args[0].shape[1]==1:
            steps.append((kwargs['position_ids'].detach().clone(),kwargs['attention_mask'].detach().clone()))
    hooks += [layer.register_forward_pre_hook(inspect,with_kwargs=True) for layer in model.native.model.text_model.layers]
    actual=engine.generate(batch,4,debug=True)
    for hook in hooks: hook.remove()
    assert calls == [1]
    reference_calls=[]
    hook=predictor.register_forward_hook(lambda m,a,y:reference_calls.append(1))
    reference=engine.generate_reference(batch,4)
    hook.remove()
    assert reference_calls == [1]
    assert actual == reference
    assert actual[0]['generated_ids']==[0] and actual[0]['stop_reason']=='eos'
    assert actual[1]['generated_token_count']>1
    assert steps
    lengths=batch['attention_mask'].sum(-1)
    for index,(positions,mask) in enumerate(steps):
        step=index//4
        assert torch.equal(positions[:,0],lengths+step)
        assert mask[0,0,0,-1] < -1e30 and mask[1,0,0,-1] == 0
    # Finished rows feed native PAD after their EOS; no new visible cache keys.
    if len(fed)>2:
        assert all(item[0,0] == model.native.generation_config.pad_token_id for item in fed[2:])
    # A real zero head deterministically selects token zero; choose another EOS
    # to exercise an uninterrupted budget independently of mixed-row dynamics.
    model.native.lm_head.weight.zero_()
    model.native.generation_config.eos_token_id=2
    exhausted=engine.generate(batch,3)
    assert all(row['generated_token_count']==3 and row['stop_reason']=='max_new_tokens' for row in exhausted)
    assert exhausted==engine.generate_reference(batch,3)

    model.native.generation_config.eos_token_id=0
    early=engine.generate(batch,4)
    assert all(row['generated_ids']==[0] and row['stop_reason']=='eos' for row in early)
    assert early==engine.generate_reference(batch,4)

def test_suffix_receives_exact_raw_block_two_gather_in_source_order():
    from smolvlm_map_predictor_after2.model import MapPredictor
    from smolvlm_map_predictor_after2.generation import PredictorGreedyDecoder
    model,batch=fixture()
    predictor=MapPredictor(model.native.config.text_config,hidden_dim=8,projection_dim=4,visual_tokens=24).eval()
    engine=PredictorGreedyDecoder(model,predictor,12)
    boundary=[]; suffix=[]
    def before_suffix(module,args,kwargs):
        suffix.append((args[0].detach().clone(),kwargs['position_ids'].detach().clone(),kwargs['attention_mask'].detach().clone()))
    hooks=[model.native.model.text_model.layers[2].register_forward_hook(lambda m,a,y:boundary.append(y[0].detach().clone())),
        model.native.model.text_model.layers[3].register_forward_pre_hook(before_suffix,with_kwargs=True)]
    records=engine.generate(batch,1)
    for hook in hooks: hook.remove()
    assert len(boundary)==len(suffix)==1
    hidden,pos,mask=suffix[0]
    for row,record in enumerate(records):
        valid=batch['attention_mask'][row].bool()
        image=torch.where((batch['input_ids'][row]==model.native.config.image_token_id)&valid)[0]
        selected=torch.tensor(record['visual_indices'])
        keep=valid.clone(); keep[image]=False; keep[image[selected]]=True
        columns=torch.where(keep)[0]
        start=hidden.shape[1]-len(columns)
        # Equality is exact: no predictor-transformed states, norm or residual.
        assert torch.equal(hidden[row,start:],boundary[0][row,columns])
        source_positions=(valid.long().cumsum(-1)-1).masked_fill(~valid,0)
        assert torch.equal(pos[row,start:],source_positions[columns])
        assert len(columns)==int(valid.sum())-12
