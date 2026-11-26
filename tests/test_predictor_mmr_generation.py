import json
import pytest
import torch
from test_smolvlm_pruned_decoder import fixture
from smolvlm_predictor_mmr.generation import Decoder


@pytest.mark.parametrize('variant',['before','after2'])
def test_cached_reference_allkeep_and_once(variant):
    if variant=='before': from smolvlm_map_predictor.model import MapPredictor
    else: from smolvlm_map_predictor_after2.model import MapPredictor
    model,batch=fixture()
    predictor=MapPredictor(model.native.config.text_config,hidden_dim=8,projection_dim=4,visual_tokens=24).eval()
    calls=[]; captured=[]
    hook=predictor.register_forward_hook(lambda *args:calls.append(1))
    decoder=Decoder(model,predictor,variant,12)
    original=decoder.selector
    class Capture:
        def __call__(self,scores,features,count):
            captured.append(features.clone()); return original(scores,features,count)
    capture=Capture()
    if variant=='after2': decoder.engine.selection_fn=capture
    else: decoder.selector=capture
    # Keep diagnostics via the original selector after using feature instrumentation.
    if variant=='before': capture.last_debug=original.last_debug
    boundary=[]
    if variant=='after2':
        handle=model.native.model.text_model.layers[2].register_forward_hook(lambda m,a,y:boundary.append(y[0].clone()))
    else:
        from smolvlm_map_predictor_after2.prefix import fuse_inputs
        fused=fuse_inputs(model,batch)
        boundary=[fused['inputs_embeds']]
    rng=torch.get_rng_state().clone(); weights={k:v.clone() for k,v in model.state_dict().items()}
    result=decoder.generate(batch,3)
    hook.remove()
    if variant=='after2': handle.remove()
    assert calls==[1] and len(captured)==1
    image=(batch['input_ids']==model.native.config.image_token_id)&batch['attention_mask'].bool()
    for row in range(2): torch.testing.assert_close(captured[0][row],boundary[0][row,image[row]],rtol=0,atol=0)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(value,weights[key]) for key,value in model.state_dict().items())
    # Restore selector for serializable diagnostics.
    decoder.selector=original
    if variant=='after2': decoder.engine.selection_fn=original
    assert result==decoder.generate_reference(batch,3,debug=True)
    json.dumps(decoder.last_debug)
    allkeep=Decoder(model,predictor,variant,24).generate(batch,3)
    native=model.generate_from_features(**{k:batch[k] for k in ('input_ids','attention_mask','visual_features')},max_new_tokens=3,num_beams=1,do_sample=False)
    assert [row['generated_ids'] for row in allkeep]==native.tolist()
    model.native.lm_head.weight.zero_(); model.native.generation_config.eos_token_id=0
    assert all(row['generated_ids']==[0] for row in decoder.generate(batch,4))

@pytest.mark.parametrize('variant',['before','after2'])
def test_projected_inputs_match_raw_and_guards(variant):
    if variant=='before': from smolvlm_map_predictor.model import MapPredictor
    else: from smolvlm_map_predictor_after2.model import MapPredictor
    from smolvlm_predictor_io.projected import project_native
    model,batch=fixture()
    predictor=MapPredictor(model.native.config.text_config,hidden_dim=8,projection_dim=4,visual_tokens=24).eval()
    projected=dict(batch)
    projected['projected_visual_features']=torch.from_numpy(project_native(model,batch['visual_features'])).reshape(len(batch['input_ids']),24,-1)
    del projected['visual_features']
    decoder=Decoder(model,predictor,variant,12)
    raw=decoder.generate(batch,3)
    assert decoder.generate(projected,3)==raw
    predictor.train()
    with pytest.raises(ValueError,match='eval'): decoder.generate(batch,1)
    predictor.eval()
    with torch.autocast('cpu',dtype=torch.bfloat16):
        with pytest.raises(ValueError,match='FP32'): decoder.generate(batch,1)


@pytest.mark.parametrize('variant',['before','after2'])
def test_malformed_selection_rejected(variant):
    if variant=='before': from smolvlm_map_predictor.model import MapPredictor
    else: from smolvlm_map_predictor_after2.model import MapPredictor
    from smolvlm_map_predictor.evaluation import predict_batch
    from smolvlm_map_predictor_after2.generation import PredictorGreedyDecoder
    model,batch=fixture()
    predictor=MapPredictor(model.native.config.text_config,hidden_dim=8,projection_dim=4,visual_tokens=24).eval()
    wrong=lambda scores,states,count: torch.zeros((len(scores),count-1),dtype=torch.long)
    with pytest.raises(ValueError,match='Selection'):
        if variant=='before': predict_batch(model,predictor,batch,keep_tokens=12,max_new_tokens=1,selection_fn=wrong)
        else: PredictorGreedyDecoder(model,predictor,12,selection_fn=wrong).generate(batch,1)


