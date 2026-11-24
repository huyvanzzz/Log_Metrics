import torch
from test_smolvlm_pruned_decoder import fixture


def projected_batch(model, batch):
    result = dict(batch)
    result['projected_visual_features'] = model._project(batch['visual_features'], 2).reshape(2, model.visual_token_len, -1)
    result.pop('visual_features')
    columns = [row.nonzero().flatten() for row in batch['question_mask']]
    width = max(map(len, columns))
    result['question_columns'] = torch.zeros(2, width, dtype=torch.long)
    result['question_valid'] = torch.zeros(2, width, dtype=torch.bool)
    for i, col in enumerate(columns):
        result['question_columns'][i,:len(col)] = col
        result['question_valid'][i,:len(col)] = True
    result['position_ids'] = batch['source_position_ids']
    result['last_query_columns'] = torch.where(batch['attention_mask'].bool(), torch.arange(batch['input_ids'].shape[1]), -1).max(-1).values
    return result


def test_projected_features_skip_connector_and_match_raw(monkeypatch):
    from smolvlm_map_predictor.model import frozen_features
    from smolvlm_map_predictor_after2.prefix import frozen_features as after_features
    model, batch = fixture()
    references = [fn(model,batch) for fn in (frozen_features,after_features)]
    optimized = projected_batch(model,batch)
    monkeypatch.setattr(model, '_project', lambda *a,**k: (_ for _ in ()).throw(AssertionError('connector called')))
    for fn, expected in zip((frozen_features,after_features),references):
        actual=fn(model,optimized)
        for key in expected:
            torch.testing.assert_close(actual[key],expected[key])
        assert not actual['visual'].is_inference()


def test_projected_generation_matches_raw_and_cached_reference(monkeypatch):
    from smolvlm_map_predictor.model import MapPredictor
    from smolvlm_map_predictor.evaluation import predict_batch
    from smolvlm_map_predictor_after2.model import MapPredictor as AfterPredictor
    from smolvlm_map_predictor_after2.generation import PredictorGreedyDecoder
    model,batch=fixture()
    pred=MapPredictor(model.native.config.text_config,hidden_dim=8,projection_dim=4,visual_tokens=24).eval()
    after=AfterPredictor(model.native.config.text_config,hidden_dim=8,projection_dim=4,visual_tokens=24).eval()
    raw=predict_batch(model,pred,batch,keep_tokens=5,max_new_tokens=3)
    engine=PredictorGreedyDecoder(model,after,5)
    raw_after=engine.generate(batch,max_new_tokens=3)
    optimized=projected_batch(model,batch)
    monkeypatch.setattr(model,'_project',lambda *a,**k: (_ for _ in ()).throw(AssertionError('connector called')))
    assert predict_batch(model,pred,optimized,keep_tokens=5,max_new_tokens=3)==raw
    assert engine.generate(optimized,max_new_tokens=3)==raw_after
    assert engine.generate_reference(optimized,max_new_tokens=3)==raw_after


def test_prefix_reuses_rotary_and_mask_without_extra_decoder_calls():
    from smolvlm_map_predictor_after2.prefix import fuse_inputs, run_prefix
    model,batch=fixture()
    rotary_calls=[]
    seen=[]
    hook=model.native.model.text_model.rotary_emb.register_forward_hook(lambda *args:rotary_calls.append(1))
    hooks=[layer.register_forward_pre_hook(lambda m,a,k:seen.append((k['attention_mask'], k['position_embeddings'])),with_kwargs=True) for layer in model.native.model.text_model.layers]
    try:
        run_prefix(model,fuse_inputs(model,projected_batch(model,batch)))
    finally:
        hook.remove()
        for h in hooks:h.remove()
    assert len(seen)==3 and len(rotary_calls)==1
    assert all(mask is seen[0][0] and rope is seen[0][1] for mask,rope in seen)
