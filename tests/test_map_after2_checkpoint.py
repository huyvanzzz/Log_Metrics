from copy import deepcopy
import pytest
import torch


def test_checkpoint_requires_after2_method_and_execution_identity(tmp_path,monkeypatch):
    from after2_support import make_inputs,tiny_config
    from smolvlm_map_predictor_after2.training import train,make_predictor
    from smolvlm_map_predictor_after2 import checkpoint
    from smolvlm_map_predictor_after2.config import recipe
    from smolvlm_map_predictor.model import MapPredictor as BeforePredictor
    from smolvlm_map_predictor_after2 import runtime
    model,receipt=make_inputs(tmp_path/'inputs'); cfg=tiny_config()
    out=tmp_path/'out'
    train(cfg,receipt,None,out,cpu=True,backbone=model,stop_after_epoch=1)
    path=out/'checkpoints/predictor_latest.pt'
    payload=torch.load(path,weights_only=False); identity=payload['identity']
    predictor=make_predictor(model,cfg)
    assert payload['recipe']['execution']==dict(sources=runtime.sources(),versions=runtime.versions())
    before=BeforePredictor(model.native.config.text_config,hidden_dim=8,projection_dim=4,visual_tokens=384)
    assert {k:tuple(v.shape) for k,v in predictor.state_dict().items()}=={k:tuple(v.shape) for k,v in before.state_dict().items()}
    with pytest.raises(ValueError,match='identity'):
        checkpoint.read(path,before,identity)
    checkpoint.read(path,predictor,identity,training=True,recipe=recipe(cfg),world_size=1)
    original=runtime.versions
    monkeypatch.setattr(runtime,'versions',lambda:dict(original(),torch='different-library'))
    with pytest.raises(ValueError,match='recipe/world'):
        checkpoint.read(path,predictor,identity,training=True,recipe=recipe(cfg),world_size=1)
    monkeypatch.setattr(runtime,'versions',original)
    source=runtime.sources
    monkeypatch.setattr(runtime,'sources',lambda:dict(source(),extra='changed-code'))
    with pytest.raises(ValueError,match='recipe/world'):
        checkpoint.read(path,predictor,identity,training=True,recipe=recipe(cfg),world_size=1)
    # Inference requires weights/method/input identity, not training execution or RNG.
    checkpoint.read(path,predictor,identity)
