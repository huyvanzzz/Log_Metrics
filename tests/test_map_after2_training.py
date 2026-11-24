import json
import torch


def test_real_after2_training_resume_and_frozen_prefix_calls(tmp_path):
    from after2_support import make_inputs,tiny_config
    from smolvlm_map_predictor_after2.training import train
    model,receipt=make_inputs(tmp_path/'inputs'); cfg=tiny_config()
    original={k:v.clone() for k,v in model.state_dict().items()}
    full,resumed=tmp_path/'full',tmp_path/'resumed'
    a=train(cfg,receipt,None,full,cpu=True,backbone=model)
    train(cfg,receipt,None,resumed,cpu=True,backbone=model,stop_after_epoch=1)
    b=train(cfg,receipt,None,resumed,cpu=True,backbone=model,resume=resumed/'checkpoints/predictor_latest.pt')
    assert a==b and a['completed_epochs']==2 and a['optimizer_step']==4
    x=torch.load(full/'checkpoints/predictor_latest.pt',weights_only=False)
    y=torch.load(resumed/'checkpoints/predictor_latest.pt',weights_only=False)
    for key,value in x['model_state_dict'].items(): torch.testing.assert_close(value,y['model_state_dict'][key],rtol=0,atol=0)
    assert all(torch.equal(v,original[k]) for k,v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    rows=[json.loads(line) for line in (full/'debug/debug_events.jsonl').read_text(encoding='utf-8').splitlines()]
    steps=[row for row in rows if row['event']=='TRAIN_STEP']
    assert steps and all(row['prefix_block_calls'][0]>0 and len(set(row['prefix_block_calls']))==1 for row in steps)
    assert all(row['suffix_block_calls']==0 for row in steps)
    assert 'decoder_calls' not in rows[0]
    assert x['identity']['method']['id']=='map_after2_lastprompt_direct_v1'
    events=[json.loads(line) for line in (full/'debug/train_progress.jsonl').read_text(encoding='utf-8').splitlines()]
    for epoch_end in (e for e in events if e['event']=='epoch_end'):
        losses=[e['loss'] for e in events if e['event']=='train_step' and e['epoch']==epoch_end['epoch']]
        assert epoch_end['train_loss']==sum(losses)/len(losses)
        assert epoch_end['sample_count']==3
    assert [e['checkpoint_kind'] for e in events if e['event']=='checkpoint_saved'][:2]==['best','latest']
    assert not any(module._forward_pre_hooks for module in model.modules())


def test_after2_guards_allow_only_prefix_and_restore_hooks(tmp_path):
    import pytest
    from after2_support import make_inputs
    from smolvlm_map_predictor_after2.training import forbid_backbone_forward
    model,_=make_inputs(tmp_path)
    guards=forbid_backbone_forward(model)
    try:
        with pytest.raises(AssertionError,match='suffix|Suffix'):
            model.native.model.text_model.layers[3](torch.zeros(1,2,16))
        assert guards.prefix_counts==[0,0,0]
    finally:
        for handle in guards: handle.remove()
    assert not model.native.model.text_model.layers[3]._forward_pre_hooks


def test_real_partial_epoch_cursor_resume_matches_weights(tmp_path):
    from after2_support import make_inputs,tiny_config
    from smolvlm_map_predictor_after2.training import train
    model, receipt = make_inputs(tmp_path)
    cfg = tiny_config()
    full, resumed = tmp_path/'full', tmp_path/'resumed'
    train(cfg,receipt,None,full,cpu=True,backbone=model)
    state = train(cfg,receipt,None,resumed,cpu=True,backbone=model,max_updates=1)
    assert state['epoch'] == 0 and state['batch_cursor'] == 1
    train(cfg,receipt,None,resumed,cpu=True,backbone=model,
          resume=resumed/'checkpoints/predictor_latest.pt')
    a = torch.load(full/'checkpoints/predictor_latest.pt',weights_only=False)
    b = torch.load(resumed/'checkpoints/predictor_latest.pt',weights_only=False)
    assert a['metadata']['optimizer_step'] == b['metadata']['optimizer_step'] == 4
    for key,value in a['model_state_dict'].items():
        torch.testing.assert_close(value,b['model_state_dict'][key],rtol=0,atol=0)


def test_improved_best_is_published_before_latest_metadata(tmp_path,monkeypatch):
    import pytest
    from after2_support import make_inputs,tiny_config
    from smolvlm_map_predictor_after2 import checkpoint
    from smolvlm_map_predictor_after2.training import train
    model,receipt = make_inputs(tmp_path)
    cfg = tiny_config(); out = tmp_path/'output'
    save = checkpoint.save
    def interrupted(path,*args,**kwargs):
        if path.name == 'predictor_latest.pt':
            assert (out/'checkpoints/predictor_best.pt').is_file()
            raise RuntimeError('interrupted before latest publication')
        return save(path,*args,**kwargs)
    monkeypatch.setattr(checkpoint,'save',interrupted)
    with pytest.raises(RuntimeError,match='before latest'):
        train(cfg,receipt,None,out,cpu=True,backbone=model,stop_after_epoch=1)
    assert not (out/'checkpoints/predictor_latest.pt').exists()
    best = torch.load(out/'checkpoints/predictor_best.pt',weights_only=False)
    assert best['metadata']['best_epoch'] == 1
