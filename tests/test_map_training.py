from copy import deepcopy
import json
import torch


def test_accumulation_tail_matches_mean_microbatch_reference():
    from smolvlm_map_predictor.training import loss_for_backward, window_boundary, window_size
    values = [torch.tensor(x, requires_grad=True) for x in (1., 2., 3.)]
    gradients = []
    for i, x in enumerate(values):
        (loss_for_backward(x*x, 2, window_size(i, 3, 2))/2).backward()
        gradients.append(x.grad.item())
    assert gradients == [1., 2., 6.]
    assert [window_boundary(i, 3, 2) for i in range(3)] == [False, True, True]


def test_progress_schema_flat_cpu_and_interval_clock(tmp_path):
    from smolvlm_map_predictor.training import ProgressWriter, memory
    writer = ProgressWriter(tmp_path)
    writer.write('train_step', loss=.4, elapsed_seconds=2., steps_per_second=.5,
                 estimated_epoch_remaining_seconds=10., **memory(torch.device('cpu')))
    ProgressWriter(tmp_path, enabled=False).write('ignored')
    rows = [json.loads(x) for x in writer.path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]['event'] == 'train_step' and rows[0]['timestamp']
    assert 'payload' not in rows[0] and rows[0]['cuda_memory_allocated'] is None


def test_production_recipe_is_separate_and_fixed():
    from smolvlm_map_predictor.config import load_config, recipe
    cfg = load_config()
    assert type(cfg['training']['epochs']) is int and cfg['training']['epochs'] > 0
    assert cfg['training']['batch_size'] == 32
    assert cfg['evaluation']['batch_size'] == 64
    assert recipe(cfg)['validation'] == 'sample_weighted_kl'
    assert cfg['training']['prefetch_factor'] == 1


def tiny_recipe():
    from smolvlm_map_predictor.config import load_config
    cfg = load_config()
    cfg['architecture'].update(hidden_dim=8, projection_dim=4)
    cfg['training'].update(world_size=1, batch_size=2, epochs=2, num_workers=0,
                           persistent_workers=False, pin_memory=False, progress_log_every_steps=1)
    return cfg


def test_real_training_epoch_resume_matches_continuous(tmp_path):
    from map_support import make_inputs
    from smolvlm_map_predictor.training import train
    (tmp_path/'inputs').mkdir()
    model, receipt = make_inputs(tmp_path/'inputs')
    original = {k: v.clone() for k,v in model.state_dict().items()}
    cfg = tiny_recipe()
    full = tmp_path/'full'; resumed = tmp_path/'resumed'
    a = train(cfg, receipt, None, full, cpu=True, backbone=model)
    train(cfg, receipt, None, resumed, cpu=True, backbone=model, stop_after_epoch=1)
    b = train(cfg, receipt, None, resumed, cpu=True, backbone=model,
              resume=resumed/'checkpoints/predictor_latest.pt')
    x = torch.load(full/'checkpoints/predictor_latest.pt', weights_only=False)
    y = torch.load(resumed/'checkpoints/predictor_latest.pt', weights_only=False)
    assert a == b
    assert a['completed_epochs'] == 2 and a['optimizer_step'] == 4
    for k,v in x['model_state_dict'].items():
        torch.testing.assert_close(v, y['model_state_dict'][k], rtol=0, atol=0)
    for k,v in model.state_dict().items():
        assert torch.equal(v, original[k])
    assert all(p.grad is None for p in model.parameters())
    events = [json.loads(line) for line in (resumed/'debug/train_progress.jsonl').read_text().splitlines()]
    assert [e['epoch'] for e in events if e['event'] == 'epoch_end'] == [1,2]
    assert [e['checkpoint_kind'] for e in events if e['event'] == 'checkpoint_saved'][:2] == ['best','latest']
    for end in (e for e in events if e['event'] == 'epoch_end'):
        steps = [e['loss'] for e in events if e['event'] == 'train_step' and e['epoch'] == end['epoch']]
        assert end['train_loss'] == sum(steps)/len(steps)
    assert all('section' not in e for e in events)


def test_real_partial_epoch_cursor_resume_matches_weights(tmp_path):
    from map_support import make_inputs
    from smolvlm_map_predictor.training import train
    model, receipt = make_inputs(tmp_path)
    cfg = tiny_recipe()
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


def test_real_predictor_accumulation_full_and_tail_gradient_reference():
    from transformers import LlamaConfig
    from smolvlm_map_predictor.model import MapPredictor,kl_per_sample
    from smolvlm_map_predictor.training import accelerator_for,loss_for_backward,window_size,window_boundary
    from smolvlm_map_predictor.config import load_config
    cfg = load_config(); cfg['training']['gradient_accumulation_steps'] = 2
    accelerator = accelerator_for(cfg,cpu=True)
    predictor = MapPredictor(LlamaConfig(hidden_size=8,num_attention_heads=2),hidden_dim=8,projection_dim=4,visual_tokens=3)
    reference = deepcopy(predictor)
    batches = [dict(visual=torch.randn(1,3,8),question=torch.randn(1,2,8),
                   question_mask=torch.tensor([[True,True]]),visual_positions=torch.tensor([[0,1,2]]),
                   question_positions=torch.tensor([[4,5]])) for _ in range(3)]
    target = torch.tensor([[.1,.2,.7]]).log()
    for index,batch in enumerate(batches):
        size = window_size(index,len(batches),2)
        actual_loss = kl_per_sample(predictor(**batch),target).mean()
        accelerator.backward(loss_for_backward(actual_loss,2,size))
        (kl_per_sample(reference(**batch),target).mean()/size).backward()
        if window_boundary(index,len(batches),2):
            for a,b in zip(predictor.parameters(),reference.parameters()):
                torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
            predictor.zero_grad(); reference.zero_grad()


def test_improved_best_is_published_before_latest_metadata(tmp_path,monkeypatch):
    import pytest
    from map_support import make_inputs
    from smolvlm_map_predictor import checkpoint
    from smolvlm_map_predictor.training import train
    model,receipt = make_inputs(tmp_path)
    cfg = tiny_recipe(); out = tmp_path/'output'
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
