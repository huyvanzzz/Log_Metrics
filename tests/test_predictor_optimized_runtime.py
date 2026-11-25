from pathlib import Path
import json

def test_optimized_configs_have_positive_epochs_and_separate_outputs():
    from smolvlm_map_predictor.config import load_config
    from smolvlm_map_predictor_after2.config import load_config as after
    for cfg in (load_config(),after()):
        assert type(cfg['training']['epochs']) is int and cfg['training']['epochs'] > 0
        assert '_optimized_' in cfg['output_dir'] and cfg['output_dir'].endswith('_outputs')

def test_progress_writes_files_without_console_json(tmp_path,capsys):
    from smolvlm_map_predictor.training import ProgressWriter
    writer=ProgressWriter(tmp_path)
    writer.write('train_step',global_step=1,step_in_epoch=1,total_steps_epoch=10,epoch=1,loss=.2)
    assert capsys.readouterr().out == ''
    assert json.loads(writer.path.read_text(encoding='utf-8'))['loss'] == .2

def test_projected_preparation_does_not_require_raw_payload(tmp_path):
    import numpy as np
    from map_support import make_inputs
    from smolvlm_map_predictor.data import prepare_inputs
    from smolvlm_predictor_io.projected import build_cache,finalize_cache,project_native
    model, receipt=make_inputs(tmp_path)
    info=json.loads(receipt.read_text(encoding='utf-8')); raw=Path(info['cache_dir'])
    descriptor=json.loads((tmp_path/'fixture_validation/cache_validation.json').read_text(encoding='utf-8')) if (tmp_path/'fixture_validation/cache_validation.json').exists() else None
    from smolvlm_attention.teacher_targets.runtime import validate_teacher_cache
    descriptor=json.loads(validate_teacher_cache(raw,tmp_path/'prepared',tmp_path/'project_validation').read_text(encoding='utf-8'))
    identity=dict(raw_cache_manifest=json.loads((raw/'manifest.json').read_text(encoding='utf-8')),raw_cache_descriptor=descriptor,
                  checkpoint_sha256=info['identity']['checkpoint_sha256'],model=model.architecture_signature)
    index=json.loads((raw/'index.json').read_text(encoding='utf-8')); shape=tuple(info['shape'])
    array=np.memmap(raw/'features.float16.memmap',mode='r',dtype='<f2',shape=shape)
    dest=tmp_path/'projected'
    build_cache(array,index,dest,lambda x:project_native(model,x),identity,batch_size=2,projected_shape=(6,64,model.geometry['text_dim']))
    finalize_cache(dest)
    del array
    (raw/'features.float16.memmap').rename(raw/'hidden.memmap')
    result=prepare_inputs(tmp_path/'targets',None,tmp_path/'prepared',model,tmp_path/'optimized_inputs',
       expected_counts=info['counts'],checkpoint_sha256=info['identity']['checkpoint_sha256'],projected_cache=dest)
    data=json.loads(result.read_text(encoding='utf-8'))
    assert data['projected_cache_dir']==str(dest.resolve())
    assert str(raw/'features.float16.memmap') not in data['files']

def test_compact_training_both_variants_resumes_with_single_pool(tmp_path):
    import numpy as np,torch
    from map_support import make_inputs
    from smolvlm_predictor_io.projected import build_cache,finalize_cache,project_native
    from smolvlm_predictor_io.compact import prepare_store
    from smolvlm_map_predictor.training import train
    from smolvlm_map_predictor_after2.training import train as after_train
    from smolvlm_map_predictor_after2.data import adapt_receipt
    from test_map_training import tiny_recipe
    model, receipt=make_inputs(tmp_path)
    info=json.loads(receipt.read_text(encoding='utf-8')); raw=Path(info['cache_dir']); index=json.loads((raw/'index.json').read_text())
    array=np.memmap(raw/'features.float16.memmap',mode='r',dtype='<f2',shape=tuple(info['shape']))
    build_cache(array,index,tmp_path/'projected',lambda x:project_native(model,x),{},batch_size=2,projected_shape=(6,64,model.geometry['text_dim']))
    finalize_cache(tmp_path/'projected')
    for name,fn,source in [('before',train,receipt),('after',after_train,adapt_receipt(receipt))]:
        store=prepare_store(source,model.processor,tmp_path/name,projected_path=tmp_path/'projected')
        cfg=tiny_recipe()
        if name=='after':
            from smolvlm_map_predictor_after2.config import METHOD
            cfg['method']=METHOD
        full=tmp_path/(name+'_full'); resumed=tmp_path/(name+'_resume')
        fn(cfg,store,None,full,cpu=True,backbone=model)
        fn(cfg,store,None,resumed,cpu=True,backbone=model,stop_after_epoch=1)
        fn(cfg,store,None,resumed,cpu=True,backbone=model,resume=resumed/'checkpoints/predictor_latest.pt')
        a=torch.load(full/'checkpoints/predictor_latest.pt',weights_only=False)
        b=torch.load(resumed/'checkpoints/predictor_latest.pt',weights_only=False)
        for key,value in a['model_state_dict'].items(): torch.testing.assert_close(value,b['model_state_dict'][key],rtol=0,atol=0)

def test_status_is_throttled_and_file_only(tmp_path,monkeypatch,capsys):
    import smolvlm_predictor_io.progress as progress
    clock=[1.]
    monkeypatch.setattr(progress.time,'monotonic',lambda:clock[0])
    writer=progress.StatusWriter(tmp_path,0)
    writer.update('train',0,10,epoch=1)
    writer.update('train',1,10,epoch=1)
    path=tmp_path/'status/rank_0.json'
    assert json.loads(path.read_text())['completed']==0
    clock[0]=3.1
    writer.update('train',2,10,epoch=1)
    assert json.loads(path.read_text())['completed']==2
    assert capsys.readouterr().out==''

def test_progress_does_not_replay_lagging_rank_phase():
    from smolvlm_predictor_io.progress import latest_phase
    rows=[dict(phase='train',rank=0,epoch=1),dict(phase='validation',rank=1,epoch=1)]
    assert latest_phase(rows)==[rows[1]]
    rows[0]['epoch']=2
    assert latest_phase(rows)==[rows[0]]

def test_epoch_count_configurable_for_both_predictors(tmp_path):
    from smolvlm_map_predictor.config import load_config
    from smolvlm_map_predictor_after2.config import load_config as after
    for loader in (load_config, after):
        for epochs in (1, 5, 10, 12):
            cfg=loader(); cfg['training']['epochs']=epochs
            path=tmp_path/'config.json'; path.write_text(json.dumps(cfg))
            assert loader(path)['training']['epochs']==epochs


def test_epoch_count_must_be_positive_integer(tmp_path):
    import pytest
    from smolvlm_map_predictor.config import load_config
    from smolvlm_map_predictor_after2.config import load_config as after
    for loader in (load_config, after):
        for epochs in (0, -1, True, 5.0, '5', None):
            cfg=loader(); cfg['training']['epochs']=epochs
            path=tmp_path/'config.json'; path.write_text(json.dumps(cfg))
            with pytest.raises(ValueError): loader(path)


def test_predictor_input_checks_use_one_scalar_read():
    import torch
    from transformers import LlamaConfig
    from smolvlm_map_predictor.model import MapPredictor
    model=MapPredictor(LlamaConfig(hidden_size=8,num_attention_heads=2),hidden_dim=8,projection_dim=4,visual_tokens=3)
    inputs=dict(visual=torch.ones(2,3,8),question=torch.ones(2,2,8),question_mask=torch.ones(2,2,dtype=torch.bool),
                visual_positions=torch.tensor([[0,1,2]]*2),question_positions=torch.tensor([[4,5]]*2))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        model(**inputs)
    scalar_reads=sum(event.count for event in profile.key_averages() if event.key=='aten::_local_scalar_dense')
    assert scalar_reads==1
