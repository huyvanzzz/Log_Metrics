"""Real native greedy evaluation through the compact projected input path."""
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch


@pytest.mark.parametrize('after2',[False,True])
def test_compact_real_evaluation_resume_and_finalize_without_targets(tmp_path,after2):
    from map_support import make_inputs
    from smolvlm_predictor_io.compact import prepare_store
    model,receipt=make_inputs(tmp_path)
    package='smolvlm_map_predictor_after2' if after2 else 'smolvlm_map_predictor'
    if after2:
        from smolvlm_map_predictor_after2.data import adapt_receipt
        receipt=adapt_receipt(receipt)
    info=json.loads(receipt.read_text(encoding='utf-8'))
    raw=np.memmap(Path(info['cache_dir'])/'features.float16.memmap',mode='r',dtype='<f2',shape=tuple(info['shape']))
    directory=tmp_path/'projected'; directory.mkdir()
    with torch.no_grad(): projected=model._project(torch.tensor(np.array(raw)),len(raw)).reshape(len(raw),6,64,-1).numpy()
    np.save(directory/'projected.npy',projected)
    (directory/'manifest.json').write_text(json.dumps({'shape':list(projected.shape)}))
    compact=prepare_store(receipt,model.processor,tmp_path/'compact',projected_path=directory)
    config=importlib.import_module(package+'.config').load_config()
    config['architecture'].update(hidden_dim=8,projection_dim=4)
    config['training'].update(world_size=1,batch_size=2,epochs=2,num_workers=0,pin_memory=False,persistent_workers=False)
    config['evaluation'].update(batch_size=2,keep_tokens=4,max_new_tokens=2)
    from smolvlm_attention.sample_store import atomic_json,digest
    runtime=importlib.import_module(package+'.runtime')
    output=tmp_path/'output'
    run=dict(version=1,receipt=str(compact),receipt_sha256=digest(compact),identity=info['identity'],
        config=config,output_dir=str(output),backbone_checkpoint=str(tmp_path/'base_fixture.pt'),
        execution=dict(sources=runtime.sources(),versions=runtime.versions()),
        _evaluation_world_size=1,_evaluation_session='compact-cpu')
    run_path=output/'run.json';atomic_json(run_path,run)
    diagnose=importlib.import_module(package+'.diagnostics').diagnose
    assert diagnose(config,compact,None,output,cpu=True,backbone=model)['ok']
    train=importlib.import_module(package+'.training').train
    train(config,compact,None,output,cpu=True,backbone=model,stop_after_epoch=1)
    result=train(config,compact,None,output,cpu=True,backbone=model,resume=output/'checkpoints/predictor_latest.pt')
    assert result['completed_epochs']==2
    checkpoint=output/'checkpoints/predictor_best.pt'
    for path in (tmp_path/'targets').rglob('*'):
        if path.is_file(): path.unlink()
    evaluation=importlib.import_module(package+'.evaluation')
    assert evaluation.evaluate_worker(run,0,checkpoint,device='cpu',backbone=model)['count']==3
    assert evaluation.evaluate_worker(run,0,checkpoint,device='cpu',backbone=model)['resumed_batches']==2
    summary=evaluation.finalize_evaluation(run,checkpoint)
    assert summary['diagnostics']['prediction_count']==3
    archive=runtime.bundle(run_path)
    import zipfile
    with zipfile.ZipFile(archive) as stream:
        names=stream.namelist()
        assert 'predictions.jsonl' in names and 'metrics.json' in names
        assert 'checkpoints/predictor_best.pt' in names
        assert any(name.startswith('checkpoints/predictor_latest.pt.rng/') for name in names)
        assert not any('projected.npy' in name or 'log_probs.npy' in name or name.startswith('internal/') for name in names)
