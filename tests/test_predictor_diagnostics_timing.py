import torch


def test_fixed_recipe_measures_real_updates_and_timing_fields():
    from smolvlm_predictor_io.diagnostics import measure_updates
    from accelerate import Accelerator
    p=torch.nn.Linear(3,2)
    opt=torch.optim.AdamW(p.parameters(),lr=.001)
    acc=Accelerator(cpu=True)
    p,opt=acc.prepare(p,opt)
    batch=dict(projected_visual_features=torch.ones(2,3),log_target=torch.full((2,2),-0.69314718),sample_ids=['a','b'])
    before=[v.clone() for v in p.parameters()]
    result=measure_updates(iter([batch]*23),p,opt,acc,lambda x:dict(input=x['projected_visual_features']),
        lambda m,x:m(x['input']).log_softmax(-1),lambda a,b:torch.nn.functional.kl_div(a,b,log_target=True,reduction='none').sum(-1),1.,warmup=3,measured=20)
    assert result['timing']['warmup_updates']==3 and result['timing']['measured_updates']==20
    assert result['timing']['measured_samples']==40
    assert result['timing']['pipeline_seconds']>0 and result['timing']['samples_per_second']>0
    assert result['timing']['h2d_cuda_seconds'] is None
    assert any(not torch.equal(a,b) for a,b in zip(before,p.parameters()))
    assert all(float(v['step'])==23 for v in opt.state.values())


def test_compact_diagnostic_both_variants_real_updates_and_rng(tmp_path):
    import importlib,json
    from pathlib import Path
    import numpy as np
    from map_support import make_inputs
    from smolvlm_predictor_io.compact import prepare_store
    model,receipt=make_inputs(tmp_path)
    info=json.loads(receipt.read_text(encoding='utf-8'))
    raw=np.memmap(Path(info['cache_dir'])/'features.float16.memmap',mode='r',dtype='<f2',shape=tuple(info['shape']))
    projected=tmp_path/'projected';projected.mkdir()
    with torch.no_grad(): values=model._project(torch.tensor(np.array(raw)),len(raw)).reshape(len(raw),6,64,-1).numpy()
    np.save(projected/'projected.npy',values)
    (projected/'manifest.json').write_text(json.dumps({'shape':list(values.shape)}))
    for package in ('smolvlm_map_predictor','smolvlm_map_predictor_after2'):
        current=receipt
        if package.endswith('after2'):
            current=importlib.import_module(package+'.data').adapt_receipt(receipt)
        store=prepare_store(current,model.processor,tmp_path/package/'compact',projected_path=projected)
        cfg=importlib.import_module(package+'.config').load_config()
        cfg['architecture'].update(hidden_dim=8,projection_dim=4)
        cfg['training'].update(world_size=1,batch_size=2,num_workers=0,pin_memory=False)
        state=torch.get_rng_state().clone()
        result=importlib.import_module(package+'.diagnostics').diagnose(cfg,store,None,tmp_path/package/'out',cpu=True,backbone=model)
        assert torch.equal(state,torch.get_rng_state())
        assert result['timing']['measured_updates']==20 and result['timing']['warmup_updates']==3
        assert result['input_feature_dtype']=='torch.float32'
        assert result['optimizer_state_count']==10
        if package.endswith('after2'): assert result['prefix_block_calls']==[23,23,23]

