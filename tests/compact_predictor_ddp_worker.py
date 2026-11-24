"""Two-rank Gloo compact-loader integration; never use a production checkpoint."""
import importlib
import json
import os
from pathlib import Path
import sys
import torch


def main():
    root=Path(sys.argv[1]); package=sys.argv[2]
    if os.name=='nt':
        torch.distributed.init_process_group('gloo',init_method=(root/'rendezvous').as_uri(),rank=int(os.environ['RANK']),world_size=2)
    training=importlib.import_module(package+'.training'); model_module=importlib.import_module(package+'.model')
    cfg=importlib.import_module(package+'.config').load_config()
    cfg['architecture'].update(hidden_dim=8,projection_dim=4)
    cfg['training'].update(world_size=2,batch_size=2,num_workers=0,pin_memory=False,epochs=2,persistent_workers=False)
    model=torch.load(root/'fixture_model.pt',map_location='cpu',weights_only=False)
    calls=[]; model.native.model.connector.register_forward_hook(lambda *args:calls.append(1))
    receipt=root/'compact/store.json'
    training.train(cfg,receipt,None,root/'full',cpu=True,backbone=model)
    training.train(cfg,receipt,None,root/'resumed',cpu=True,backbone=model,stop_after_epoch=1)
    result=training.train(cfg,receipt,None,root/'resumed',cpu=True,backbone=model,resume=root/'resumed/checkpoints/predictor_latest.pt')
    from smolvlm_predictor_io.loader import PersistentPool
    accelerator=training.accelerator_for(cfg,cpu=True)
    torch.manual_seed(17)
    predictor=training.make_predictor(model,cfg); optimizer=torch.optim.AdamW(predictor.parameters())
    predictor,optimizer=accelerator.prepare(predictor,optimizer)
    pool=PersistentPool(receipt,rank=accelerator.process_index,world_size=2,batch_size=2,num_workers=0,pin_memory=False)
    train,val=pool.views()
    try:
        iterator=iter(train); batch=training.move_batch(next(iterator),accelerator.device)
        # Explicitly exhaust before val so no stale phase requests may survive.
        assert next(iterator,None) is None
        frozen=importlib.import_module(package+'.prefix').frozen_features if package.endswith('after2') else model_module.frozen_features
        loss=model_module.kl_per_sample(predictor(**frozen(model,batch)),batch['log_target']).mean()
        accelerator.backward(loss)
        gradient=torch.cat([p.grad.flatten() for p in accelerator.unwrap_model(predictor).parameters()]); grads=[torch.empty_like(gradient) for _ in range(2)]
        torch.distributed.all_gather(grads,gradient); assert torch.equal(grads[0],grads[1])
        optimizer.step(); weights=torch.cat([p.detach().flatten() for p in accelerator.unwrap_model(predictor).parameters()]); states=[torch.empty_like(weights) for _ in range(2)]
        torch.distributed.all_gather(states,weights); assert torch.equal(states[0],states[1])
        ids=[None,None]; torch.distributed.all_gather_object(ids,batch['sample_ids'])
        valid=[sid for values in val for sid in values['sample_ids']]; vids=[None,None]; torch.distributed.all_gather_object(vids,valid)
        assert len(sum(ids,[]))==4 and len(set(sum(ids,[])))==3
        assert len(sum(vids,[]))==len(set(sum(vids,[])))==3
        assert not calls and all(p.grad is None for p in model.parameters())
        result.update(gradient_ranks_equal=True,weights_ranks_equal=True,sample_ids=ids,validation_ids=vids,connector_calls=len(calls))
        (root/f'worker_{os.environ["RANK"]}.json').write_text(json.dumps(result))
    finally: pool.close()


if __name__=='__main__': main()
