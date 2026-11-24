import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import numpy as np
import pytest
import torch


@pytest.mark.parametrize('package',['smolvlm_map_predictor','smolvlm_map_predictor_after2'])
def test_compact_two_rank_training_rng_resume_and_validation(tmp_path,package):
    from map_support import make_inputs
    from smolvlm_predictor_io.compact import prepare_store
    assert torch.distributed.is_gloo_available()
    model,receipt=make_inputs(tmp_path)
    if package.endswith('after2'):
        from smolvlm_map_predictor_after2.data import adapt_receipt
        receipt=adapt_receipt(receipt)
    info=json.loads(receipt.read_text(encoding='utf-8')); raw=np.memmap(Path(info['cache_dir'])/'features.float16.memmap',mode='r',dtype='<f2',shape=tuple(info['shape']))
    projected_dir=tmp_path/'projected'; projected_dir.mkdir()
    with torch.no_grad(): projected=model._project(torch.tensor(np.array(raw)),len(raw)).reshape(len(raw),6,64,-1).numpy()
    np.save(projected_dir/'projected.npy',projected); (projected_dir/'manifest.json').write_text(json.dumps({'shape':list(projected.shape)}))
    prepare_store(receipt,model.processor,tmp_path/'compact',projected_path=projected_dir)
    torch.save(model,tmp_path/'fixture_model.pt')
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([os.getcwd(),str(Path.cwd()/'tests')]),USE_TF='0',USE_LIBUV='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    if os.name=='nt':
        processes=[]
        with socket.socket() as sock: sock.bind(('127.0.0.1',0)); port=str(sock.getsockname()[1])
        try:
            for rank in range(2):
                child=dict(env,RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE='2',LOCAL_WORLD_SIZE='2',MASTER_ADDR='127.0.0.1',MASTER_PORT=port)
                processes.append(subprocess.Popen([sys.executable,'tests/compact_predictor_ddp_worker.py',str(tmp_path),package],env=child,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True))
            outputs=[p.communicate(timeout=180)[0] for p in processes]
            assert all(p.returncode==0 for p in processes),'\n'.join(outputs)
        finally:
            for p in processes:
                if p.poll() is None: p.terminate()
            for p in processes: p.wait(timeout=15)
    else:
        run=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','tests/compact_predictor_ddp_worker.py',str(tmp_path),package],env=env,capture_output=True,text=True,timeout=180)
        assert run.returncode==0,run.stdout+run.stderr
    full=torch.load(tmp_path/'full/checkpoints/predictor_latest.pt',weights_only=False)
    resumed=torch.load(tmp_path/'resumed/checkpoints/predictor_latest.pt',weights_only=False)
    assert {k:v for k,v in full['metadata'].items() if k!='checkpoint_id'}=={k:v for k,v in resumed['metadata'].items() if k!='checkpoint_id'}
    assert full['metadata']['global_step']==full['metadata']['optimizer_step']==2
    assert full['scheduler_state_dict']==resumed['scheduler_state_dict']
    for key,value in full['model_state_dict'].items(): torch.testing.assert_close(value,resumed['model_state_dict'][key],rtol=0,atol=0)
    for rank in range(2):
        saved_rng=[]
        for label,payload in [('full',full),('resumed',resumed)]:
            side=tmp_path/label/'checkpoints/predictor_latest.pt.rng'/payload['metadata']['checkpoint_id']/f'rank_{rank:05d}.pt'
            saved_rng.append(torch.load(side,weights_only=False)['rng'])
        assert torch.equal(saved_rng[0]['torch'],saved_rng[1]['torch'])
        assert saved_rng[0]['python']==saved_rng[1]['python']
        np.testing.assert_array_equal(saved_rng[0]['numpy'][1],saved_rng[1]['numpy'][1])
        result=json.loads((tmp_path/f'worker_{rank}.json').read_text())
        assert result['connector_calls']==0 and result['gradient_ranks_equal'] and result['weights_ranks_equal']
        assert result['val_loss']==full['metadata']['val_loss']
    events=[json.loads(line) for line in (tmp_path/'full/debug/train_progress.jsonl').read_text().splitlines()]
    assert all(row['sample_count']==3 for row in events if row['event']=='validation_end')
