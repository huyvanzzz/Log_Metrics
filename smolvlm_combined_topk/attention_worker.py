"""Per-sample combined attention worker; durable fixed-size batches, no OOM fallback."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from smolvlm_attention.sample_store import atomic_json,digest,shard_bounds


def process_pending(indices,store,measure,progress=None):
    remaining=[i for i in indices if i not in store.completed]
    for start in range(0,len(remaining),64):
        selected=remaining[start:start+64]
        try:
            values=measure(selected)
            if hasattr(values,'detach'):
                if values.device.type!='cpu':raise ValueError('Collector must return CPU scores')
                values=values.detach().numpy()
            values=np.asarray(values)
            if (values.dtype!=np.float32 or values.shape!=(len(selected),10,384)
                    or not np.isfinite(values).all() or (values<0).any()
                    or (values.sum(-1,dtype=np.float64)>1+1e-6).any()):
                raise ValueError('Invalid per-sample combined scores')
            store.write(selected,values)
            store.commit()
        except Exception as exc:
            raise RuntimeError(f'Fixed batch 64 failed, no retry/reduction; rows {selected}: {exc!r}') from exc
        if progress:
            progress(dict(stage='measuring',processed=len(store.completed),committed=len(store.completed),
                          batch_size=64,actual_batch_size=len(selected)))


def run_worker(config_path,shard):
    import torch
    from filelock import FileLock
    from smolvlm_inference.model import load_model,CAMERA_ORDER
    from smolvlm_attention.combined_collector import collect_combined_visual_attention,numerical_preflight
    from .config import LAYER_IDS
    from .store import read_run,input_rows,load_shard
    from .runtime import assert_measurement_compatible
    from .projected import open_projected_receipt,ProjectedSmolVLM
    from .inputs import make_batch_builder
    run=read_run(config_path);root=Path(run['output_root']);logs=root/'logs'
    logs.mkdir(parents=True,exist_ok=True)
    begin=time.monotonic();start,end=shard_bounds(run['count'],2,shard)
    state=dict(status='running',stage='starting',shard=shard,fingerprint=run['fingerprint'],
               total=end-start,processed=0,committed=0,batch_size=64)
    def progress(event):
        state.update(event,elapsed_seconds=round(time.monotonic()-begin,2))
        atomic_json(logs/f'attention_{shard}.status.json',state)
        print(json.dumps(state),flush=True)
    with FileLock(str(logs/f'attention_{shard}.worker.lock'),timeout=0):
        try:
            assert_measurement_compatible(run)
            if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
            torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            checkpoint=run['artifacts']['model']['checkpoint']
            if digest(checkpoint)!=run['identity']['checkpoint_sha256']:raise ValueError('Checkpoint changed')
            projected=open_projected_receipt(run['projected_receipt'])
            if projected.manifest['fingerprint']!=run['identity']['projected_fingerprint']:
                raise ValueError('Projected cache identity changed')
            store=load_shard(run,shard)
            progress(dict(stage='loading_model',processed=len(store.completed),committed=len(store.completed)))
            model=ProjectedSmolVLM.from_model(load_model(checkpoint,device='cuda:0',
                expected_prompt_camera_order=CAMERA_ORDER))
            if model.architecture_signature!=run['identity']['architecture']:
                raise ValueError('Worker architecture changed')
            build=make_batch_builder(run['store'],max_new_tokens=0)
            selected=input_rows(run)[start:end]
            progress(dict(stage='numerical_preflight'))
            with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
                result=numerical_preflight(model,selected,build,layer_ids=LAYER_IDS,
                    progress=lambda detail:progress(dict(stage='numerical_preflight',detail=detail)))
                atomic_json(logs/f'attention_{shard}.preflight.json',dict(result,
                    fingerprint=run['fingerprint'],gpu=torch.cuda.get_device_name(0)))
                process_pending(list(range(start,end)),store,lambda rows:collect_combined_visual_attention(
                    model,build(rows),layer_ids=LAYER_IDS,query_chunk_size=8),progress)
            if len(store.completed)!=end-start:raise ValueError('Incomplete shard coverage')
            progress(dict(status='complete',stage='complete',processed=end-start,committed=end-start,
                peak_vram_bytes=torch.cuda.max_memory_reserved(0)))
        except Exception as exc:
            progress(dict(status='failed',stage='failed',error=repr(exc)))
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--shard',type=int,choices=(0,1),required=True)
    args=parser.parse_args();run_worker(args.config,args.shard)


if __name__=='__main__':main()
