"""Fixed-batch CUDA worker for combined same-layer visual scores."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .combined_state import (BATCH_SIZE, LAYER_IDS, QUERY_CHUNK_SIZE, input_rows,
    load_shard, read_run, save_shard, shard_path)
from .sample_store import atomic_json, digest, shard_bounds


def process_pending(rows, accumulator, measure, checkpoint, progress=None):
    pending=set(accumulator.pending_ids)
    remaining=[row for row in rows if row['sample_id'] in pending]
    if [r['sample_id'] for r in remaining]!=accumulator.pending_ids:
        raise ValueError('Rows must preserve original shard order and coverage')
    for start in range(0,len(remaining),BATCH_SIZE):
        selected=remaining[start:start+BATCH_SIZE]
        ids=[row['sample_id'] for row in selected]
        try:
            values=measure(selected)
            if hasattr(values,'detach'):
                if values.device.type!='cpu':raise ValueError('Collector must return CPU scores')
                values=values.detach().numpy()
            values=np.asarray(values)
            if (values.dtype!=np.float32 or values.shape!=(len(selected),15,384)
                or not np.isfinite(values).all() or (values<0).any()
                or (values.sum(-1,dtype=np.float64)>1+1e-6).any()):
                raise ValueError('Invalid FP32 combined visual scores')
            accumulator.add(ids,values)
            save_shard(accumulator,checkpoint)
        except Exception as exc:
            raise RuntimeError(f'Fixed batch 64 failed; no batch reduction; sample IDs {ids}: {exc}') from exc
        if progress:
            progress(dict(stage='measuring',processed=accumulator.count,committed=accumulator.count,
                          batch_size=BATCH_SIZE,actual_batch_size=len(selected)))
    return BATCH_SIZE


def run_worker(config_path,shard):
    import torch
    from filelock import FileLock
    from smolvlm_inference.cache import CAMERA_ORDER
    from smolvlm_inference.model import load_model,processor_policy
    from .combined_collector import collect_combined_visual_attention,numerical_preflight
    from .combined_runtime import assert_measurement_compatible
    from .data import make_batch,open_validated_cache
    run=read_run(config_path)
    logs=Path(run['output_root'])/'logs';logs.mkdir(parents=True,exist_ok=True)
    begin=time.monotonic()
    start,end=shard_bounds(run['count'],2,shard)
    status=dict(status='running',stage='starting',shard=shard,total=end-start,
        processed=0,committed=0,batch_size=BATCH_SIZE,fingerprint=run['fingerprint'])
    def progress(event):
        status.update(event,elapsed_seconds=round(time.monotonic()-begin,2))
        atomic_json(logs/f'combined_{shard}.status.json',status)
        print(json.dumps(status),flush=True)
    with FileLock(str(logs/f'combined_{shard}.worker.lock'),timeout=0):
        try:
            assert_measurement_compatible(run)
            if not torch.cuda.is_available():raise RuntimeError('Worker requires CUDA')
            torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            device=torch.device('cuda:0')
            selected=input_rows(run)[start:end];acc=load_shard(run,shard)
            progress(dict(stage='loading_model',processed=acc.count,committed=acc.count))
            spec=run['artifacts']['model']
            if digest(spec['checkpoint'])!=run['identity']['checkpoint_sha256']:raise ValueError('Checkpoint changed')
            reader=open_validated_cache(run['cache_descriptor'])
            data=run['identity']['data']
            if (reader.manifest['fingerprint']!=data['cache_fingerprint']
                or reader.manifest['prepared_hashes']!=data['prepared_hashes']
                or reader.manifest['model_id']!=data['model_id']
                or reader.manifest['revision']!=data['native_revision']):
                raise ValueError('Validated cache differs from measurement identity')
            model=load_model(spec['checkpoint'],device=device,expected_prompt_camera_order=CAMERA_ORDER)
            if (model.native.config.text_config.num_hidden_layers!=30 or model.visual_token_len!=384
                or model.geometry!=reader.manifest['geometry']
                or processor_policy(model.processor)!=reader.manifest['processor_policy']
                or model.native.config.image_token_id!=run['identity']['data']['image_token_id']):
                raise ValueError('Worker model/cache geometry or processor mismatch')
            model.native.eval()
            def make(rows):
                batch=make_batch(reader,rows,model.processor,CAMERA_ORDER)
                return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            progress(dict(stage='numerical_preflight'))
            with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
                preflight_result=numerical_preflight(model,selected,make,
                    progress=lambda detail:progress(dict(stage='numerical_preflight',detail=detail)))
                atomic_json(logs/f'combined_{shard}.preflight.json',dict(preflight_result,
                    fingerprint=run['fingerprint'],atol=1e-6,batch_atol=1e-5,rtol=1e-4))
                process_pending(selected,acc,lambda rows:collect_combined_visual_attention(
                    model,make(rows),layer_ids=LAYER_IDS,query_chunk_size=QUERY_CHUNK_SIZE),
                    shard_path(run,shard),progress=progress)
            if acc.pending_ids:raise ValueError('Missing test samples')
            progress(dict(status='complete',stage='complete',processed=acc.count,committed=acc.count,
                gpu=torch.cuda.get_device_name(device),peak_vram_bytes=torch.cuda.max_memory_reserved(device)))
        except Exception as exc:
            progress(dict(status='failed',stage='failed',error=repr(exc)))
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--shard',type=int,required=True)
    args=parser.parse_args();run_worker(args.config,args.shard)


if __name__=='__main__':main()
