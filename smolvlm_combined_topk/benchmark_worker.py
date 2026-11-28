"""Ten sequential per-layer benchmarks using the native cached greedy decoder."""
import argparse
import json
from pathlib import Path
import time
from smolvlm_attention.sample_store import atomic_json, digest
from smolvlm_topk_benchmark.worker import schedule_batches
from smolvlm_topk_benchmark.state import read_run, input_rows, load_batches, load_selection, save_batch, finalize_config


from .config import worker_configs


def run_worker(config_path, worker):
    import torch
    from filelock import FileLock
    from smolvlm_inference.model import load_model
    from smolvlm_inference.cache import CAMERA_ORDER
    from smolvlm_combined_topk.projected import ProjectedSmolVLM, open_projected_receipt
    from smolvlm_combined_topk.inputs import make_batch_builder
    from smolvlm_topk_benchmark.generation import fuse_batch, compact_prompt, SharedCausalLM, greedy_config, generate_compacted
    from smolvlm_topk_benchmark.preflight import numerical_preflight
    from .benchmark import assert_generation_compatible, input_store_fingerprint
    run=read_run(config_path); operational=run['operational']; rows=input_rows(run)
    logs=Path(operational.get('logs_root',str(Path(run['output_root'])/'logs'))); logs.mkdir(parents=True,exist_ok=True)
    names=worker_configs(worker); started=time.monotonic(); completed={}
    status=dict(status='running',stage='starting',worker=worker,fingerprint=run['fingerprint'],
                counts={name:0 for name in names},total=len(rows)*len(names),batch_size=64)
    def progress(event):
        status.update(event,elapsed_seconds=round(time.monotonic()-started,2))
        if completed:
            status['counts']={name:sum(min(64,len(rows)-start) for start in done) for name,done in completed.items()}
        status['processed']=sum(status['counts'].values())
        atomic_json(logs/f'benchmark_{worker}.status.json',status)
        if status['stage']!='committed' or status.get('start',0)%(64*25)==0: print(json.dumps(status),flush=True)
    with FileLock(str(logs/f'benchmark_{worker}.worker.lock'),timeout=0):
        try:
            assert_generation_compatible(run)
            completed.update({name:set(load_batches(run,name)) for name in names})
            if all(sum(min(64,len(rows)-s) for s in done)==len(rows) for done in completed.values()):
                for name in names: finalize_config(run,name)
                progress(dict(status='complete',stage='complete')); return
            if not torch.cuda.is_available(): raise RuntimeError('Enable two CUDA GPUs')
            torch.set_num_threads(1); torch.set_float32_matmul_precision('highest')
            torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
            device=torch.device('cuda:0')
            if digest(operational['checkpoint'])!=run['identity']['checkpoint_sha256']: raise ValueError('Checkpoint changed')
            projected=open_projected_receipt(operational['projected_receipt'])
            if projected.manifest['fingerprint']!=run['identity']['projected_fingerprint']:
                raise ValueError('Projected cache differs from attention measurement')
            if input_store_fingerprint(operational['store'])!=run['identity']['store_fingerprint']:
                raise ValueError('Compact prompt store changed')
            progress(dict(stage='loading_model'))
            model=ProjectedSmolVLM.from_model(load_model(operational['checkpoint'],device=device,expected_prompt_camera_order=CAMERA_ORDER))
            builder=make_batch_builder(operational['store'],max_new_tokens=512)
            selections={name:load_selection(run,name)[0] for name in names}
            progress(dict(stage='numerical_preflight'))
            with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
                result=numerical_preflight(model,rows,builder,selections,
                    progress=lambda detail:progress(dict(stage='numerical_preflight',detail=detail)))
                atomic_json(logs/f'benchmark_{worker}.preflight.json',dict(result,fingerprint=run['fingerprint']))
                adapter=SharedCausalLM(model.native); generation=greedy_config(model,512)
                def generate(name, selected, fused):
                    keep=[selections[name][row['row_index']].tolist() for row in selected]
                    compact=compact_prompt(fused,keep)
                    image=(compact['input_ids']==model.native.config.image_token_id)&compact['attention_mask'].bool()
                    if not (image.sum(-1)==32).all(): raise ValueError('Physical visual count is not 32')
                    values=generate_compacted(adapter,compact,model.processor.tokenizer,generation)
                    if len(values)!=len(selected): raise ValueError('Generation count differs')
                    return [dict(value,sample_id=row['sample_id'],eval_id=row['eval_id'],reference=row['answer']) for row,value in zip(selected,values)]
                schedule_batches(rows,names,completed,lambda selected:fuse_batch(model,builder(selected)),generate,
                    lambda name,start,records:save_batch(run,name,start,records,rows=rows),progress=progress)
                for name in names: finalize_config(run,name)
            progress(dict(status='complete',stage='complete',peak_vram_bytes=torch.cuda.max_memory_reserved(device)))
        except Exception as exc:
            progress(dict(status='failed',stage='failed',error=repr(exc))); raise


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--config',required=True)
    parser.add_argument('--worker',type=int,choices=(0,1),required=True)
    args=parser.parse_args(); run_worker(args.config,args.worker)

if __name__=='__main__': main()
