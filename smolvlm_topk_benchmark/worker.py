"""One resident model per GPU; finish each full-test configuration in order."""
import argparse
import json
from pathlib import Path
import time

from .config import BATCH_SIZE, CONFIG_SETS, worker_configs
from .state import (atomic_json, digest, read_run, input_rows, load_batches, load_selection, save_batch)


def schedule_batches(rows,names,completed,build_shared,generate,save,*,progress=None):
    """Fixed logical/physical batch 64 (last remainder), commit only complete results."""
    for name in names:
        for start in range(0,len(rows),BATCH_SIZE):
            if start in completed[name]:continue
            selected=rows[start:start+BATCH_SIZE]
            context=dict(start=start,sample_ids=[r['sample_id'] for r in selected],batch_size=len(selected))
            try:
                shared=build_shared(selected)
            except Exception as exc:
                raise RuntimeError(f'{name}: Fusion failed, batch {BATCH_SIZE} (no automatic reduction), {context}: {exc!r}') from exc
            try:
                if progress:progress(dict(stage='generating',config=name,**context))
                try:
                    records=generate(name,selected,shared)
                    save(name,start,records)
                except Exception as exc:
                    raise RuntimeError(f'{name}, fixed batch {BATCH_SIZE} (no retry/reduction), {context}: {exc!r}') from exc
                completed[name].add(start)
                if progress:progress(dict(stage='committed',config=name,**context))
            finally:
                del shared


def run_worker(config_path,config_set,worker):
    import torch
    from filelock import FileLock
    from smolvlm_inference.model import load_model,processor_policy
    from smolvlm_inference.cache import CAMERA_ORDER
    from smolvlm_attention.data import make_batch,open_validated_cache
    from .generation import fuse_batch,compact_prompt,SharedCausalLM,greedy_config,generate_compacted
    from .preflight import numerical_preflight
    from .runtime import assert_compatible
    run=read_run(config_path); root=Path(run['output_root']); logs=root/'logs'
    names=worker_configs(config_set,worker)
    started=time.monotonic(); rows=input_rows(run)
    completed={}; status=dict(status='running',stage='starting',worker=worker,config_set=config_set,
        fingerprint=run['fingerprint'],counts={name:0 for name in names},total=run['count']*len(names))
    status_path=logs/f'bench_{config_set}_{worker}.status.json'
    def progress(event):
        status.update(event,elapsed_seconds=round(time.monotonic()-started,2))
        if completed:
            status['counts']={name:sum(min(BATCH_SIZE,run['count']-start) for start in done)
                              for name,done in completed.items()}
        status['processed']=sum(status['counts'].values())
        atomic_json(status_path,status)
        if status['stage']!='committed' or status.get('start',0)%(BATCH_SIZE*25)==0:
            print(json.dumps(status),flush=True)
    logs.mkdir(parents=True,exist_ok=True)
    with FileLock(str(logs/f'bench_{config_set}_{worker}.worker.lock'),timeout=0):
        try:
            assert_compatible(run)
            if not torch.cuda.is_available():raise RuntimeError('Enable CUDA GPU')
            torch.set_num_threads(1); torch.set_float32_matmul_precision('highest')
            torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
            device=torch.device('cuda:0')
            completed.update({name:set(load_batches(run,name)) for name in names})
            selections={name:load_selection(run,name)[0] for name in names}
            progress(dict(stage='loading_model'))
            artifacts=run['operational']['artifacts']; spec=artifacts['model']
            if digest(spec['checkpoint'])!=run['identity']['checkpoint_sha256']:
                raise ValueError('Checkpoint hash changed')
            reader=open_validated_cache(run['operational']['cache_descriptor'])
            model=load_model(spec['checkpoint'],device=device,expected_prompt_camera_order=CAMERA_ORDER)
            if (model.native.config.text_config.num_hidden_layers!=30 or model.visual_token_len!=384
                    or model.geometry!=reader.manifest['geometry']
                    or processor_policy(model.processor)!=reader.manifest['processor_policy']
                    or reader.manifest['fingerprint']!=run['identity']['data']['cache_fingerprint']):
                raise ValueError('Model/cache geometry or processor differs')
            def build_batch(selected):return make_batch(reader,selected,model.processor,CAMERA_ORDER)
            progress(dict(stage='numerical_preflight'))
            with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
                result=numerical_preflight(model,rows,build_batch,selections,
                    progress=lambda detail:progress(dict(stage='numerical_preflight',detail=detail)))
                atomic_json(logs/f'preflight_{config_set}_{worker}.json',dict(result,
                    fingerprint=run['fingerprint'],gpu=torch.cuda.get_device_name(device)))
                adapter=SharedCausalLM(model.native)
                generation=greedy_config(model)
                def build(selected):return fuse_batch(model,build_batch(selected))
                def generate(name,selected,fused):
                    keep=selections[name]
                    indices=[None]*len(selected) if keep is None else [keep[r['row_index']].tolist() for r in selected]
                    compact=compact_prompt(fused,indices)
                    count=((compact['input_ids']==model.native.config.image_token_id)&compact['attention_mask'].bool()).sum(-1)
                    expected=384 if name=='base' else 32
                    if not (count==expected).all():raise ValueError('Incorrect physical visual count')
                    measured=generate_compacted(adapter,compact,model.processor.tokenizer,generation)
                    if len(measured)!=len(selected):raise ValueError('Generation count differs')
                    return [dict(value,sample_id=source['sample_id'],eval_id=source['eval_id'],reference=source['answer'])
                            for source,value in zip(selected,measured)]
                schedule_batches(rows,names,completed,build,generate,
                                 lambda name,start,records:save_batch(run,name,start,records,rows=rows),progress=progress)
            progress(dict(status='complete',stage='complete',peak_vram_bytes=torch.cuda.max_memory_reserved(device)))
        except Exception as exc:
            progress(dict(status='failed',stage='failed',error=repr(exc)))
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--config-set',choices=tuple(CONFIG_SETS),required=True)
    parser.add_argument('--worker',type=int,choices=(0,1),required=True)
    args=parser.parse_args()
    run_worker(args.config,args.config_set,args.worker)


if __name__=='__main__':main()
