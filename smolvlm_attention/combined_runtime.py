"""Kaggle orchestration for same-layer combined visual scores (base only)."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from filelock import FileLock

from .combined_state import (BATCH_SIZE, LAYER_IDS, POLICY, QUERY_CHUNK_SIZE,
    input_rows, load_shard, merge_shards, read_run, write_run)
from .sample_store import atomic_bytes, atomic_json, digest, pid_is_alive, shard_bounds

_PROCESSES = {}


def measurement_sources():
    root=Path(__file__).resolve().parents[1]
    names=['smolvlm_attention/'+name for name in (
        'combined_collector.py','combined_state.py','combined_worker.py','collector.py',
        'sample_inputs.py','data.py','config.py','aggregate.py')]
    names += ['smolvlm_inference/'+name for name in ('model.py','text.py','cache.py')]
    return {name:digest(root/name) for name in names}


def assert_measurement_compatible(run):
    from .sample_inputs import library_versions
    if (run['identity'].get('sources')!=measurement_sources()
            or run['identity'].get('versions')!=library_versions()):
        raise ValueError('Measurement code/library versions changed; choose a new OUTPUT')


def _read_final(run):
    root=Path(run['output_root'])
    meta=json.loads((root/'result.json').read_text(encoding='utf-8'))
    if (meta.get('status')!='complete' or meta.get('fingerprint')!=run['fingerprint']
            or meta.get('count')!=run['count'] or meta.get('layer_ids')!=list(LAYER_IDS)):
        raise ValueError('Combined result identity/coverage mismatch')
    if digest(root/'combined_attention.npz')!=meta['sha256']:
        raise ValueError('Combined result hash mismatch')
    with np.load(root/'combined_attention.npz',allow_pickle=False) as data:
        total=data['sum'].copy(); mean=data['mean'].copy()
        if (total.dtype!=np.float64 or total.shape!=(15,384) or mean.shape!=total.shape
                or data['count'].item()!=run['count'] or data['fingerprint'].item()!=run['fingerprint']
                or data['layer_ids'].tolist()!=list(LAYER_IDS)
                or not np.isfinite(total).all() or (total<0).any()
                or (total.sum(-1)>run['count']*(1+1e-6)).any()
                or not np.array_equal(mean,total/run['count'])):
            raise ValueError('Invalid combined result values')
    return mean,meta


def resolve_run_mode(output_root,mode='auto'):
    if mode not in ('auto','measure','resume','plot'):
        raise ValueError('Select auto/measure/resume/plot')
    path=Path(output_root)/'combined_run.json'
    if not path.exists():
        if mode in ('resume','plot'):raise FileNotFoundError('No existing combined run')
        return 'measure'
    run=read_run(path)
    if mode=='measure':raise ValueError('Run exists; use resume or a new OUTPUT')
    if (path.parent/'result.json').exists():
        _read_final(run)
        return 'plot'
    if mode=='plot':raise ValueError('Plot requires a complete merged result; run cell 6 first')
    return 'resume'


def prepare_run(output_root,*,mode='auto',cache_input_dir='',prepared_dir=None,
                artifacts=None,source_commit=None,source_sha256=None):
    output=Path(output_root).resolve(); resolved=resolve_run_mode(output,mode)
    path=output/'combined_run.json'
    if resolved=='plot':return path
    if path.exists():assert_measurement_compatible(read_run(path))
    import torch
    from .sample_inputs import prepare_inputs,library_versions
    from .sample_runtime import resolve_artifacts
    from .config import MODELS,PREPARED_REPO,PREPARED_REVISION
    from smolvlm_inference.cache import CAMERA_ORDER
    versions=library_versions()
    if versions['transformers']!='4.46.3' or versions['tokenizers']!='0.20.4':
        raise ValueError('Requires transformers==4.46.3 and tokenizers==0.20.4')
    if torch.cuda.device_count()<2:raise RuntimeError('Enable two CUDA GPUs, e.g. T4 x2')
    output.mkdir(parents=True,exist_ok=True)
    artifacts=artifacts or resolve_artifacts(output,cache_input_dir,prepared_dir)
    if (artifacts['model']['repo']!=MODELS['base']['repo']
            or artifacts['model']['revision']!=MODELS['base']['revision']):
        raise ValueError('This experiment uses only the pinned base checkpoint')
    rows,receipt,data=prepare_inputs(artifacts,output)
    identity=dict(policy=POLICY,layer_ids=list(LAYER_IDS),batch_size=BATCH_SIZE,
        query_chunk_size=QUERY_CHUNK_SIZE,precision='float32_no_tf32',
        reduction='mean_heads_then_same_layer_formula_then_equal_sample_mean',
        query='last_valid_native_prompt_position',intermediaries='valid_nonvisual_except_query',
        softmax='all_valid_causal_keys_including_self',image_renormalization=False,
        model=MODELS['base']['repo'],revision=MODELS['base']['revision'],
        checkpoint_sha256=digest(artifacts['model']['checkpoint']),
        camera_order=list(CAMERA_ORDER),data={k:v for k,v in data.items() if k!='cache_payload_stat'},
        prepared_repo=PREPARED_REPO,prepared_revision=PREPARED_REVISION,
        sources=measurement_sources(),versions=versions)
    config=dict(identity=identity,artifacts=artifacts,cache_descriptor=str(receipt.resolve()),
                source_commit=source_commit,source_sha256=source_sha256)
    path=write_run(output,config,rows,mode=resolved)
    print(f'Ready: {len(rows)} test questions, layers {LAYER_IDS}, two GPUs, fixed batch=64.',flush=True)
    return path


def _pid_path(run,shard):
    return Path(run['output_root'])/'logs'/f'combined_{shard}.pid.json'


def _pid_alive(path):
    return path.exists() and pid_is_alive(json.loads(path.read_text(encoding='utf-8'))['pid'])


def launch_workers(config_path):
    path=Path(config_path).resolve(); run=read_run(path); assert_measurement_compatible(run)
    import torch
    if torch.cuda.device_count()<2:raise RuntimeError('Two CUDA GPUs are required')
    logs=path.parent/'logs';logs.mkdir(parents=True,exist_ok=True)
    with FileLock(str(logs/'launch.lock'),timeout=0):
        for shard in (0,1):
            key=(str(path),shard); current=_PROCESSES.get(key)
            if (current is not None and current.poll() is None) or _pid_alive(_pid_path(run,shard)):
                print(f'GPU {shard}: worker already running; reconnect using cell 5.',flush=True)
                continue
            acc=load_shard(run,shard)
            complete=not acc.pending_ids
            atomic_json(logs/f'combined_{shard}.status.json',dict(
                status='complete' if complete else 'starting',stage='complete' if complete else 'starting',
                processed=acc.count,committed=acc.count,total=len(acc.sample_ids),batch_size=BATCH_SIZE,
                fingerprint=run['fingerprint']))
            if complete:
                _PROCESSES.pop(key,None)
                continue
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='1',
                     TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
            source=str(Path(__file__).resolve().parents[1])
            env['PYTHONPATH']=source+os.pathsep+env.get('PYTHONPATH','')
            kwargs={'start_new_session':True} if os.name!='nt' else {'creationflags':subprocess.CREATE_NO_WINDOW}
            log=logs/f'combined_{shard}.log'
            with log.open('a',encoding='utf-8') as stream:
                process=subprocess.Popen([sys.executable,'-m','smolvlm_attention.combined_worker',
                    '--config',str(path),'--shard',str(shard)],cwd=source,env=env,
                    stdout=stream,stderr=subprocess.STDOUT,**kwargs)
            _PROCESSES[key]=process
            atomic_json(_pid_path(run,shard),dict(pid=process.pid,config=str(path)))
            print(f'GPU {shard}: PID {process.pid}; {len(acc.sample_ids)} samples; log {log}',flush=True)
    return path


def wait_workers(config_path,*,poll_seconds=1):
    from tqdm.auto import tqdm
    path=Path(config_path).resolve();run=read_run(path);logs=path.parent/'logs'
    bars={i:tqdm(total=shard_bounds(run['count'],2,i)[1]-shard_bounds(run['count'],2,i)[0],
                 desc=f'GPU {i}',position=i,unit='question') for i in (0,1)}
    total=tqdm(total=run['count'],desc='Total',position=2,unit='question')
    try:
        while True:
            finished=True;committed=0
            for shard in (0,1):
                status_path=logs/f'combined_{shard}.status.json'
                if not status_path.exists():raise RuntimeError('Launch workers using cell 4 first')
                state=json.loads(status_path.read_text(encoding='utf-8'))
                if state['fingerprint']!=run['fingerprint']:raise ValueError('Worker fingerprint differs')
                process=_PROCESSES.get((str(path),shard));code=process.poll() if process else None
                alive=code is None if process else _pid_alive(_pid_path(run,shard))
                if state['status']=='failed' or (code is not None and code!=0):
                    live=[i for i in (0,1) if _pid_alive(_pid_path(run,i))]
                    raise RuntimeError(f'GPU {shard}: {state.get("error",code)}; workers possibly alive: {live}; '
                                       f'log {logs/f"combined_{shard}.log"}')
                if not alive and state['status']!='complete':
                    raise RuntimeError(f'GPU {shard} exited without completion; log {logs/f"combined_{shard}.log"}')
                count=state['committed'];bars[shard].update(count-bars[shard].n)
                bars[shard].set_postfix_str(f"{state['stage']}; fixed batch=64",refresh=False)
                committed+=count
                finished=finished and not alive and state['status']=='complete'
            total.update(committed-total.n)
            if finished:
                merge_shards(run)  # Validate coverage before allowing plotting/finalization.
                return
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print('Workers remain running. Rerun cell 5 to reconnect.',flush=True)
        raise
    finally:
        for bar in list(bars.values())+[total]:bar.close()


def finalize_results(config_path):
    run=read_run(config_path);root=Path(run['output_root'])
    with FileLock(str(root/'finalize.lock'),timeout=0):
        if (root/'result.json').exists():
            _read_final(run)
            return root/'combined_attention.npz'
        total,count=merge_shards(run)
        payload=io.BytesIO()
        np.savez(payload,sum=total,mean=total/count,count=count,layer_ids=np.array(LAYER_IDS),fingerprint=run['fingerprint'])
        atomic_bytes(root/'combined_attention.npz',payload.getvalue())
        atomic_json(root/'result.json',dict(status='complete',fingerprint=run['fingerprint'],
            identity=run['identity'],count=count,layer_ids=list(LAYER_IDS),
            measurement_source_commit=run.get('source_commit'),measurement_source_sha256=run.get('source_sha256'),
            sha256=digest(root/'combined_attention.npz')))
    return root/'combined_attention.npz'


def load_results(config_path,*,plot_source_commit=None):
    from .plotting import shared_scales
    run=read_run(config_path); mean,meta=_read_final(run)
    scales=shared_scales([mean],layer_ids=LAYER_IDS)
    summary=dict(meta,plot_source_commit=plot_source_commit,
        value_label='Mean combined attention score',
        caption=('S_l(q,v)=mean_heads(A_l)[q,v]+sum_t mean_heads(A_l)[q,t]*mean_heads(A_l)[t,v]. '
            'Same layer only; t includes all valid non-visual tokens (camera labels and special tokens), '
            'excluding q and padding; native causal mask and full-key softmax; no image renormalization. '
            'q is the last valid native prompt token. Equal sample weighting over the complete test split. '
            'This is a defined combined score, not rollout or causal contribution. '
            'Histogram counts the 384 sample-averaged scores per selected layer.'),
        scales={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in scales.items()})
    atomic_json(Path(run['output_root'])/'figure_metadata.json',summary)
    return mean,meta,scales,summary
