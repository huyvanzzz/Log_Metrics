"""Pinned DINO/T5 inference orchestration, resumable equal-question measurement."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
from tqdm.auto import tqdm

from smolvlm_attention.aggregate import AttentionAccumulator
from .config import (MODEL_REPO, MODEL_REVISION, CHECKPOINT_SHA256, PREPARED_REPO,
    PREPARED_REVISION, CACHE_REPO, CACHE_REVISION, CACHE_FINGERPRINT, CAMERA_ORDER,
    COUNTS, TEST_COUNT, QUERY_POLICY, MEASUREMENT_FILES, BATCH_SIZE, REFERENCE_COMMIT)

_PROCESSES = {}


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def atomic_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(dir=path.parent,suffix='.tmp')
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2)
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def measurement_source_hash(directory=None):
    directory=Path(directory) if directory else Path(__file__).parent
    return hashlib.sha256(canonical({name:digest(directory/name) for name in MEASUREMENT_FILES})).hexdigest()


def aggregation_source_hash():
    from smolvlm_attention import aggregate
    return digest(aggregate.__file__)


def read_run(path):
    run=json.loads(Path(path).read_text(encoding='utf-8'))
    if hashlib.sha256(canonical(run['identity'])).hexdigest()!=run['fingerprint']:
        raise ValueError('Run fingerprint is damaged')
    if digest(run['prepared_test'])!=run['identity']['prepared_test_sha256']:
        raise ValueError('Prepared test inputs changed')
    return run


def shard_rows(run,shard):
    if not 0<=shard<run['shard_count']:
        raise ValueError('Invalid shard')
    rows=json.loads(Path(run['prepared_test']).read_text(encoding='utf-8'))['records']
    return list(np.array_split(np.asarray(rows,dtype=object),run['shard_count'])[shard])


def _state(run,shard):
    ids=[row['sample_id'] for row in shard_rows(run,shard)]
    path=Path(run['output_root'])/'data'/f'shard_{shard}'/'attention.npz'
    state=AttentionAccumulator.load(path,(4,384),run['fingerprint'],ids) if path.is_file() else AttentionAccumulator(
        (4,384),run['fingerprint'],ids)
    if np.any(state.sum<0) or np.any(state.sum.sum(-1)>state.count+1e-4):
        raise ValueError('Invalid attention snapshot probabilities')
    return state


def resolve_run_mode(output_root,mode='auto'):
    if mode not in ('auto','measure','resume','plot'):
        raise ValueError('mode must be auto, measure, resume or plot')
    output=Path(output_root)
    path=output/'run.json'
    if not path.exists():
        if mode in ('resume','plot'):
            raise ValueError('No existing run; use measure or auto')
        if any((output/'data').rglob('attention.npz')):
            raise ValueError('Measurement state exists without run config; choose a new output directory')
        return 'measure'
    if mode=='measure':
        raise ValueError('A run already exists; use resume/plot or a new output directory')
    run=read_run(path)
    states=[_state(run,shard) for shard in range(run['shard_count'])]
    complete=all(not state.pending_ids for state in states) and sum(state.count for state in states)==run['test_count']
    if mode=='plot' and not complete:
        raise ValueError('Cannot plot an incomplete run')
    return ('plot' if complete else 'resume') if mode=='auto' else mode


def resolve_artifacts(output_root):
    from huggingface_hub import hf_hub_download
    output=Path(output_root).resolve()
    prepared=output/'artifacts'/'prepared_data'
    for filename in ('manifest.json','train.jsonl','val.jsonl','test.jsonl'):
        hf_hub_download(repo_id=PREPARED_REPO,repo_type='dataset',revision=PREPARED_REVISION,
                        filename=filename,local_dir=prepared,token=False)
    cache=output/'artifacts'/'cache'
    for filename in ('manifest.json','index.json','features.float16.memmap'):
        hf_hub_download(repo_id=CACHE_REPO,repo_type='dataset',revision=CACHE_REVISION,
                        filename=filename,local_dir=cache,token=False)
    checkpoint=hf_hub_download(repo_id=MODEL_REPO,revision=MODEL_REVISION,filename='finetune_best.pt',token=False)
    return dict(prepared_dir=str(prepared.resolve()),cache_dir=str(cache.resolve()),
                checkpoint=str(Path(checkpoint).resolve()))


def _validate_artifacts(artifacts,output):
    from .data import validate_cache,open_validated_cache
    receipt=Path(output)/'validated_cache.json'
    if receipt.is_file():
        try:
            reader=open_validated_cache(receipt)
            if (reader.directory==Path(artifacts['cache_dir']).resolve() and
                reader.prepared==Path(artifacts['prepared_dir']).resolve() and
                reader.manifest['fingerprint']==CACHE_FINGERPRINT):
                print('Reusing matching validated DINO cache receipt.',flush=True)
                return receipt
        except (OSError,ValueError,KeyError,TypeError):
            pass
    return validate_cache(artifacts['cache_dir'],artifacts['prepared_dir'],output,
                          expected_fingerprint=CACHE_FINGERPRINT,progress=True)


def prepare_run(output_root,*,mode='auto',source_commit=None,source_sha256=None,gpu_count=None):
    output=Path(output_root).resolve()
    selected=resolve_run_mode(output,mode)
    path=output/'run.json'
    if selected=='plot':
        return path
    import torch
    count=min(2,torch.cuda.device_count()) if gpu_count is None else gpu_count
    if count not in (1,2):
        raise ValueError('Enable one or two GPUs')
    if selected=='resume':
        run=read_run(path)
        if run['identity']['measurement_source']!=measurement_source_hash():
            raise ValueError('Incompatible measurement code; use a new output directory')
        if run['identity'].get('aggregation_source')!=aggregation_source_hash():
            raise ValueError('Incompatible aggregation code; use a new output directory')
        if run['shard_count']!=count:
            raise ValueError('Resume requires the same GPU/shard count as the original run')
        if run['identity'].get('versions')!=_versions():
            raise ValueError('Measurement library versions changed; use the original environment')
        # Validate current inputs before touching snapshots/configuration.
        from .data import open_validated_cache
        reader=open_validated_cache(run['cache_descriptor'])
        if reader.manifest['fingerprint']!=run['identity']['cache_fingerprint']:
            raise ValueError('Cache differs from the saved measurement')
        if digest(run['artifacts']['checkpoint'])!=run['identity']['checkpoint_sha256']:
            raise ValueError('Checkpoint differs from the saved measurement')
        print('Compatible run: resuming only pending sample IDs.',flush=True)
        return path
    from .data import open_validated_cache,prepare_test,tokenizer_policy
    from .model import load_model
    artifacts=resolve_artifacts(output)
    descriptor=_validate_artifacts(artifacts,output)
    reader=open_validated_cache(descriptor)
    if reader.counts!=COUNTS or tuple(reader.array.shape)!=(4072,6,64,384):
        raise ValueError('Expected the complete pinned DINO cache/prepared data')
    print('Strict-loading base checkpoint on CPU for identity and tokenizer validation.',flush=True)
    model,tokenizer=load_model(artifacts['checkpoint'],device='cpu')
    inputs=prepare_test(reader,tokenizer,output/'inputs',progress=True)
    checkpoint_metadata=model.checkpoint_metadata
    del model
    gc.collect()
    identity=dict(policy=QUERY_POLICY,model_repo=MODEL_REPO,model_revision=MODEL_REVISION,
        checkpoint_sha256=digest(artifacts['checkpoint']),cache_fingerprint=reader.manifest['fingerprint'],
        prepared_revision=PREPARED_REVISION,cache_revision=CACHE_REVISION,
        prepared_test_sha256=digest(inputs),tokenizer=tokenizer_policy(tokenizer),camera_order=CAMERA_ORDER,
        measurement_source=measurement_source_hash(),aggregation_source=aggregation_source_hash(),
        precision='float32',versions=_versions())
    if identity['checkpoint_sha256']!=CHECKPOINT_SHA256:
        raise ValueError('Checkpoint SHA256 differs from pinned base checkpoint')
    fingerprint=hashlib.sha256(canonical(identity)).hexdigest()
    run=dict(version=1,output_root=str(output),artifacts=artifacts,cache_descriptor=str(Path(descriptor).resolve()),
        prepared_test=str(Path(inputs).resolve()),identity=identity,fingerprint=fingerprint,test_count=TEST_COUNT,
        shard_count=count,source_commit=source_commit,source_sha256=source_sha256,
        reference_commit=REFERENCE_COMMIT,checkpoint_metadata=checkpoint_metadata,requested_batch_size=BATCH_SIZE)
    atomic_json(path,run)
    print(f'Ready: {TEST_COUNT} questions, {count} GPU(s), batch {BATCH_SIZE}/GPU.',flush=True)
    return path


def _versions():
    import torch,transformers,tokenizers
    return dict(torch=torch.__version__,transformers=transformers.__version__,tokenizers=tokenizers.__version__,numpy=np.__version__)


def process_pending(rows,accumulator,measure,*,checkpoint,batch_size=BATCH_SIZE,progress=None):
    import torch
    if batch_size<1:
        raise ValueError('Batch must be positive')
    pending=set(accumulator.pending_ids)
    remaining=sorted((r for r in rows if r['sample_id'] in pending),key=lambda r:(-len(r['input_ids']),r['sample_id']))
    cursor=0
    batches=0
    while cursor<len(remaining):
        batch=remaining[cursor:cursor+batch_size]
        try:
            attention=measure(batch)
        except torch.cuda.OutOfMemoryError:
            if batch_size==1:
                raise
            batch_size=max(1,batch_size//2)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if progress:
                progress(dict(stage='oom_retry',processed=accumulator.count,batch_size=batch_size))
            continue
        values=attention.numpy() if isinstance(attention,torch.Tensor) else np.asarray(attention)
        if np.any(values<0) or not np.isfinite(values).all() or np.any(values.sum(-1)>1+1e-5):
            raise ValueError('Invalid measured attention probabilities')
        accumulator.add([r['sample_id'] for r in batch],attention)
        del attention
        cursor+=len(batch)
        batches+=1
        if progress:
            progress(dict(stage='measuring',processed=accumulator.count,batch_size=batch_size))
        if batches%100==0:
            accumulator.save(checkpoint)
    accumulator.save(checkpoint)
    return batch_size


def load_results(config_path,*,plot_source_commit=None,plot_source_sha256=None):
    run=read_run(config_path)
    rows=json.loads(Path(run['prepared_test']).read_text(encoding='utf-8'))['records']
    expected=[row['sample_id'] for row in rows]
    if len(set(expected))!=len(expected) or len(expected)!=run['test_count']:
        raise ValueError('Invalid selected test coverage')
    states=[_state(run,shard) for shard in range(run['shard_count'])]
    ids=[sample_id for state in states for sample_id in state.processed_ids]
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected):
        raise ValueError('Incomplete or duplicate measured sample IDs')
    count=sum(state.count for state in states)
    mean=sum((state.sum for state in states),np.zeros((4,384),dtype=np.float64))/count
    metadata=dict(run['identity'],fingerprint=run['fingerprint'],count=count,shape=[4,384],
        measurement_source_commit=run.get('source_commit'),measurement_source_sha256=run.get('source_sha256'),
        plot_source_commit=plot_source_commit,plot_source_sha256=plot_source_sha256,
        reference_commit=run.get('reference_commit',REFERENCE_COMMIT),checkpoint_metadata=run.get('checkpoint_metadata'),
        shard_count=run['shard_count'],requested_batch_size=32,
        interpretation='Decoder-start <pad> cross-attention to encoder visual positions; mean heads then equal questions',
        histogram_population='384 head-and-sample-averaged image-token positions per layer')
    output=Path(run['output_root'])
    atomic_json(output/'figure_metadata.json',metadata)
    fd,temporary=tempfile.mkstemp(dir=output,suffix='.npz')
    try:
        with os.fdopen(fd,'wb') as stream:
            np.savez(stream,mean=mean,count=count,fingerprint=run['fingerprint'])
        os.replace(temporary,output/'summary.npz')
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return mean,metadata


def numerical_preflight(model,batch):
    """Verify first-token generation and padded-batch parity before full measurement."""
    import torch
    with torch.inference_mode():
        return _numerical_preflight(model,batch)


def _numerical_preflight(model,batch):
    import torch
    from .collector import pad_cross_forward
    attention,logits=pad_cross_forward(model,batch)
    device=next(model.parameters()).device
    ids=batch['input_ids'].to(device)
    mask=batch['attention_mask'].to(device)
    features=batch['visual_features'].to(device=device,dtype=torch.float32)
    embeds,encoder_mask,*_=model._native_inputs(ids,mask,features)
    generated=model.text_model.generate(inputs_embeds=embeds,attention_mask=encoder_mask,
        max_new_tokens=1,num_beams=1,do_sample=False,return_dict_in_generate=True,output_scores=True,
        decoder_attention_mask=torch.ones((len(ids),1),dtype=torch.long,device=device))
    torch.testing.assert_close(logits,generated.scores[0].float().cpu(),rtol=1e-4,atol=2e-5)
    if not torch.equal(logits.argmax(-1),generated.sequences[:,-1].cpu()):
        raise ValueError('Native first-token greedy generation differs')
    for i in range(len(ids)):
        length=int(mask[i].sum())
        single=dict(input_ids=batch['input_ids'][i:i+1,:length],
                    attention_mask=batch['attention_mask'][i:i+1,:length],
                    visual_features=batch['visual_features'][i:i+1])
        single_attention,single_logits=pad_cross_forward(model,single)
        torch.testing.assert_close(attention[i:i+1],single_attention,rtol=2e-4,atol=2e-6)
        torch.testing.assert_close(logits[i:i+1],single_logits,rtol=2e-4,atol=3e-5)
    if attention.shape[1:]!=(4,384):
        raise ValueError(f'Expected four layers and 384 visual positions, got {attention.shape}')
    return dict(shape=list(attention.shape),native_first_token_parity=True,batch_single_parity=True)


def _pid_alive(path):
    if not Path(path).is_file():
        return False
    pid=json.loads(Path(path).read_text(encoding='utf-8'))['pid']
    try:
        os.kill(pid,0)
    except OSError:
        return False
    return True


def _process_key(config_path,shard):
    return (str(Path(config_path).resolve()),shard)


def run_worker(config_path,shard):
    import torch
    from .model import load_model
    from .collector import collect_pad_cross_attention
    from .data import open_validated_cache,make_batch,tokenizer_policy
    run=read_run(config_path)
    output=Path(run['output_root'])
    logdir=output/'logs'
    status_path=logdir/f'shard_{shard}.json'
    pid_path=logdir/f'shard_{shard}.pid.json'
    lock=logdir/f'shard_{shard}.lock'
    lock.parent.mkdir(parents=True,exist_ok=True)
    # OS lock survives notebook interrupts but releases even when a worker crashes.
    import filelock
    started=time.monotonic()
    with filelock.FileLock(str(lock),timeout=0):
        atomic_json(pid_path,dict(pid=os.getpid(),fingerprint=run['fingerprint']))
        state=None
        current={}
        def report(**event):
            current.update(event)
            current.update(fingerprint=run['fingerprint'],shard=shard,elapsed_seconds=round(time.monotonic()-started,2))
            atomic_json(status_path,current)
            print(json.dumps(current),flush=True)
        try:
            if (run['identity']['measurement_source']!=measurement_source_hash() or run['identity']['versions']!=_versions()
                    or run['identity']['aggregation_source']!=aggregation_source_hash()):
                raise ValueError('Measurement code/environment changed; prepare a compatible run')
            state=_state(run,shard)
            rows=shard_rows(run,shard)
            report(status='running',stage='loading',processed=state.count,total=len(rows),batch_size=BATCH_SIZE)
            if state.pending_ids:
                if not torch.cuda.is_available():
                    raise RuntimeError('CUDA GPU is required for measurement')
                torch.set_num_threads(1)
                torch.backends.cuda.matmul.allow_tf32=False
                torch.backends.cudnn.allow_tf32=False
                model,tokenizer=load_model(run['artifacts']['checkpoint'],device='cuda:0')
                inputs=json.loads(Path(run['prepared_test']).read_text(encoding='utf-8'))
                if inputs['tokenizer_policy']!=tokenizer_policy(tokenizer):
                    raise ValueError('Worker tokenizer policy differs from prepared inputs')
                reader=open_validated_cache(run['cache_descriptor'])
                selected=sorted(rows,key=lambda row:len(row['input_ids']))
                smoke=[selected[0],selected[-1]] if len(selected)>1 else selected
                report(stage='preflight')
                report(stage='preflight_passed',preflight=numerical_preflight(model,make_batch(reader,smoke,tokenizer)))
                bar=tqdm(total=len(rows),initial=state.count,desc=f'GPU {shard}',unit='questions')
                def progress(event):
                    bar.update(event['processed']-bar.n)
                    bar.set_postfix(batch=event['batch_size'])
                    report(**event)
                try:
                    process_pending(rows,state,lambda group:collect_pad_cross_attention(model,make_batch(reader,group,tokenizer)),
                        checkpoint=output/'data'/f'shard_{shard}'/'attention.npz',progress=progress)
                finally:
                    bar.close()
            report(status='complete',stage='complete',processed=state.count,total=len(rows))
        except BaseException as error:
            # Only committed batches are included; retry the failed batch after resume.
            if state is not None:
                state.save(output/'data'/f'shard_{shard}'/'attention.npz')
            report(status='failed',stage='failed',error=repr(error))
            traceback.print_exc()
            raise
        finally:
            pid_path.unlink(missing_ok=True)


def launch_workers(config_path):
    """Launch one pinned-base worker per shard; repeated calls reconnect to live workers."""
    import torch
    run=read_run(config_path)
    output=Path(run['output_root'])
    root=Path(__file__).resolve().parent.parent
    if torch.cuda.device_count()<run['shard_count']:
        raise ValueError('Not enough visible GPUs for the saved shard topology')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    devices=visible.split(',') if visible else [str(i) for i in range(torch.cuda.device_count())]
    for shard in range(run['shard_count']):
        key=_process_key(config_path,shard)
        pidpath=output/'logs'/f'shard_{shard}.pid.json'
        if (key in _PROCESSES and _PROCESSES[key].poll() is None) or _pid_alive(pidpath):
            print(f'Reconnecting to GPU shard {shard}.',flush=True)
            continue
        state=_state(run,shard)
        if not state.pending_ids:
            atomic_json(output/'logs'/f'shard_{shard}.json',dict(status='complete',stage='complete',
                processed=state.count,total=state.count,fingerprint=run['fingerprint']))
            continue
        atomic_json(output/'logs'/f'shard_{shard}.json',dict(status='running',stage='launching',
            processed=state.count,total=len(state.sample_ids),batch_size=BATCH_SIZE,fingerprint=run['fingerprint']))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[shard],PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',
                 PYTHONPATH=str(root)+os.pathsep+os.environ.get('PYTHONPATH',''))
        with (output/'logs'/f'shard_{shard}.log').open('a',encoding='utf-8') as log:
            process=subprocess.Popen([sys.executable,'-m','dinov2_attention.runtime','--config',str(Path(config_path).resolve()),
                                      '--shard',str(shard)],stdout=log,stderr=subprocess.STDOUT,cwd=root,env=env)
        _PROCESSES[key]=process
        atomic_json(pidpath,dict(pid=process.pid,fingerprint=run['fingerprint']))
    return dict(_PROCESSES)


def wait_workers(config_path,*,poll_seconds=1):
    """Live notebook bars; interrupting this function leaves subprocesses running."""
    run=read_run(config_path)
    logs=Path(run['output_root'])/'logs'
    totals=[len(shard_rows(run,i)) for i in range(run['shard_count'])]
    bars=[tqdm(total=n,desc=f'GPU {i}',unit='questions',position=i) for i,n in enumerate(totals)]
    total=tqdm(total=sum(totals),desc='Total',unit='questions',position=len(bars))
    try:
        while True:
            complete=[]
            for shard,bar in enumerate(bars):
                process=_PROCESSES.get(_process_key(config_path,shard))
                code=process.poll() if process is not None else None
                # Read AFTER poll: a worker may atomically complete immediately before exit.
                path=logs/f'shard_{shard}.json'
                status=json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
                if status.get('fingerprint') not in (None,run['fingerprint']):
                    raise ValueError('Worker status fingerprint differs')
                log=logs/f'shard_{shard}.log'
                if status.get('status')=='failed' or code not in (None,0):
                    raise RuntimeError(f"GPU shard {shard}: {status.get('error',f'worker exited {code}')}. Read {log}")
                bar.update(max(0,status.get('processed',0)-bar.n))
                bar.set_postfix(stage=status.get('stage','waiting'),batch=status.get('batch_size','-'))
                done=status.get('status')=='complete'
                if done and (status.get('processed')!=totals[shard] or _state(run,shard).pending_ids):
                    raise ValueError('Worker reported completion without full snapshot coverage')
                if not done and (code==0 or (process is None and not _pid_alive(logs/f'shard_{shard}.pid.json'))):
                    raise RuntimeError(f'GPU shard {shard} stopped without complete results. Read {log}')
                complete.append(done)
            total.update(sum(bar.n for bar in bars)-total.n)
            if all(complete):
                return
            time.sleep(poll_seconds)
    finally:
        for bar in [*bars,total]:
            bar.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--shard',required=True,type=int)
    args=parser.parse_args()
    run_worker(args.config,args.shard)
