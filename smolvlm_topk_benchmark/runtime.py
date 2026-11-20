"""Prepare, launch, reconnect, score and bundle fifteen layers or a separate base."""
import csv
import importlib.metadata
import io
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from filelock import FileLock
from .config import BATCH_SIZE,CONFIGS,CONFIG_SETS,K,MAX_NEW_TOKENS,POLICY,TEST_COUNT,worker_configs,config_layer,active_workers
from .state import (atomic_bytes,atomic_json,digest,fingerprint,initialize_run,read_run,input_rows,
                    load_batches,finalize_config,validate_result,pid_is_alive)

_PROCESSES={}


def generation_sources():
    root=Path(__file__).resolve().parents[1]
    names=['smolvlm_topk_benchmark/'+n for n in ('config.py','data.py','generation.py','worker.py','preflight.py','state.py')]
    names+=['smolvlm_inference/'+n for n in ('model.py','text.py','cache.py')]
    names+=['smolvlm_attention/data.py']
    names+=['smolvlm_attention/sample_inputs.py']
    result={n:digest(root/n) for n in names}
    from smolvlm_attention.sample_store import open_prompt_attention,validate_samples
    import hashlib
    for name,method in (('open_prompt_attention',open_prompt_attention),('validate_samples',validate_samples)):
        result['smolvlm_attention/sample_store.py:'+name]=hashlib.sha256(inspect.getsource(method).encode()).hexdigest()
    return result


def generation_versions():
    return {n:importlib.metadata.version(n) for n in ('torch','transformers','tokenizers','numpy','huggingface-hub')}


def assert_compatible(run):
    if run['identity'].get('sources')!=generation_sources() or run['identity'].get('versions')!=generation_versions():
        raise ValueError('Generation code/library versions changed; choose a new OUTPUT')


def _config_complete(run,name):
    path=Path(run['output_root'])/'results'/name
    if (path/'metadata.json').exists():
        validate_result(path,run); return True
    batches=load_batches(run,name)
    return set(batches)==set(range(0,run['count'],BATCH_SIZE))


def resolve_run_mode(output,config_set='layers',mode='auto'):
    if config_set not in CONFIG_SETS or mode not in ('auto','measure','resume','score'):
        raise ValueError('Select layers/base and auto/measure/resume/score')
    path=Path(output)/'benchmark_run.json'
    if not path.exists():
        if mode in ('score','resume'):raise FileNotFoundError('No existing benchmark run')
        return 'measure'
    run=read_run(path)
    if mode=='measure':raise ValueError('Run exists; use auto/resume or a new OUTPUT')
    complete=all(_config_complete(run,n) for n in CONFIG_SETS[config_set])
    if complete:return 'score'
    if mode=='score':raise ValueError('Generation incomplete for requested config set')
    return 'resume'


def _scorer_command(output,*,input_path=None,expected_count=None):
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-m','smolvlm_topk_benchmark.scoring','--output',str(output)]
    if input_path is None:command+=['--preflight']
    else:command+=['--input',str(input_path),'--expected-count',str(expected_count)]
    source=str(Path(__file__).resolve().parents[1]); env=dict(os.environ)
    env['PYTHONPATH']=source+os.pathsep+env.get('PYTHONPATH','')
    log=output.with_suffix('.log')
    try:
        root=output.parent.parent if output.parent.name=='scores' else output.parent
        logs=root/'logs';logs.mkdir(parents=True,exist_ok=True)
        key=('scorer',str(output.resolve()));pid_path=logs/'scorer.pid.json'
        with FileLock(str(logs/'scorer.lock'),timeout=0),log.open('w',encoding='utf-8') as stream:
            kwargs={'start_new_session':True} if os.name!='nt' else {'creationflags':subprocess.CREATE_NO_WINDOW}
            process=subprocess.Popen(command,cwd=source,env=env,stdout=stream,stderr=subprocess.STDOUT,**kwargs)
            try:
                _PROCESSES[key]=process
                atomic_json(pid_path,dict(pid=process.pid,stage='scoring',output=str(output)))
                code=process.wait(timeout=1800)
            except BaseException:
                if process.poll() is None:
                    if os.name=='nt':
                        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
                    else:
                        import signal
                        os.killpg(process.pid,signal.SIGKILL)
                    process.wait(timeout=10)
                raise
            finally:
                _PROCESSES.pop(key,None);pid_path.unlink(missing_ok=True)
        if code:raise RuntimeError(f'Scorer failed (code {code}); read {log}')
    except Exception as exc:
        atomic_json(output.with_suffix('.status.json'),dict(status='failed',error=repr(exc),log=str(log)))
        raise
    result=json.loads(output.read_text(encoding='utf-8'))
    atomic_json(output.with_suffix('.status.json'),dict(status='complete',log=str(log)))
    return result


def prepare_run(output,*,config_set='layers',mode='auto',attention_input=None,cache_input_dir='',prepared_dir=None,
                artifacts=None,source_commit=None,source_sha256=None):
    output=Path(output).resolve(); resolved=resolve_run_mode(output,config_set,mode)
    if resolved=='score':return output/'benchmark_run.json'
    previous=read_run(output/'benchmark_run.json') if (output/'benchmark_run.json').exists() else None
    if previous:assert_compatible(previous)
    versions=generation_versions()
    if versions['transformers']!='4.46.3' or versions['tokenizers']!='0.20.4':
        raise ValueError('Generation requires transformers==4.46.3 and tokenizers==0.20.4')
    import torch
    if torch.cuda.device_count()<len(active_workers(config_set)):
        raise RuntimeError(f'{config_set} requires {len(active_workers(config_set))} CUDA GPU(s)')
    from smolvlm_attention.sample_runtime import resolve_artifacts
    from smolvlm_attention.sample_inputs import prepare_inputs
    from .data import find_attention,check_export,prepare_selection
    output.mkdir(parents=True,exist_ok=True)
    directory,export=find_attention(attention_input)
    artifacts=artifacts or resolve_artifacts(output,cache_input_dir,prepared_dir)
    candidates,receipt,data_identity=prepare_inputs(artifacts,output)
    checkpoint_sha=digest(artifacts['model']['checkpoint'])
    rows=check_export(export,candidates,data_identity,checkpoint_sha)
    if any(r['prompt_length']+MAX_NEW_TOKENS>8192 for r in rows):
        raise ValueError('Original prompt plus 512 exceeds context; no truncation permitted')
    # Dependency failures must surface before the expensive full generation.
    _scorer_command(output/'metric_preflight.json')
    identity=dict(policy=POLICY,attention_fingerprint=export.metadata['fingerprint'],
        attention_manifest_sha256=digest(directory/'manifest.json'),checkpoint_sha256=checkpoint_sha,
        model=artifacts['model']['repo'],revision=artifacts['model']['revision'],
        data={k:v for k,v in data_identity.items() if k!='cache_payload_stat'},
        batch_size=BATCH_SIZE,k=K,generation=dict(do_sample=False,num_beams=1,max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,repetition_penalty=1.0,special_ids='pinned native generation config'),
        precision='float32_no_tf32',position_policy='preserve_original_rope_compact_cache_slots',
        selection='per_sample_single_layer_stable_top32_then_original_order',
        sources=generation_sources(),versions=generation_versions())
    operational=dict(artifacts=artifacts,cache_descriptor=str(receipt.resolve()),attention_dir=str(directory))
    path=initialize_run(output,identity,rows,operational,mode=resolved,
                        source_commit=source_commit,source_sha256=source_sha256)
    run=read_run(path)
    for name in CONFIG_SETS[config_set]:prepare_selection(run,name,export)
    print(f'Ready: {config_set}, {len(CONFIG_SETS[config_set])} configs x {len(rows)} samples; '
          f'{len(active_workers(config_set))} worker(s); sequential configs per GPU; fixed batch={BATCH_SIZE}; greedy<=512.',flush=True)
    return path


def _pid_path(run,config_set,worker):
    return Path(run['output_root'])/'logs'/f'bench_{config_set}_{worker}.pid.json'


def _pid_alive(path):
    return path.exists() and pid_is_alive(json.loads(path.read_text())['pid'])


def launch_workers(config_path,config_set='layers'):
    path=Path(config_path).resolve();run=read_run(path);assert_compatible(run)
    import torch
    workers=active_workers(config_set)
    if torch.cuda.device_count()<len(workers):
        raise RuntimeError(f'{config_set} requires {len(workers)} CUDA GPU(s)')
    logs=Path(run['output_root'])/'logs';logs.mkdir(parents=True,exist_ok=True)
    with FileLock(str(logs/'launch.lock'),timeout=0):
        for other in CONFIG_SETS:
            if other!=config_set and any(_pid_alive(_pid_path(run,other,w)) for w in (0,1)):
                raise RuntimeError('Another config set is still running; finish it first')
        for worker in workers:
            key=(str(path),config_set,worker);current=_PROCESSES.get(key)
            if (current is not None and current.poll() is None) or _pid_alive(_pid_path(run,config_set,worker)):
                print(f'GPU {worker}: reconnect with wait cell.',flush=True);continue
            names=worker_configs(config_set,worker)
            counts={n:sum(len(v) for v in load_batches(run,n).values()) for n in names}
            complete=all(_config_complete(run,n) for n in names)
            if complete:counts={n:run['count'] for n in names}
            status=dict(status='complete' if complete else 'starting',stage='complete' if complete else 'starting',
                fingerprint=run['fingerprint'],counts=counts,processed=sum(counts.values()),
                worker=worker,config_set=config_set,total=run['count']*len(names))
            atomic_json(logs/f'bench_{config_set}_{worker}.status.json',status)
            if complete:_PROCESSES.pop(key,None);continue
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(worker),TOKENIZERS_PARALLELISM='false',
                     PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1')
            source=str(Path(__file__).resolve().parents[1]);env['PYTHONPATH']=source+os.pathsep+env.get('PYTHONPATH','')
            kwargs={'start_new_session':True} if os.name!='nt' else {'creationflags':subprocess.CREATE_NO_WINDOW}
            with (logs/f'bench_{config_set}_{worker}.log').open('a',encoding='utf-8') as stream:
                process=subprocess.Popen([sys.executable,'-m','smolvlm_topk_benchmark.worker','--config',str(path),
                    '--config-set',config_set,'--worker',str(worker)],cwd=source,env=env,stdout=stream,
                    stderr=subprocess.STDOUT,**kwargs)
            _PROCESSES[key]=process;atomic_json(_pid_path(run,config_set,worker),dict(pid=process.pid,config=str(path)))
            print(f'GPU {worker}: {names}, PID {process.pid}, log {logs/(f"bench_{config_set}_{worker}.log")}',flush=True)
    return config_path


def wait_workers(config_path,config_set='layers',*,poll_seconds=1):
    from tqdm.auto import tqdm
    path=Path(config_path).resolve();run=read_run(path);logs=Path(run['output_root'])/'logs'
    bars={n:tqdm(total=run['count'],desc=n,position=i,unit='question') for i,n in enumerate(CONFIG_SETS[config_set])}
    workers=active_workers(config_set); config_count=len(CONFIG_SETS[config_set])
    gpu={w:tqdm(total=run['count']*len(worker_configs(config_set,w)),desc=f'GPU {w}',
                position=config_count+i,unit='question') for i,w in enumerate(workers)}
    total=tqdm(total=run['count']*config_count,desc='Total',position=config_count+len(workers),unit='question')
    try:
        while True:
            finished=True;processed=0
            for worker in workers:
                status_path=logs/f'bench_{config_set}_{worker}.status.json'
                if not status_path.exists():raise RuntimeError('Launch workers first')
                status=json.loads(status_path.read_text(encoding='utf-8'))
                if status['fingerprint']!=run['fingerprint']:raise ValueError('Worker identity differs')
                process=_PROCESSES.get((str(path),config_set,worker));code=process.poll() if process else None
                alive=code is None if process else _pid_alive(_pid_path(run,config_set,worker))
                if status['status']=='failed' or (code is not None and code!=0):
                    living=[w for w in (0,1) if _pid_alive(_pid_path(run,config_set,w))]
                    raise RuntimeError(f'GPU {worker}: {status.get("error",code)}; workers possibly still running: {living}; log {logs/(f"bench_{config_set}_{worker}.log")}')
                if not alive and status['status']!='complete':raise RuntimeError(f'GPU {worker} exited without completion; read {status_path.with_suffix(".log")}')
                finished=finished and not alive and status['status']=='complete'
                for name,count in status['counts'].items():
                    bars[name].update(count-bars[name].n)
                n=sum(status['counts'].values());gpu[worker].update(n-gpu[worker].n)
                gpu[worker].set_postfix_str(f"{status['stage']} {status.get('config','')}; fixed batch={BATCH_SIZE}",refresh=False)
                processed+=n
            total.update(processed-total.n)
            if finished:return
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print('Workers keep running. Rerun the wait cell to reconnect.',flush=True);raise
    finally:
        for bar in list(bars.values())+list(gpu.values())+[total]:bar.close()


def _scoring_fingerprint(identity):
    return fingerprint(dict(module_sha256=digest(Path(__file__).with_name('scoring.py')),
        version=identity['pycocoevalcap_version'],java=identity['java_version'],jar=identity['meteor_jar_sha256'],
        sources={k:v['sha256'] for k,v in identity['scorer_sources'].items()},
        tokenization=identity['tokenization'],meteor_sanitization=identity['meteor_sanitization']))


def score_results(config_path,config_set='layers',*,previous_results=None):
    from tqdm.auto import tqdm
    run=read_run(config_path);root=Path(run['output_root'])
    for name in CONFIG_SETS[config_set]:
        if not _config_complete(run,name):raise ValueError(f'{name}: generation incomplete')
        if not (root/'results'/name/'metadata.json').exists():finalize_config(run,name)
    sources={}
    for base in ([Path(previous_results)] if previous_results else [])+[root]:
        if base.name=='results':base=base.parent
        for name in CONFIGS:
            directory=base/'results'/name
            if (directory/'metadata.json').exists():
                info,_=validate_result(directory,run)
                if name in sources and digest(sources[name]/'predictions.jsonl')!=digest(directory/'predictions.jsonl'):
                    raise ValueError(f'Conflicting prior/current results for {name}')
                sources[name]=directory
    probe=_scorer_command(root/'metric_preflight.json');scoring_identity=_scoring_fingerprint(probe)
    summaries=[];rows=input_rows(run)
    for name in tqdm([n for n in CONFIGS if n in sources],desc='Score configurations',unit='config'):
        directory=sources[name];info,records=validate_result(directory,run)
        if any((r['sample_id'],r['eval_id'],r['reference'])!=(s['sample_id'],s['eval_id'],s['answer'])
               for r,s in zip(records,rows)):
            raise ValueError('Prediction reference mapping differs from benchmark inputs')
        prediction_hash=digest(directory/'predictions.jsonl')
        score_path=root/'scores'/f'{name}.json'
        cached=json.loads(score_path.read_text()) if score_path.exists() else None
        cache_key=fingerprint(dict(predictions=prediction_hash,scorer=scoring_identity))
        if cached and cached.get('fingerprint')==cache_key:
            result=cached['result']
            if cached.get('result_sha256')!=fingerprint(result):raise ValueError('Score checksum mismatch')
        else:
            result=_scorer_command(root/'scores'/f'{name}.raw.json',input_path=directory/'predictions.jsonl',expected_count=run['count'])
            if _scoring_fingerprint(result['scoring_identity'])!=scoring_identity:raise ValueError('Scorer identity changed during run')
            atomic_json(score_path,dict(fingerprint=cache_key,result=result,result_sha256=fingerprint(result),
                experiment_fingerprint=run['fingerprint'],prediction_sha256=prediction_hash))
        diagnostics=dict(result['diagnostics'],eos_count=sum(r['stop_reason']=='eos' for r in records),
                         max_new_tokens_count=sum(r['stop_reason']=='max_new_tokens' for r in records))
        atomic_json(root/'scores'/f'{name}.diagnostics.json',diagnostics)
        summaries.append(dict(config=name,layer=config_layer(name),k=info['k'],count=len(records),
            **result['metrics'],empty_prediction_count=diagnostics['empty_prediction_count'],
            eos_count=diagnostics['eos_count'],max_new_tokens_count=diagnostics['max_new_tokens_count']))
    baseline=next((r for r in summaries if r['config']=='base'),None)
    from .scoring import METRICS
    for item in summaries:
        item['delta_vs_base']={m:item[m]-baseline[m] for m in METRICS} if baseline else None
    summary=dict(experiment_fingerprint=run['fingerprint'],scoring_fingerprint=scoring_identity,
        config_set=config_set,baseline_available=baseline is not None,results=summaries,
        scope='Answer quality only; source layer selects input tokens; not pruning at that layer',
        generation=dict(greedy=True,max_new_tokens=512,batch_size=BATCH_SIZE,precision='float32_no_tf32'))
    atomic_json(root/'summary.json',summary)
    stream=io.StringIO();fields=['config','layer','k','count',*METRICS,'empty_prediction_count','eos_count','max_new_tokens_count']
    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
    for item in summaries:writer.writerow({k:item[k] for k in fields})
    atomic_bytes(root/'summary.csv',stream.getvalue().encode('utf-8'))
    atomic_json(root/'result_sources.json',{name:str(p) for name,p in sources.items()})
    return summary


def bundle_results(config_path):
    run=read_run(config_path);root=Path(run['output_root'])
    if not (root/'summary.json').exists():raise ValueError('Score results first')
    sources=json.loads((root/'result_sources.json').read_text(encoding='utf-8'))
    target=root/'topk_benchmark_results.zip'
    with FileLock(str(root/'bundle.lock'),timeout=0):
        files={n:root/n for n in ('benchmark_run.json','inputs.jsonl','summary.json','summary.csv','metric_preflight.json')}
        for name,source in sources.items():
            directory=Path(source);info,_=validate_result(directory,run)
            for file in ['metadata.json',*info['sha256']]:files[f'results/{name}/{file}']=directory/file
            for path in (root/'scores').glob(name+'.*.json'):files['scores/'+path.name]=path
            files[f'scores/{name}.json']=root/'scores'/f'{name}.json'
        manifest={name:digest(path) for name,path in files.items()}
        temporary=target.with_suffix('.partial.zip')
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as archive:
            for name,path in files.items():archive.write(path,arcname=name)
            archive.writestr('bundle_manifest.json',json.dumps(dict(version=1,experiment_fingerprint=run['fingerprint'],sha256=manifest),indent=2))
            archive.writestr('README.md','Top-32 benchmark results. Extract before reading. This ZIP excludes attention, checkpoints and feature cache. Partial generation resume requires the full OUTPUT directory.\n')
        os.replace(temporary,target)
    return target
