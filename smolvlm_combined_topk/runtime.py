"""Two-phase Kaggle controller; completed results remain usable without GPU/cache."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from filelock import FileLock
from smolvlm_attention.sample_store import atomic_json, digest, pid_is_alive, shard_bounds

_PROCESSES = {}


def measurement_sources():
    root = Path(__file__).resolve().parents[1]
    names = [
        'smolvlm_combined_topk/projected.py', 'smolvlm_combined_topk/inputs.py',
        'smolvlm_combined_topk/config.py', 'smolvlm_attention/combined_collector.py',
        'smolvlm_attention/collector.py', 'smolvlm_inference/model.py',
        'smolvlm_inference/text.py', 'smolvlm_inference/cache.py',
        'smolvlm_predictor_io/compact.py', 'smolvlm_predictor_io/projected.py',
    ]
    return {name: digest(root/name) for name in names}


def assert_measurement_compatible(run):
    from smolvlm_attention.sample_inputs import library_versions
    if (run['identity'].get('sources') != measurement_sources()
            or run['identity'].get('versions') != library_versions()):
        raise ValueError('Measurement code/library identity changed; choose a new OUTPUT.')


def resolve_run_mode(output, mode='auto'):
    from .store import read_run
    if mode not in ('auto', 'measure', 'resume', 'score'):
        raise ValueError('RUN_MODE must be auto / measure / resume / score')
    path = Path(output)/'combined_run.json'
    if not path.exists():
        if mode in ('resume', 'score'):
            raise FileNotFoundError('No existing combined Top-K run at OUTPUT')
        return 'measure'
    read_run(path)
    if mode == 'measure':
        raise ValueError('OUTPUT already contains a run; use auto/resume or a new OUTPUT.')
    if mode == 'score':
        from .benchmark import validate_all_results
        validate_all_results(Path(output)/'benchmark'/'benchmark_run.json')
        return 'score'
    from .config import CONFIGS
    result_root=Path(output)/'benchmark'/'results'
    if all((result_root/name/'metadata.json').is_file() for name in CONFIGS):
        from .benchmark import validate_all_results
        validate_all_results(Path(output)/'benchmark'/'benchmark_run.json')
        return 'score'
    return 'resume'


def prepare_run(output, *, mode='auto', projected_input='', prepared_dir=None,
                checkpoint=None, source_commit=None, source_sha256=None):
    from .store import read_run, write_run, open_attention
    root = Path(output).resolve()
    selected_mode = resolve_run_mode(root, mode)
    path = root/'combined_run.json'
    if path.exists():
        run = read_run(path)
        if selected_mode == 'score':
            return path
        if (root/'attention_export'/'manifest.json').exists():
            open_attention(root/'attention_export')
            print('Complete per-sample attention reused; checking benchmark artifact paths.', flush=True)
            from .resume import resume_artifacts
            return resume_artifacts(run,projected_input=projected_input,
                                    prepared_dir=prepared_dir,checkpoint=checkpoint)
        assert_measurement_compatible(run)
        from .resume import resume_artifacts
        path=resume_artifacts(run,projected_input=projected_input,
                             prepared_dir=prepared_dir,checkpoint=checkpoint)
        print('Resuming compatible attention shards; committed rows will be verified.', flush=True)
        return path
    from .projected import discover_projected_cache, validate_projected_cache
    from .inputs import prepare_inputs
    from smolvlm_predictor_io.setup import resolve_artifacts
    from smolvlm_inference.model import load_model, processor_policy, CAMERA_ORDER
    from smolvlm_attention.config import MODELS, PREPARED_REPO, PREPARED_REVISION
    from smolvlm_attention.sample_inputs import library_versions
    from .config import LAYER_IDS, BATCH_SIZE
    versions = library_versions()
    if versions['transformers'] != '4.46.3' or versions['tokenizers'] != '0.20.4':
        raise ValueError('Requires transformers==4.46.3 and tokenizers==0.20.4')
    projected_dir = discover_projected_cache(projected_input)
    # Explicit complete projected input prevents the neutral resolver's raw-cache fallback.
    artifacts = resolve_artifacts(root, prepared_dir=prepared_dir,
        backbone_checkpoint=checkpoint, projected_input=str(projected_dir))
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root/'pipeline_prepare.lock'), timeout=0):
        if path.exists():
            raise RuntimeError('Another prepare created this run; rerun using auto.')
        print('Strict-load base checkpoint on CPU; validate projected cache once.', flush=True)
        model = load_model(artifacts['model']['checkpoint'], device='cpu',
                           expected_prompt_camera_order=CAMERA_ORDER)
        if (model.visual_token_len != 384 or model.geometry['text_dim'] != 576
                or model.native.config.text_config.num_hidden_layers != 30):
            raise ValueError('Requires native base architecture: 30 layers, 384 visual tokens, width 576')
        receipt = validate_projected_cache(projected_dir, model,
            artifacts['model']['checkpoint'], artifacts['prepared_dir'],
            root/'validation'/'projected_receipt.json')
        from .projected import open_projected_receipt
        projected = open_projected_receipt(receipt)
        identity = dict(
            policy='same_layer_mean_heads_direct_plus_nonvisual_two_hop_v1',
            layer_ids=list(LAYER_IDS), batch_size=BATCH_SIZE, query_chunk_size=8,
            precision='float32_no_tf32_no_autocast',
            query='last_valid_native_prompt_position',
            intermediaries='all_valid_nonvisual_except_query',
            softmax='all_valid_causal_keys_including_self', image_renormalization=False,
            reduction='mean_heads_before_products_no_sample_or_layer_mean',
            model=MODELS['base']['repo'], revision=MODELS['base']['revision'],
            checkpoint_sha256=digest(artifacts['model']['checkpoint']),
            architecture=model.architecture_signature,
            processor_policy=processor_policy(model.processor), camera_order=list(CAMERA_ORDER),
            projected_fingerprint=projected.manifest['fingerprint'],
            prepared_repo=PREPARED_REPO, prepared_revision=PREPARED_REVISION,
            sources=measurement_sources(), versions=versions)
        prepared = prepare_inputs(model, projected, artifacts['prepared_dir'],
                                  root/'inputs', identity)
        config = dict(identity=identity, artifacts=artifacts, batch_size=BATCH_SIZE,
            layer_ids=list(LAYER_IDS), geometry=[len(LAYER_IDS),384], shard_count=2,
            camera_order=list(CAMERA_ORDER),
            projected_receipt=str(Path(receipt).resolve()), store=str(Path(prepared['store']).resolve()),
            source_commit=source_commit, source_sha256=source_sha256)
        path = write_run(root, config, prepared['rows'], mode='measure')
        del model
    print('Ready: 16,817 questions; ten layers; two GPUs; fixed batch 64.', flush=True)
    return path


def _alive(path):
    return path.exists() and pid_is_alive(json.loads(path.read_text(encoding='utf-8'))['pid'])


def _launch(config_path, phase):
    from .store import read_run, shards_complete
    from .config import worker_configs
    path = Path(config_path).resolve()
    if phase == 'attention':
        run = read_run(path)
        root = Path(run['output_root'])
        if (root/'attention_export'/'manifest.json').exists():
            from .store import open_attention
            open_attention(root/'attention_export')
            return path
        assert_measurement_compatible(run)
        if shards_complete(run):
            print('All attention shards complete; run finalize cell.', flush=True)
            return path
    else:
        from .benchmark import assert_generation_compatible
        from smolvlm_topk_benchmark.state import read_run as read_bench
        run = read_bench(path)
        assert_generation_compatible(run)
        root = Path(run['operational']['logs_root']).parent
    import torch
    if torch.cuda.device_count() < 2:
        raise RuntimeError('Enable two CUDA GPUs, e.g. Kaggle T4 x2')
    logs = root/'logs'
    logs.mkdir(parents=True, exist_ok=True)
    with FileLock(str(logs/'launch.lock'), timeout=0):
        # Stages must not overlap; prevents accidental double models on a GPU.
        other = 'benchmark' if phase == 'attention' else 'attention'
        if any(_alive(logs/f'{other}_{i}.pid.json') for i in (0, 1)):
            raise RuntimeError('The preceding phase is still running; wait before launching this phase.')
        for worker in (0, 1):
            key = (str(path), worker)
            current = _PROCESSES.get(key)
            if (current is not None and current.poll() is None) or _alive(logs/f'{phase}_{worker}.pid.json'):
                print(f'{phase} GPU {worker}: already running; reconnect with wait cell.', flush=True)
                continue
            if phase == 'benchmark':
                from smolvlm_topk_benchmark.state import load_batches
                names = worker_configs(worker)
                done = {name:sum(min(64, run['count']-start) for start in load_batches(run,name)) for name in names}
                complete = all(count == run['count'] for count in done.values())
                initial = dict(counts=done,total=run['count']*5,processed=sum(done.values()))
            else:
                from .store import load_shard
                shard = load_shard(run, worker)
                start, end = shard_bounds(run['count'],2,worker)
                complete = len(shard.completed) == end-start
                initial = dict(total=end-start,processed=len(shard.completed),committed=len(shard.completed))
                del shard
            atomic_json(logs/f'{phase}_{worker}.status.json', dict(initial,
                status='complete' if complete else 'starting', stage='complete' if complete else 'starting',
                fingerprint=run['fingerprint'], batch_size=64))
            if complete:
                _PROCESSES.pop(key,None)
                continue
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(worker), OMP_NUM_THREADS='1',
                TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
            source=str(Path(__file__).resolve().parents[1])
            env['PYTHONPATH']=source+os.pathsep+env.get('PYTHONPATH','')
            module='smolvlm_combined_topk.'+('attention_worker' if phase=='attention' else 'benchmark_worker')
            argument='--shard' if phase=='attention' else '--worker'
            kwargs={'start_new_session':True} if os.name!='nt' else {'creationflags':subprocess.CREATE_NO_WINDOW}
            with (logs/f'{phase}_{worker}.log').open('a',encoding='utf-8') as stream:
                process=subprocess.Popen([sys.executable,'-m',module,'--config',str(path),argument,str(worker)],
                    cwd=source,env=env,stdout=stream,stderr=subprocess.STDOUT,**kwargs)
            _PROCESSES[key]=process
            atomic_json(logs/f'{phase}_{worker}.pid.json',dict(pid=process.pid,config=str(path)))
            print(f'{phase} GPU {worker}: PID {process.pid}; log {logs/f"{phase}_{worker}.log"}',flush=True)
    return path


def launch_attention(config_path):
    return _launch(config_path,'attention')


def launch_benchmark(benchmark_config):
    return _launch(benchmark_config,'benchmark')


def wait_workers(config_path, phase='attention', *, poll_seconds=1):
    from tqdm.auto import tqdm
    if phase not in ('attention','benchmark'):
        raise ValueError('phase must be attention or benchmark')
    path=Path(config_path).resolve()
    if phase=='attention':
        from .store import read_run,shards_complete
        run=read_run(path);root=Path(run['output_root'])
        if shards_complete(run) and not any(_alive(root/'logs'/f'attention_{i}.pid.json') for i in (0,1)):
            return
        totals=[shard_bounds(run['count'],2,i)[1]-shard_bounds(run['count'],2,i)[0] for i in (0,1)]
    else:
        from smolvlm_topk_benchmark.state import read_run
        run=read_run(path);root=Path(run['operational']['logs_root']).parent
        totals=[run['count']*5]*2
    logs=root/'logs'
    bars=[tqdm(total=totals[i],desc=f'{phase} GPU {i}',position=i,unit='question') for i in (0,1)]
    overall=tqdm(total=sum(totals),desc=f'{phase} total',position=2,unit='question')
    config_bars={}
    if phase=='benchmark':
        from .config import worker_configs
        for i,name in enumerate([n for worker in (0,1) for n in worker_configs(worker)]):
            config_bars[name]=tqdm(total=run['count'],desc=name,position=3+i,unit='question',leave=True)
    try:
        while True:
            finished=True;total=0
            for i in (0,1):
                status_file=logs/f'{phase}_{i}.status.json'
                if not status_file.exists():raise RuntimeError('Run the launch cell first.')
                state=json.loads(status_file.read_text(encoding='utf-8'))
                if state['fingerprint']!=run['fingerprint']:raise ValueError('Worker identity differs')
                proc=_PROCESSES.get((str(path),i));code=proc.poll() if proc else None
                alive=code is None if proc else _alive(logs/f'{phase}_{i}.pid.json')
                if state['status']=='failed' or (code is not None and code!=0):
                    raise RuntimeError(f"{phase} GPU {i}: {state.get('error',code)}; log {logs/f'{phase}_{i}.log'}; "
                        'the other worker may still be running.')
                if not alive and state['status']!='complete':
                    raise RuntimeError(f'{phase} GPU {i} exited without completion; inspect {logs/f"{phase}_{i}.log"}')
                value=state.get('committed',state.get('processed',0))
                bars[i].update(value-bars[i].n)
                bars[i].set_postfix_str(f"{state['stage']}; fixed batch=64",refresh=False)
                for name,count in state.get('counts',{}).items():
                    config_bars[name].update(count-config_bars[name].n)
                total+=value
                # Wait for process exit too: native CUDA allocations must be released before stage B.
                finished=finished and state['status']=='complete' and not alive
            overall.update(total-overall.n)
            if finished:return
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print('Workers remain running. Rerun this wait cell to reconnect.',flush=True)
        raise
    finally:
        for bar in bars+[overall]+list(config_bars.values()):bar.close()


def finalize_attention(config_path):
    from .store import finalize_export,prepare_selections,read_run
    from .benchmark import prepare_benchmark
    run=read_run(config_path)
    export=finalize_export(config_path)
    selections=prepare_selections(export,Path(run['output_root'])/'selections')
    benchmark=prepare_benchmark(run,export,selections)
    return export,benchmark
