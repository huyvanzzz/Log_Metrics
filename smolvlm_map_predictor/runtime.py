"""Parent-only input preparation and supervised two-GPU launchers."""
from contextlib import ExitStack
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import zipfile

from filelock import FileLock
from smolvlm_attention.sample_store import atomic_json, digest, fingerprint
from .config import load_config, recipe, SOURCE_TRAINING_COMMIT


def versions():
    return {name: importlib.metadata.version(name) for name in
            ('torch', 'transformers', 'tokenizers', 'accelerate', 'numpy', 'huggingface-hub')}


def sources():
    root = Path(__file__).resolve().parents[1]
    packages = ('smolvlm_map_predictor','smolvlm_predictor_io')
    return {str(p.relative_to(root)).replace('\\','/'):digest(p)
            for package in packages for p in sorted((root/package).glob('*.py'))}


def resource_snapshot():
    from smolvlm_attention.teacher_targets.loader import memory_snapshot
    result = memory_snapshot()
    for path in (Path('/sys/fs/cgroup/memory.events'), Path('/sys/fs/cgroup/memory/memory.oom_control')):
        if path.is_file():
            result['memory_events'] = {line.split()[0]: int(line.split()[1])
                for line in path.read_text().splitlines() if len(line.split()) == 2}
            break
    for path in (Path('/sys/fs/cgroup/memory.stat'), Path('/sys/fs/cgroup/memory/memory.stat')):
        if path.is_file():
            values = {line.split()[0]: int(line.split()[1]) for line in path.read_text().splitlines()
                      if len(line.split()) == 2}
            result['cgroup_memory_stat'] = {key: values[key] for key in
                ('anon','file','inactive_file','active_file','shmem','kernel','total_inactive_file') if key in values}
            inactive = values.get('inactive_file', values.get('total_inactive_file',0))
            current = result.get('cgroup_current_bytes')
            result['working_set_approx_current_minus_inactive_file_bytes'] = max(0,current-inactive) if current is not None else None
            break
    path = Path('/proc/self/status')
    if path.is_file():
        result['process_memory_bytes'] = {line.split(':',1)[0]: int(line.split()[1])*1024
            for line in path.read_text().splitlines() if line.startswith(('VmRSS:','RssAnon:','RssFile:','RssShmem:'))}
    return result


def prepare(cfg, targets, output, *, cache_input='', prepared_dir=None, backbone_checkpoint=None, resume=False, projected_input='', projected_dir=''):
    from smolvlm_predictor_io.setup import resolve_artifacts
    from smolvlm_attention.config import MODELS, PREPARED_REPO, PREPARED_REVISION, CACHE_FINGERPRINT, CACHE_DATASET
    from smolvlm_inference.model import load_model, CAMERA_ORDER
    from .data import prepare_inputs
    from .training import seed, precision
    spec = cfg['inputs']
    pins = dict(checkpoint_repo=MODELS['base']['repo'], checkpoint_revision=MODELS['base']['revision'],
        prepared_repo=PREPARED_REPO, prepared_revision=PREPARED_REVISION,
        cache_dataset=CACHE_DATASET, cache_fingerprint=CACHE_FINGERPRINT)
    if spec != pins:
        raise ValueError('Predictor input pins differ from the established base artifacts')
    output = Path(output).resolve()
    cfg = dict(cfg, output_dir=str(output))
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / 'parent.lock'), timeout=0):
        path = output / 'run.json'
        previous = json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
        if bool(previous) != bool(resume):
            raise ValueError('Choose fresh OUTPUT for new run, or resume an existing prepared run')
        if previous and previous.get('execution') != dict(sources=sources(), versions=versions()):
            raise ValueError('Predictor code/library versions changed; incompatible training resume')
        precision(); seed(cfg['training']['seed'])
        artifacts = resolve_artifacts(output / 'internal' / 'artifacts', cache_input, prepared_dir,
            backbone_checkpoint, projected_input=projected_input, projected_dir=projected_dir)
        model = load_model(artifacts['model']['checkpoint'], device='cpu', expected_prompt_camera_order=CAMERA_ORDER)
        if model.geometry['text_dim'] != 576 or model.visual_token_len != 384:
            raise ValueError('Production predictor requires native SmolVLM-256M geometry')
        from smolvlm_predictor_io.orchestration import resolve_projected
        projected = resolve_projected(cfg, artifacts, model, output, projected_input, projected_dir)
        from smolvlm_predictor_io.projected import ProjectedCache
        projected_cache = ProjectedCache(projected, full=False)
        directory = output / 'internal' / f'inputs-{uuid.uuid4().hex}'
        receipt = prepare_inputs(targets, artifacts['cache_dir'], artifacts['prepared_dir'], model, directory,
            expected_counts=dict(train=341381, val=19785, test=16817),
            expected_cache_fingerprint=spec['cache_fingerprint'], checkpoint_sha256=digest(artifacts['model']['checkpoint']), projected_cache=projected_cache)
        from smolvlm_predictor_io.compact import prepare_store
        receipt = prepare_store(receipt, model.processor, directory / 'compact', projected_path=projected)
        value = json.loads(receipt.read_text(encoding='utf-8'))
        if previous and (previous['identity'] != value['identity'] or recipe(previous['config']) != recipe(cfg)):
            raise ValueError('Prepared predictor identity/recipe differs from resumed run')
        run = dict(version=1, output_dir=str(output), config=cfg, identity=value['identity'],
            receipt=str(receipt.resolve()), receipt_sha256=digest(receipt),
            backbone_checkpoint=artifacts['model']['checkpoint'],
            execution=dict(sources=sources(), versions=versions()), reference_training_commit=SOURCE_TRAINING_COMMIT)
        atomic_json(output / 'runtime_config.json', cfg)
        atomic_json(path, run)
        print(json.dumps(dict(phase='prepared', run=str(path), counts=value['counts'],
                              input_fingerprint=fingerprint(value['identity']))), flush=True)
        return path


def read_run(path, *, include_targets=True):
    from .data import read_receipt
    run = json.loads(Path(path).read_text(encoding='utf-8'))
    if run.get('version') != 1 or run['execution'] != dict(sources=sources(), versions=versions()):
        raise ValueError('Predictor code/library identity mismatch')
    if digest(run['receipt']) != run['receipt_sha256']:
        raise ValueError('Predictor receipt checksum mismatch')
    description = read_receipt(run['receipt'], include_targets=include_targets)
    if description['identity'] != run['identity']:
        raise ValueError('Predictor input receipt identity mismatch')
    if digest(run['backbone_checkpoint']) != run['identity']['checkpoint_sha256']:
        raise ValueError('Frozen base checkpoint changed')
    return run


def launch(run_path, command, *, resume=None, predictor=None):
    import torch
    run = read_run(run_path, include_targets=command != 'evaluate')
    if torch.cuda.device_count() < 2:
        raise RuntimeError('Two CUDA GPUs required; no CPU production fallback')
    output = Path(run['output_dir'])
    logs = output / 'logs'; logs.mkdir(exist_ok=True)
    with FileLock(str(output / 'parent.lock'), timeout=0):
        before = resource_snapshot()
        atomic_json(logs / f'{command}_resources_before.json', before)
        env = dict(os.environ, USE_TF='0', USE_FLAX='0', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        if command == 'evaluate':
            env['SMOLVLM_MAP_EVALUATION_SESSION'] = uuid.uuid4().hex
            calls = []
            visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
            visible = [item.strip() for item in visible if item.strip()]
            if not visible:
                visible = ['0', '1']
            if len(visible) < 2:
                raise RuntimeError('Two visible CUDA devices are required for independent evaluation workers')
            for rank in (0, 1):
                child_env = dict(env, CUDA_VISIBLE_DEVICES=visible[rank])
                args = [sys.executable, '-m', 'smolvlm_map_predictor', 'evaluate-worker', '--run', str(Path(run_path).resolve()),
                        '--rank', str(rank), '--predictor', str(predictor)]
                calls.append((args, child_env, logs / f'evaluate_gpu_{rank}.log'))
            _parallel(calls)
            from .evaluation import finalize_evaluation
            finalize_evaluation(run, predictor)
        else:
            if command == 'train':
                diagnose = json.loads((output / 'diagnose.json').read_text(encoding='utf-8'))
                if not diagnose.get('ok') or diagnose['identity'] != run['identity'] or diagnose['recipe'] != recipe(run['config']):
                    raise ValueError('Run diagnose successfully with this input identity/recipe before training')
            args = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                    '-m', 'smolvlm_map_predictor', command + '-worker', '--run', str(Path(run_path).resolve())]
            if resume:
                args += ['--resume', str(Path(resume).resolve())]
            process = None
            try:
                with (logs / f'{command}.log').open('a', encoding='utf-8') as stream:
                    process = subprocess.Popen(args, env=env, stdout=stream, stderr=subprocess.STDOUT)
                    status = process.wait()
                if status:
                    raise subprocess.CalledProcessError(status, args)
            except subprocess.CalledProcessError as exc:
                after = resource_snapshot()
                atomic_json(logs / f'{command}_failure.json', dict(returncode=exc.returncode, before=before, after=after))
                raise RuntimeError(f'Predictor {command} exited {exc.returncode}; see elastic rank error and {logs}. No settings were reduced.') from exc
            finally:
                if process is not None and process.poll() is None:
                    process.terminate()  # elastic forwards termination to all its ranks
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
            if command == 'diagnose':
                rows = [json.loads((output / f'diagnose_rank_{rank}.json').read_text(encoding='utf-8')) for rank in (0,1)]
                if not all(row.get('ok') for row in rows):
                    raise RuntimeError('Predictor training stress did not pass on both ranks')
                timing = [row.get('timing') for row in rows]
                combined = {}
                if all(timing):
                    slowest = max(item['pipeline_seconds'] for item in timing)
                    combined = dict(samples_per_second=sum(item['measured_samples'] for item in timing)/slowest,
                        seconds_per_update=slowest/timing[0]['measured_updates'],
                        scope='two_rank_pipeline_slowest_rank',
                        startup_seconds_by_rank=[row['startup_seconds'] for row in rows])
                atomic_json(output / 'diagnose.json', dict(ok=True, identity=run['identity'], recipe=recipe(run['config']), ranks=rows, throughput=combined))
        atomic_json(logs / f'{command}_resources_after.json', resource_snapshot())


def _parallel(calls):
    processes = []
    try:
        with ExitStack() as stack:
            readers = []
            for args, env, path in calls:
                stream = stack.enter_context(path.open('a', encoding='utf-8'))
                reader = stack.enter_context(path.open('r', encoding='utf-8')); reader.seek(0, 2)
                readers.append(reader)
                processes.append(subprocess.Popen(args, env=env, stdout=stream, stderr=subprocess.STDOUT))
            while True:
                for rank, process in enumerate(processes):
                    status = process.poll()
                    if status is not None and status != 0:
                        raise RuntimeError(f'Predictor evaluation rank {rank} exited {status}; {calls[rank][2]}; resources={resource_snapshot()}')
                if all(process.poll() == 0 for process in processes):
                    break
                time.sleep(.5)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()


def bundle(run_path):
    run = read_run(run_path, include_targets=False)
    output = Path(run['output_dir'])
    with FileLock(str(output / 'parent.lock'), timeout=0):
        names = ('runtime_config.json', 'run.json', 'training_status.json', 'predictions.jsonl', 'metrics.json',
                 'metrics_display.json', 'evaluation_summary.json', 'scoring_identity.json', 'diagnose.json')
        files = [output / name for name in names]
        for directory in ('checkpoints', 'debug', 'logs', 'evaluation'):
            files.extend((output / directory).rglob('*'))
        target = output / 'map_predictor_artifacts.zip'
        temporary = target.with_suffix('.zip.tmp')
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for path in sorted(files):
                if path.is_file() and path.suffix not in ('.lock', '.tmp'):
                    archive.write(path, path.relative_to(output))
        os.replace(temporary, target)
        atomic_json(target.with_suffix('.sha256.json'), dict(filename=target.name, sha256=digest(target)))
        return target
