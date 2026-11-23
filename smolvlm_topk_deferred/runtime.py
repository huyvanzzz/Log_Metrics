"""Two GPU paired QA experiment; no training, attention recomputation or timing claims."""
import argparse
from contextlib import ExitStack
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np
from filelock import FileLock

from smolvlm_attention.sample_store import atomic_json, atomic_bytes, digest, fingerprint
from .config import CONFIGS, BATCH_SIZE

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs/smolvlm_top32_score22_prune_input_vs_after2.json'


def load_config(path=DEFAULT_CONFIG):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if value != json.loads(DEFAULT_CONFIG.read_text(encoding='utf-8')):
        raise ValueError('Use the pinned paired score22/input-vs-after2 protocol')
    return value


def select_scores(attention, rows):
    from smolvlm_topk_benchmark.data import topk_indices
    if attention.shape != (len(rows), 30, 384):
        raise ValueError('Expected complete test probability export [N,30,384]')
    selected = np.empty((len(rows), 32), dtype=np.uint16)
    for start in range(0, len(rows), 256):
        batch = rows[start:start+256]
        values = np.asarray(attention[[r['attention_row'] for r in batch], 22, :])
        selected[start:start+len(batch)] = topk_indices(values)
    return selected


def schedule_batches(rows, completed, generate, save, *, progress=None):
    """Fixed 64 plus final remainder. Do not catch OOM or retry with smaller batch."""
    for start in range(0, len(rows), BATCH_SIZE):
        if start in completed:
            continue
        batch = rows[start:start+BATCH_SIZE]
        result = generate(batch)
        save(start, result)
        completed.add(start)
        if progress:
            progress(start, len(batch))


def sources():
    paths = list(Path(__file__).parent.glob('*.py'))
    names = ('smolvlm_topk_benchmark/generation.py', 'smolvlm_topk_benchmark/data.py',
             'smolvlm_topk_benchmark/state.py', 'smolvlm_topk_benchmark/preflight.py',
             'smolvlm_topk_benchmark/scoring.py', 'smolvlm_topk_benchmark/runtime.py',
             'smolvlm_attention/data.py', 'smolvlm_attention/config.py',
             'smolvlm_attention/sample_inputs.py', 'smolvlm_attention/sample_store.py',
             'smolvlm_attention/sample_runtime.py', 'smolvlm_inference/model.py',
             'smolvlm_inference/text.py', 'smolvlm_inference/cache.py')
    paths += [ROOT / name for name in names]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


def read_checked(path):
    from .state import read_run
    from smolvlm_topk_benchmark.runtime import generation_versions
    run = read_run(path)
    if (run['identity']['sources'] != sources() or run['identity']['versions'] != generation_versions()
            or run['identity']['protocol'] != load_config()):
        raise ValueError('Generation code/library/protocol changed; use a new OUTPUT')
    return run


def prepare(output, *, mode='auto', config=DEFAULT_CONFIG, attention_input=None, cache_input='', prepared_dir=None):
    from smolvlm_attention.config import MODELS, PREPARED_REPO, PREPARED_REVISION, CACHE_DATASET
    from smolvlm_attention.sample_runtime import resolve_artifacts
    from smolvlm_attention.sample_inputs import prepare_inputs
    from smolvlm_topk_benchmark.data import find_attention, check_export
    from smolvlm_topk_benchmark.runtime import generation_versions, _scorer_command
    from .state import initialize_run, write_selection, load_selection, validate_result
    cfg = load_config(config)
    if mode not in ('auto', 'measure', 'resume'):
        raise ValueError('Choose auto, measure or resume')
    if (MODELS['base']['repo'] != cfg['checkpoint_repo'] or MODELS['base']['revision'] != cfg['checkpoint_revision']
            or PREPARED_REPO != cfg['prepared_repo'] or PREPARED_REVISION != cfg['prepared_revision']
            or CACHE_DATASET != cfg['cache_dataset']):
        raise ValueError('Resolver pins differ from paired protocol')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with _control_locks(output):
        path = output / 'benchmark_run.json'
        if path.exists():
            previous = read_checked(path)
            if mode == 'measure':
                raise ValueError('Run exists; choose resume/auto or a new OUTPUT')
            if all((output / 'results' / name / 'metadata.json').exists() for name in CONFIGS):
                for name in CONFIGS:
                    validate_result(output / 'results' / name, previous, expected_config=name)
                print('Complete generation results validated; no model/input reload.', flush=True)
                return path
        elif mode == 'resume':
            raise FileNotFoundError('No existing paired benchmark to resume')
        versions = generation_versions()
        if versions['transformers'] != '4.46.3' or versions['tokenizers'] != '0.20.4':
            raise ValueError('Use Transformers 4.46.3 and Tokenizers 0.20.4')
        directory, export = find_attention(attention_input)
        artifacts = resolve_artifacts(output, cache_input, prepared_dir)
        candidates, receipt, data_identity = prepare_inputs(artifacts, output)
        checkpoint_sha = digest(artifacts['model']['checkpoint'])
        rows = check_export(export, candidates, data_identity, checkpoint_sha)
        if any(r['prompt_length']+512 > 8192 for r in rows):
            raise ValueError('Original prompt plus 512 exceeds context; no truncation')
        _scorer_command(output / 'metric_preflight.json')
        selection = select_scores(export.attention, rows)
        identity = dict(protocol=cfg, sources=sources(), versions=versions,
            batch_size=64, k=32, score_layer_index=22, score_layer_number=23,
            attention_fingerprint=export.metadata['fingerprint'],
            attention_manifest_sha256=digest(directory / 'manifest.json'),
            checkpoint_sha256=checkpoint_sha, model=artifacts['model']['repo'], revision=artifacts['model']['revision'],
            data={k: v for k, v in data_identity.items() if k != 'cache_payload_stat'},
            position_policy='original_rope_layer_local_physical_cache',
            selection_policy='same_stable_global_top32_then_source_order_for_both_configs')
        operational = dict(artifacts=artifacts, cache_descriptor=str(receipt.resolve()), attention_dir=str(directory))
        path = initialize_run(output, identity, rows, operational, mode=mode)
        run = read_checked(path)
        write_selection(run, selection)
        load_selection(run)
        print(json.dumps(dict(stage='prepared', configs=CONFIGS, count=len(rows), batch_size=64,
            score_layer_index=22, prune_after_layer_index=2, fingerprint=run['fingerprint'])), flush=True)
        return path


def _control_locks(output, *, workers=True):
    stack = ExitStack()
    try:
        stack.enter_context(FileLock(str(Path(output) / 'paired_control.lock'), timeout=0))
        if workers:
            for rank in (0, 1):
                stack.enter_context(FileLock(str(Path(output) / f'paired_worker_{rank}.lock'), timeout=0))
        return stack
    except BaseException:
        stack.close()
        raise


def run_worker(path, rank, gate):
    try:
        return _run_worker_impl(path, rank, gate)
    except BaseException as exc:
        from .state import read_run
        run = read_run(path)
        output = Path(run['output_root'])
        atomic_json(output / 'logs' / f'worker_{rank}.status.json',
                    dict(config=CONFIGS[rank], gpu=rank, stage='failed', error=repr(exc)))
        raise


def _run_worker_impl(path, rank, gate):
    import torch
    from smolvlm_attention.data import open_validated_cache, make_batch
    from smolvlm_inference.model import load_model, processor_policy
    from smolvlm_topk_benchmark.generation import (
        fuse_batch, compact_prompt, SharedCausalLM, greedy_config, generate_compacted)
    from .generation import DeferredGreedyDecoder
    from .preflight import numerical_preflight, stress
    from .state import input_rows, load_selection, load_batches, save_batch, finalize_config
    run = read_checked(path)
    output, name = Path(run['output_root']), CONFIGS[rank]
    with FileLock(str(output / f'paired_worker_{rank}.lock'), timeout=0):
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if torch.cuda.device_count() != 1:
            raise RuntimeError('Worker must see exactly its assigned GPU')
        torch.cuda.set_device(0)
        artifacts = run['operational']['artifacts']
        if digest(artifacts['model']['checkpoint']) != run['identity']['checkpoint_sha256']:
            raise ValueError('Checkpoint changed')
        reader = open_validated_cache(run['operational']['cache_descriptor'])
        model = load_model(artifacts['model']['checkpoint'], device='cuda:0',
            expected_prompt_camera_order=run['identity']['protocol']['camera_order'])
        if (model.visual_token_len != 384 or len(model.native.model.text_model.layers) != 30 or
                model.geometry != reader.manifest['geometry'] or
                processor_policy(model.processor) != reader.manifest['processor_policy']):
            raise ValueError('Worker model/cache geometry differs')
        model.requires_grad_(False)
        rows, selection = input_rows(run), load_selection(run)
        completed = set(load_batches(run, name))
        def build(batch):
            value = make_batch(reader, batch, model.processor, run['identity']['protocol']['camera_order'])
            if not torch.isfinite(value['visual_features']).all():
                raise ValueError('Nonfinite cache features')
            return value
        logs = output / 'logs'
        def report(stage, **values):
            event = dict(config=name, gpu=rank, stage=stage, **values)
            atomic_json(logs / f'worker_{rank}.status.json', event)
            print(json.dumps(event), flush=True)
        report('numerical_preflight')
        with torch.inference_mode(), torch.autocast('cuda', enabled=False):
            parity = numerical_preflight(model, rows, build, selection, name)
            report('stress', batch_size=64)
            stress_result = stress(model, rows, build, selection, name, batch_size=64, max_new_tokens=512)
            atomic_json(logs / f'preflight_{rank}.json', dict(parity=parity, stress=stress_result,
                config=name, fingerprint=run['fingerprint'], gpu=torch.cuda.get_device_name(0)))
            atomic_json(logs / f'ready_{rank}.json', dict(gate=str(gate), fingerprint=run['fingerprint']))
            report('waiting_for_peer')
            while not Path(gate).exists():
                time.sleep(.2)
            adapter = SharedCausalLM(model.native) if rank == 0 else DeferredGreedyDecoder(model, 2)
            generation = greedy_config(model)
            first_debug = [True]
            def generate(selected):
                fused = fuse_batch(model, build(selected))
                keep = [selection[r['row_index']].tolist() for r in selected]
                if rank == 0:
                    compact = compact_prompt(fused, keep)
                    if first_debug[0]:
                        atomic_json(logs / 'input_first_batch_debug.json', dict(
                            score_layer_index=22, prune_after_layer_index=None,
                            selected_source_indices=keep, visual_counts_by_layer=[32]*30,
                            prefill_cache_lengths=[compact['input_ids'].shape[1]]*30))
                        first_debug[0] = False
                    result = generate_compacted(adapter, compact, model.processor.tokenizer, generation)
                else:
                    result = adapter.generate(fused, keep, generation, debug=first_debug[0])
                    if first_debug[0]:
                        atomic_json(logs / 'after2_first_batch_debug.json', adapter.last_debug)
                        first_debug[0] = False
                return [dict(value, sample_id=row['sample_id'], eval_id=row['eval_id'], reference=row['answer'])
                        for row, value in zip(selected, result)]
            start_time, initial = time.perf_counter(), sum(min(64, len(rows)-s) for s in completed)
            last = [0.0]
            def progress(start, size):
                elapsed = time.perf_counter()-start_time
                committed = sum(min(64, len(rows)-s) for s in completed)
                if elapsed-last[0] >= 10 or start+size == len(rows):
                    speed = (committed-initial)/max(elapsed, 1e-9)
                    report('generation', committed=committed, total=len(rows), batch_size=size,
                        elapsed_seconds=elapsed, samples_per_second=speed,
                        eta_seconds=(len(rows)-committed)/max(speed, 1e-9),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
                    last[0] = elapsed
            schedule_batches(rows, completed, generate,
                lambda start, records: save_batch(run, name, start, records, rows=rows), progress=progress)
            finalize_config(run, name)
        report('complete', committed=len(rows), total=len(rows))


def launch(path):
    import torch
    from .state import validate_result
    run = read_checked(path)
    output = Path(run['output_root'])
    with _control_locks(output, workers=False):
        if all((output / 'results' / n / 'metadata.json').exists() for n in CONFIGS):
            for name in CONFIGS:
                validate_result(output / 'results' / name, run, expected_config=name)
            return
        if torch.cuda.device_count() < 2:
            raise RuntimeError('Enable two CUDA GPUs; no single-GPU fallback')
        logs = output / 'logs'
        logs.mkdir(exist_ok=True)
        gate = logs / f'gate_{time.time_ns()}.json'
        processes = []
        try:
            with ExitStack() as files:
                readers = []
                # Respect an existing CUDA_VISIBLE_DEVICES mapping instead of assuming physical IDs.
                visible = os.environ.get('CUDA_VISIBLE_DEVICES')
                devices = visible.split(',') if visible else ['0', '1']
                if len(devices) < 2:
                    raise RuntimeError('Two visible GPU identifiers required')
                for rank in (0, 1):
                    log = logs / f'worker_{rank}.log'
                    stream = files.enter_context(log.open('a', encoding='utf-8'))
                    reader = files.enter_context(log.open('r', encoding='utf-8'))
                    reader.seek(0, os.SEEK_END)
                    readers.append(reader)
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[rank], PYTHONUNBUFFERED='1', USE_TF='0', USE_FLAX='0')
                    processes.append(subprocess.Popen([sys.executable, '-m', 'smolvlm_topk_deferred',
                        'worker', '--run', str(Path(path).resolve()), '--rank', str(rank), '--gate', str(gate)],
                        env=env, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT))
                while True:
                    for rank, reader in enumerate(readers):
                        for line in reader:
                            print(f'[GPU {rank}] {line.rstrip()}', flush=True)
                    if any(p.poll() is not None and p.returncode != 0 for p in processes):
                        raise RuntimeError('Paired worker failed; sibling stopped. See logs; committed batches retained.')
                    if all(p.poll() == 0 for p in processes):
                        break
                    if not gate.exists():
                        ready = [logs / f'ready_{i}.json' for i in (0, 1)]
                        if all(p.exists() and json.loads(p.read_text()).get('gate') == str(gate) for p in ready):
                            atomic_json(gate, dict(preflight_and_stress_passed=True))
                            print('Both GPUs passed preflight/stress; paired generation released.', flush=True)
                    time.sleep(.5)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for name in CONFIGS:
            validate_result(output / 'results' / name, run, expected_config=name)


def score(path):
    from .state import validate_result
    from smolvlm_topk_benchmark.runtime import _scorer_command, _scoring_fingerprint
    from smolvlm_topk_benchmark.scoring import METRICS
    run = read_checked(path)
    output = Path(run['output_root'])
    with _control_locks(output):
        preflight = _scorer_command(output / 'metric_preflight.json')
        scoring_fingerprint = _scoring_fingerprint(preflight)
        results = []
        for name in CONFIGS:
            directory = output / 'results' / name
            metadata, records = validate_result(directory, run, expected_config=name)
            measured = _scorer_command(output / 'scores' / f'{name}.json',
                input_path=directory / 'predictions.jsonl', expected_count=run['count'])
            if _scoring_fingerprint(measured['scoring_identity']) != scoring_fingerprint:
                raise ValueError('Scorer identity changed during paired evaluation')
            results.append(dict(config=name, score_layer_index=22,
                prune_after_layer_index=None if name == CONFIGS[0] else 2,
                count=len(records), **measured['metrics'], diagnostics=measured['diagnostics']))
        delta = {m: results[1][m]-results[0][m] for m in METRICS}
        summary = dict(experiment_fingerprint=run['fingerprint'], scoring_fingerprint=scoring_fingerprint, results=results,
            delta_after2_vs_input=delta, scope='QA quality; same saved Top-32, different prune depth; no latency/FLOPs protocol')
        atomic_json(output / 'summary.json', summary)
        stream = io.StringIO()
        fields = ['config', 'score_layer_index', 'prune_after_layer_index', 'count', *METRICS]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({k: result[k] for k in fields})
        writer.writerow(dict(config='delta_after2_vs_input', score_layer_index=22,
                             prune_after_layer_index='', count=run['count'], **delta))
        atomic_bytes(output / 'summary.csv', stream.getvalue().encode('utf-8'))
        return summary


def bundle(path):
    from .state import validate_result
    run = read_checked(path)
    output = Path(run['output_root'])
    with _control_locks(output):
        if not (output / 'summary.json').exists():
            raise ValueError('Score both complete configs before bundling')
        files = [output / n for n in ('benchmark_run.json', 'inputs.jsonl', 'summary.json', 'summary.csv', 'metric_preflight.json')]
        for name in CONFIGS:
            directory = output / 'results' / name
            info, _ = validate_result(directory, run, expected_config=name)
            files += [directory / 'metadata.json'] + [directory / n for n in info['sha256']]
            files.append(output / 'scores' / f'{name}.json')
        archive = output / 'top32_score22_input_vs_after2_results.zip'
        temporary = archive.with_suffix('.partial.zip')
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as target:
            for file in files:
                target.write(file, file.relative_to(output))
            target.writestr('bundle_manifest.json', json.dumps(dict(experiment_fingerprint=run['fingerprint'],
                sha256={p.relative_to(output).as_posix(): digest(p) for p in files}), indent=2))
        os.replace(temporary, archive)
        return archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'worker', 'score', 'bundle'])
    parser.add_argument('--output', default='/kaggle/working/smolvlm_top32_score22_input_vs_after2_outputs')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--mode', choices=['auto', 'measure', 'resume'], default='auto')
    parser.add_argument('--attention-input')
    parser.add_argument('--cache-input', default='')
    parser.add_argument('--prepared-dir')
    parser.add_argument('--run')
    parser.add_argument('--rank', type=int, choices=[0, 1])
    parser.add_argument('--gate')
    args = parser.parse_args(argv)
    path = Path(args.run) if args.run else Path(args.output) / 'benchmark_run.json'
    if args.command == 'prepare':
        prepare(args.output, mode=args.mode, config=args.config, attention_input=args.attention_input,
                cache_input=args.cache_input, prepared_dir=args.prepared_dir)
    elif args.command == 'worker':
        if args.rank is None or not args.gate:
            parser.error('worker requires rank and gate')
        run_worker(path, args.rank, args.gate)
    elif args.command == 'run':
        launch(path)
    elif args.command == 'score':
        print(json.dumps(score(path), indent=2))
    else:
        print(bundle(path))
