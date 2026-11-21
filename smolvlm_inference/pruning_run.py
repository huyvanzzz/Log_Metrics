"""Read-only MVPruner test runner. No training or baseline generation changes."""
import argparse
import importlib
import math
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

from .cache import CAMERA_ORDER, FEATURE_FILE, FeatureCache, digest, record_key, records

CHECKPOINT_REVISION = 'f9d4e9a4d704078f9e7f39f061d13602cf251c78'
PREPARED_REVISION = 'd2a6914f681dbb9148c4ac11e6fa92452010506e'
METRIC_NAMES = {'BLEU-1','BLEU-2','BLEU-3','BLEU-4','METEOR','ROUGE-L','CIDEr'}
FIXED = dict(stage1_keep=196, stage2_keep=32, batch_size=64, max_new_tokens=512,
             num_workers=0, num_beams=1, do_sample=False, repetition_penalty=1.0)


def validate_config(config):
    if any(config.get(k) != v or type(config.get(k)) is not type(v) for k, v in FIXED.items()):
        raise ValueError('MVPruner requires fixed FP32 batch64 greedy budgets')
    if type(config.get('stage2_layer_index')) is not int or config['stage2_layer_index'] not in (2, 15):
        raise ValueError('stage2_layer_index must be zero-based 2 or 15')
    if set(config) != set(FIXED) | {'stage2_layer_index'}:
        raise ValueError('unsupported configuration option')
    return config


def load_config(path):
    return validate_config(json.loads(Path(path).read_text(encoding='utf-8')))


def emit(**event):
    print(json.dumps(dict(time=time.time(), **event), allow_nan=False), flush=True)


def validation_progress(sink=emit, *, clock=time.monotonic):
    last_stage=None
    last_time=float('-inf')
    def callback(event):
        nonlocal last_stage,last_time
        now=clock()
        if event['stage'] != last_stage or event['completed'] == event['total'] or now-last_time >= 5:
            sink(**event)
            last_stage=event['stage'];last_time=now
    return callback


def file_receipt(cache, prepared):
    result = {}
    paths = [Path(cache)/n for n in ('manifest.json', 'index.json', FEATURE_FILE)]
    paths += [Path(prepared)/n for n in ('manifest.json', 'train.jsonl', 'val.jsonl', 'test.jsonl')]
    for path in paths:
        path = path.resolve()
        stat = path.stat()
        result[str(path)] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, device=stat.st_dev, inode=stat.st_ino,
                                sha256=None if path.name == FEATURE_FILE else digest(path))
    return result


def validate_cache(cache, prepared, identity, receipt):
    receipt = Path(receipt).resolve()
    if receipt.is_relative_to(Path(cache).resolve()) or receipt.is_relative_to(Path(prepared).resolve()):
        raise ValueError('validation receipt must be outside read-only data')
    reader = FeatureCache(cache, prepared, identity, progress=validation_progress())
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(dict(version=1, identity=identity, files=file_receipt(cache, prepared),
        fingerprint=reader.manifest['fingerprint'], counts=reader.counts), indent=2), encoding='utf-8')
    return reader


def open_validated_cache(cache, prepared, identity, receipt):
    saved = json.loads(Path(receipt).read_text(encoding='utf-8'))
    if saved.get('version') != 1 or saved.get('identity') != identity or saved.get('files') != file_receipt(cache, prepared):
        raise ValueError('validation receipt does not match current files/identity')
    reader = FeatureCache(cache, prepared, identity, full=False)
    if reader.manifest['fingerprint'] != saved['fingerprint'] or reader.counts != saved['counts']:
        raise ValueError('validation receipt fingerprint/counts mismatch')
    return reader


def validate_coverage(rows, expected):
    pairs = [(int(r['eval_id']), r['sample_id']) for r in rows]
    wanted = [(int(r['eval_id']), r['sample_id']) for r in expected]
    if len(set(x[0] for x in pairs)) != len(pairs) or len(set(pairs)) != len(pairs):
        raise ValueError('duplicate prediction eval_id')
    if len(pairs) != len(wanted) or set(pairs) != set(wanted):
        raise ValueError('predictions do not completely cover test split')


def preflight_metrics():
    if shutil.which('java') is None:
        raise RuntimeError('Java is required for METEOR; install before launching GPU workers')
    subprocess.run(['java', '-version'], check=True, capture_output=True)
    for name in ('bleu.bleu', 'meteor.meteor', 'rouge.rouge', 'cider.cider'):
        importlib.import_module('pycocoevalcap.'+name)
    # Exercise the actual Java process and scorer assets before model allocation.
    caption_metrics([dict(eval_id=0, reference='a car', prediction='a car')])


def caption_metrics(rows):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    refs = {int(r['eval_id']): [r['reference']] for r in rows}
    preds = {int(r['eval_id']): [r['prediction']] for r in rows}
    if len(refs) != len(rows):
        raise ValueError('duplicate metric eval_id')
    result = {}
    for cls, names in ((Bleu, ['BLEU-1','BLEU-2','BLEU-3','BLEU-4']), (Meteor,['METEOR']),
                       (Rouge,['ROUGE-L']), (Cider,['CIDEr'])):
        scorer = cls(4) if cls is Bleu else cls()
        try:
            score, _ = scorer.compute_score(refs, preds)
            values = score if isinstance(score, (list, tuple)) else [score]
            if len(values) != len(names):
                raise ValueError('metric scorer output count mismatch')
            result.update({name: float(value) for name, value in zip(names, values)})
        finally:
            if cls is Meteor and hasattr(scorer, 'meteor_p'):
                scorer.meteor_p.stdin.close()
                scorer.meteor_p.kill()
                scorer.meteor_p.wait()
    validate_metrics(result)
    return result


def validate_metrics(metrics):
    if set(metrics) != METRIC_NAMES or any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError('caption metrics require all seven finite metric values')
    return metrics


def score_predictions(path):
    path = Path(path)
    rows = list(records(path))
    metrics = validate_metrics(caption_metrics(rows))
    (path.parent/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    display={name:round(value*100,6) for name,value in metrics.items()}
    (path.parent/'metrics_display.json').write_text(json.dumps(display,indent=2),encoding='utf-8')
    return metrics


def memory_summary(device):
    import torch
    if torch.device(device).type != 'cuda':
        return dict(allocated=0,reserved=0,peak_allocated=0,peak_reserved=0,free=0,total=0,gpu_name='cpu')
    free,total=torch.cuda.mem_get_info(device)
    return dict(allocated=torch.cuda.memory_allocated(device),reserved=torch.cuda.memory_reserved(device),
                peak_allocated=torch.cuda.max_memory_allocated(device),peak_reserved=torch.cuda.max_memory_reserved(device),
                free=free,total=total,gpu_name=torch.cuda.get_device_name(device))


def configure_fp32():
    import torch
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def cache_identity():
    from transformers import AutoConfig, AutoProcessor
    from .model import MODEL_ID, MODEL_REVISION, _configure_processor, _geometry, processor_policy
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=False)
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=False)
    _configure_processor(processor, config)
    return dict(backend='smolvlm_no_pruner', version=1, model_id=MODEL_ID, revision=MODEL_REVISION,
        processor_policy=processor_policy(processor), geometry=_geometry(config), storage_dtype='float16',
        compute_dtype='float32', camera_order=CAMERA_ORDER)


def run_worker(args):
    import numpy as np
    import torch
    from .model import load_model
    from .pruning_text import PruningBatchCollator
    from .decoder import PrunedGreedyDecoder
    import transformers
    config = load_config(args.config)
    configure_fp32()
    identity = cache_identity()
    cache = open_validated_cache(args.cache, args.prepared, identity, args.receipt)
    rows = list(records(Path(args.prepared)/'test.jsonl'))
    if len(rows) != cache.counts['test'] or len(rows) != 16817:
        raise ValueError('expected full validated 16817-record test split')
    validate_coverage(rows, rows)
    model = load_model(args.checkpoint, device='cuda:0', expected_prompt_camera_order=CAMERA_ORDER)
    collator = PruningBatchCollator(model.processor, image_seq_len=model.geometry['visual_tokens_per_view'],
        max_length=model.native.config.text_config.max_position_embeddings, max_new_tokens=512,
        prompt_camera_order=CAMERA_ORDER)
    decoder = PrunedGreedyDecoder(model, config['stage2_layer_index'], stage1_keep=196, stage2_keep=32)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    (out/'metadata.json').write_text(json.dumps(dict(implementation_version='mvpruner_smolvlm_training_free_v1', torch_version=torch.__version__,
        transformers_version=transformers.__version__, allocator={key:os.environ.get(key) for key in
            ('PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF')}, config=config, checkpoint_revision=CHECKPOINT_REVISION,
        prepared_revision=PREPARED_REVISION, model_id=model.model_id, model_revision=model.model_revision,
        checkpoint_path=str(Path(args.checkpoint).resolve()), camera_order=CAMERA_ORDER, dtype='float32', autocast=False,
        tf32=False, eos_token_id=model.native.generation_config.eos_token_id,
        pad_token_id=model.native.generation_config.pad_token_id, test_count=len(rows), cache_fingerprint=cache.manifest['fingerprint'],
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')), indent=2))
    evaluate_batches(model,cache,rows,collator,decoder,out,args.command)


def prediction_rows(tokenizer, result, rows):
    generated = result['generated_ids']
    if generated.shape[0] != len(rows) or len(result['lengths']) != len(rows):
        raise ValueError('decoder batch output size mismatch')
    predictions = tokenizer.batch_decode([ids[:length] for ids,length in zip(generated,result['lengths'])], skip_special_tokens=True)
    return [dict(eval_id=row['eval_id'],sample_id=row['sample_id'],question=row['question'],
                 reference=str(row['answer']).strip(),prediction=prediction.strip())
            for row,prediction in zip(rows,predictions)]


def trim_debug(value, batch_size):
    if isinstance(value,dict):
        return {k:trim_debug(v,batch_size) for k,v in value.items()}
    if isinstance(value,list):
        if len(value)==batch_size and value and isinstance(value[0],(list,dict)):
            value=value[:3]
        return [trim_debug(v,batch_size) for v in value]
    return value


def evaluate_batches(model,cache,rows,collator,decoder,out,command,*,device='cuda:0',max_new_tokens=512,batch_size=64):
    import numpy as np
    import torch
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    if command == 'stress':
        # Use prompt-only collator metadata, never bulk features, to rank unique rows.
        ranked = sorted(range(len(rows)), key=lambda i: collator.prompt_length(rows[i]), reverse=True)[:batch_size]
        selected = [rows[i] for i in ranked]
    else:
        selected = rows
    written = []
    started=time.monotonic()
    stream = None if command == 'stress' else (out/'predictions.jsonl').open('w', encoding='utf-8')
    try:
        for start in range(0, len(selected), batch_size):
            chunk = selected[start:start+batch_size]
            items = []
            for row in chunk:
                feature = np.array(cache.array[cache.index[record_key(row)]], copy=True)
                if not np.isfinite(feature).all():
                    raise ValueError('nonfinite features in worker batch')
                items.append(dict(row, visual_features=torch.from_numpy(feature)))
            batch = collator(items)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
            with torch.inference_mode(), torch.autocast(torch.device(device).type, enabled=False):
                result = decoder.generate(batch, max_new_tokens=max_new_tokens, debug=start == 0)
                if command == 'stress':
                    normal_memory=memory_summary(device)
                    forced = decoder.generate(batch, max_new_tokens=max_new_tokens, ignore_eos=True, debug=True)
                    if any(n != max_new_tokens for n in forced['lengths']):
                        raise ValueError('forced cache stress did not reach 512 steps')
                    (out/'stress.json').write_text(json.dumps(dict(normal=trim_debug(result['debug'], len(chunk)), forced=trim_debug(forced['debug'], len(chunk)), ignore_eos=True,
                        forced_lengths=forced['lengths'], normal_memory=normal_memory, forced_memory=memory_summary(device), eval_ids=[r['eval_id'] for r in chunk]), indent=2))
            if start == 0:
                (out/'debug.json').write_text(json.dumps(trim_debug(result['debug'], len(chunk)), indent=2))
            if stream:
                for value in prediction_rows(model.processor.tokenizer, result, chunk):
                    stream.write(json.dumps(value, ensure_ascii=False)+'\n'); written.append(value)
                stream.flush()
            completed=min(start+batch_size,len(selected))
            elapsed=time.monotonic()-started
            event=dict(stage=command,completed=completed,total=len(selected),config=decoder.index,
                       elapsed_seconds=elapsed,samples_per_second=completed/max(elapsed,1e-9),
                       estimated_remaining_seconds=(len(selected)-completed)*elapsed/completed,
                       memory=memory_summary(device))
            with (out/'progress.jsonl').open('a',encoding='utf-8') as progress:
                progress.write(json.dumps(dict(time=time.time(),**event))+'\n')
            emit(**event)
        if stream:
            validate_coverage(written, rows)
    finally:
        if stream: stream.close()


def launch_pair(args, command):
    jobs = []
    for gpu, index in enumerate((2,15)):
        output = Path(args.output)/f'layer_index_{index}'
        output.mkdir(parents=True, exist_ok=True)
        log = (output/f'{command}.log').open('w', encoding='utf-8')
        argv = [sys.executable, '-m', 'smolvlm_inference.pruning_run', command,
            '--config', str(Path(__file__).resolve().parents[1]/'configs'/f'mvpruner_layer_index_{index}.json'),
            '--cache', args.cache, '--prepared', args.prepared, '--receipt', args.receipt,
            '--checkpoint', args.checkpoint, '--output', str(output)]
        jobs.append((subprocess.Popen(argv, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=log,
                                      stderr=subprocess.STDOUT), log, gpu, index))
    last_report=0
    while any(process.poll() is None for process,_,_,_ in jobs):
        now=time.monotonic()
        if now-last_report>=30:
            for process,_,gpu,index in jobs:
                latest=None
                progress=Path(args.output)/f'layer_index_{index}'/'progress.jsonl'
                if progress.exists():
                    with progress.open(encoding='utf-8') as stream:
                        for line in stream:
                            if line.strip():
                                try:
                                    candidate=json.loads(line)
                                    if candidate.get('stage') == command: latest=candidate
                                except json.JSONDecodeError: pass
                emit(stage=command,gpu=gpu,config=index,status='running' if process.poll() is None else process.returncode,
                     latest_progress=latest)
            last_report=now
        time.sleep(1)
    failures=[]
    for process,log,gpu,index in jobs:
        status=process.wait();log.close()
        if status: failures.append(dict(gpu=gpu,config=index,status=status))
    if failures: raise RuntimeError(f'{command} workers failed: {failures}; inspect per-config logs. No retries.')


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=['validate','stress','test','metrics','launch','preflight'])
    for name in ('cache','prepared','receipt','checkpoint','config','output'):
        parser.add_argument('--'+name)
    args = parser.parse_args(argv)
    required={'validate':('cache','prepared','receipt'),'stress':('cache','prepared','receipt','checkpoint','config','output'),
              'test':('cache','prepared','receipt','checkpoint','config','output'),
              'launch':('cache','prepared','receipt','checkpoint','output'),'metrics':('output',),'preflight':()}
    missing=[name for name in required[args.command] if not getattr(args,name)]
    if missing: parser.error('required for '+args.command+': '+', '.join('--'+name for name in missing))
    try:
        if args.command == 'preflight': preflight_metrics()
        elif args.command == 'validate': validate_cache(args.cache,args.prepared,cache_identity(),args.receipt)
        elif args.command in ('stress','test'): run_worker(args)
        elif args.command == 'metrics': score_predictions(Path(args.output)/'predictions.jsonl')
        else:
            preflight_metrics()
            launch_pair(args,'stress')
            launch_pair(args,'test')
            for index in (2,15): score_predictions(Path(args.output)/f'layer_index_{index}'/'predictions.jsonl')
            with zipfile.ZipFile(Path(args.output)/'mvpruner_artifacts.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
                for index in (2,15):
                    folder = Path(args.output)/f'layer_index_{index}'
                    for path in folder.iterdir():
                        if path.is_file() and path.suffix in ('.json','.jsonl','.log'):
                            archive.write(path, path.relative_to(args.output))
    except Exception as error:
        memory = {}
        if 'torch' in sys.modules:
            torch = sys.modules['torch']
            if torch.cuda.is_available():
                memory = dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                              peak=torch.cuda.max_memory_allocated())
        emit(stage=args.command, config=args.config, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
             error=repr(error), memory=memory)
        raise


if __name__ == '__main__':
    main()
