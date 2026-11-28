"""Portable benchmark state, corpus scoring and a payload-free result archive."""
import csv
import importlib.metadata
import io
import json
import os
from pathlib import Path
import zipfile
import numpy as np
from filelock import FileLock
from smolvlm_attention.sample_store import atomic_bytes, atomic_json, digest, fingerprint, pid_is_alive
from smolvlm_topk_benchmark.state import (initialize_run, read_run, input_rows, write_selection,
    finalize_config, validate_result)

from .config import LAYER_IDS as LAYERS, CONFIGS


def generation_sources():
    root=Path(__file__).resolve().parents[1]
    # Plotting, orchestration, scoring and archive code are deliberately excluded.
    names=['smolvlm_topk_benchmark/'+name for name in ('generation.py','preflight.py','state.py','worker.py','config.py')]
    names += ['smolvlm_inference/'+name for name in ('model.py','text.py','cache.py')]
    names += ['smolvlm_combined_topk/'+name for name in ('config.py','projected.py','inputs.py','benchmark_worker.py')]
    names += ['smolvlm_predictor_io/compact.py']
    return {name:digest(root/name) for name in names}


def assert_generation_compatible(run):
    if (run['identity']['generation_sources']!=generation_sources()
            or run['identity'].get('generation_versions')!=generation_versions()):
        raise ValueError('Generation code changed; use a new output rather than mixing predictions')


def generation_versions():
    return {name:importlib.metadata.version(name) for name in ('torch','transformers','tokenizers','numpy')}


def input_store_fingerprint(path):
    """Bind compact prompt contents, not their current paths or filesystem timestamps."""
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    identity={key:value[key] for key in ('source_identity','counts','geometry','max_length','pad_token_id')}
    identity['files']={split:{role:digest(filename) for role,filename in roles.items()
                              if role!='target' and filename is not None}
                       for split,roles in value['splits'].items()}
    return fingerprint(identity)


def prepare_benchmark(combined_run, attention_export, selections=None):
    from .store import open_attention
    if not isinstance(combined_run,dict):
        from .store import read_run as read_combined
        combined_run=read_combined(combined_run)
    run=combined_run
    root=Path(run.get('output_root',Path(run.get('config_path','.')).parent))
    exported=open_attention(attention_export)
    rows=exported.samples
    metadata=exported.metadata
    layer_ids=metadata.get('layer_ids')
    if tuple(layer_ids or ())!=LAYERS or metadata['fingerprint']!=run['fingerprint']:
        raise ValueError('Attention identity or layer-axis mapping differs')
    identity=dict(attention_manifest_sha256=digest(Path(attention_export)/'manifest.json'),
        checkpoint_sha256=run['identity']['checkpoint_sha256'],batch_size=64,k=32,
        projected_fingerprint=run['identity']['projected_fingerprint'],store_fingerprint=input_store_fingerprint(run['store']),
        max_new_tokens=512,greedy=True,precision='float32_no_tf32',layer_ids=list(LAYERS),
        generation_sources=generation_sources(),generation_versions=generation_versions())
    operational=dict(projected_receipt=run['projected_receipt'],store=run['store'],
        checkpoint=run['artifacts']['model']['checkpoint'],logs_root=str(root/'logs'),batch_size=64)
    logs=root/'logs'; logs.mkdir(parents=True,exist_ok=True)
    with FileLock(str(logs/'launch.lock'),timeout=0):
        if any(pid_is_alive(json.loads(path.read_text())['pid']) for path in logs.glob('*.pid.json')):
            raise RuntimeError('Workers still running; wait before preparing benchmark')
        config=initialize_run(root/'benchmark',identity,rows,operational,
            source_commit=run.get('source_commit'),source_sha256=run.get('source_sha256'))
    target=read_run(config)
    for axis,name in enumerate(CONFIGS):
        if selections is not None:
            from .store import load_selections
            selected=load_selections(selections,LAYERS[axis],fingerprint_value=metadata['fingerprint'])
            write_selection(target,name,selected['indices'],selected['absolute_positions'])
            continue
        # Stable descending sort retains lower visual indices on tied scores.
        indices=np.sort(np.argsort(-exported.attention[:,axis,:],axis=1,kind='stable')[:,:32],axis=1).astype(np.uint16)
        positions=np.asarray([np.flatnonzero(np.asarray(row['input_ids'])==row['image_token_id'])[keep]
                              for row,keep in zip(rows,indices)],dtype=np.int32)
        write_selection(target,name,indices,positions)
    return config


def validate_all_results(config_path):
    run=read_run(config_path); root=Path(run['output_root']); results={}
    for name in CONFIGS:
        directory=root/'results'/name
        if not (directory/'metadata.json').exists(): directory=finalize_config(run,name)
        info,records=validate_result(directory,run)
        if len(records)!=run['count']: raise ValueError('Full test coverage required for every configuration')
        results[name]=(directory,info,records)
    return run,results


def score_results(config_path, *, scorer_command=None):
    from smolvlm_topk_benchmark.runtime import _scorer_command, _scoring_fingerprint
    from smolvlm_topk_benchmark.scoring import METRICS
    from tqdm.auto import tqdm
    run,results=validate_all_results(config_path); root=Path(run['output_root'])
    command=scorer_command or _scorer_command
    probe=command(root/'metric_preflight.json'); scoring_identity=_scoring_fingerprint(probe)
    summaries=[]; rows=input_rows(run)
    for name in tqdm(CONFIGS,desc='Score configurations',unit='config'):
        directory,info,records=results[name]
        if any((r['sample_id'],r['eval_id'],r['reference'])!=(s['sample_id'],s['eval_id'],s['answer']) for r,s in zip(records,rows)):
            raise ValueError('Prediction reference mapping differs')
        prediction_hash=digest(directory/'predictions.jsonl'); path=root/'scores'/f'{name}.json'
        key=fingerprint(dict(predictions=prediction_hash,scorer=scoring_identity))
        cached=json.loads(path.read_text()) if path.exists() else None
        if cached and cached.get('fingerprint')==key:
            result=cached['result']
            if cached.get('result_sha256')!=fingerprint(result): raise ValueError('Score checksum mismatch')
        else:
            result=command(root/'scores'/f'{name}.raw.json',input_path=directory/'predictions.jsonl',expected_count=run['count'])
            if _scoring_fingerprint(result['scoring_identity'])!=scoring_identity: raise ValueError('Scorer changed during run')
            atomic_json(path,dict(fingerprint=key,result=result,result_sha256=fingerprint(result),prediction_sha256=prediction_hash))
        diagnostics=dict(result['diagnostics'],eos_count=sum(r['stop_reason']=='eos' for r in records),
                         max_new_tokens_count=sum(r['stop_reason']=='max_new_tokens' for r in records))
        atomic_json(root/'scores'/f'{name}.diagnostics.json',diagnostics)
        summaries.append(dict(config=name,layer=int(name[-2:]),k=32,count=len(records),**result['metrics'],
            empty_prediction_count=diagnostics['empty_prediction_count'],eos_count=diagnostics['eos_count'],
            max_new_tokens_count=diagnostics['max_new_tokens_count']))
    summary=dict(experiment_fingerprint=run['fingerprint'],scoring_fingerprint=scoring_identity,
                 results=summaries,generation=dict(greedy=True,max_new_tokens=512,batch_size=64,precision='float32_no_tf32'))
    atomic_json(root/'summary.json',summary)
    stream=io.StringIO(); fields=['config','layer','k','count',*METRICS,'empty_prediction_count','eos_count','max_new_tokens_count']
    writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(summaries)
    atomic_bytes(root/'summary.csv',stream.getvalue().encode())
    return summary


def bundle_results(config_path, attention_export, destination=None):
    from .store import open_attention
    open_attention(attention_export)
    run,results=validate_all_results(config_path); root=Path(run['output_root'])
    if not (root/'summary.json').exists(): raise ValueError('Score all ten configurations before ZIP')
    summary=json.loads((root/'summary.json').read_text())
    if summary['experiment_fingerprint']!=run['fingerprint'] or [r['config'] for r in summary['results']]!=list(CONFIGS):
        raise ValueError('Summary identity differs')
    target=Path(destination) if destination else root.parent/'combined_attention_topk_results.zip'
    files={f'benchmark/{name}':root/name for name in ('benchmark_run.json','inputs.jsonl','summary.json','summary.csv','metric_preflight.json')}
    if digest(Path(attention_export)/'manifest.json')!=run['identity']['attention_manifest_sha256']:
        raise ValueError('Benchmark refers to a different attention export')
    attention_files=['attention.npy','samples.jsonl','metadata.json','manifest.json','README.md']
    for name in attention_files:
        path=Path(attention_export)/name
        if path.exists(): files['attention/'+name]=path
    for name,(directory,info,_) in results.items():
        cached=json.loads((root/'scores'/f'{name}.json').read_text())
        if (cached.get('prediction_sha256')!=info['sha256']['predictions.jsonl']
                or cached.get('result_sha256')!=fingerprint(cached['result'])
                or cached.get('fingerprint')!=fingerprint(dict(predictions=info['sha256']['predictions.jsonl'],scorer=summary['scoring_fingerprint']))):
            raise ValueError('Scored prediction identity differs')
        from smolvlm_topk_benchmark.scoring import METRICS
        from smolvlm_topk_benchmark.runtime import _scoring_fingerprint
        item=summary['results'][CONFIGS.index(name)]
        result=cached['result']
        if (_scoring_fingerprint(result['scoring_identity'])!=summary['scoring_fingerprint']
                or any(item.get(metric)!=result['metrics'][metric] for metric in METRICS)
                or item.get('count')!=info['count'] or item.get('layer')!=int(name[-2:]) or item.get('k')!=32):
            raise ValueError('Summary metrics differ from verified scores')
        for file in ('metadata.json',*info['sha256']): files[f'benchmark/results/{name}/{file}']=directory/file
        for path in (root/'scores').glob(name+'.*.json'): files['benchmark/scores/'+path.name]=path
        files[f'benchmark/scores/{name}.json']=root/'scores'/f'{name}.json'
    with FileLock(str(root/'bundle.lock'),timeout=0):
        temporary=target.with_suffix('.partial.zip'); target.parent.mkdir(parents=True,exist_ok=True)
        hashes={name:digest(path) for name,path in files.items()}
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
            for name,path in files.items(): archive.write(path,arcname=name)
            archive.writestr('bundle_manifest.json',json.dumps(dict(sha256=hashes,experiment_fingerprint=run['fingerprint']),indent=2))
            archive.writestr('README.md','Extract before opening attention.npy as a memory map. This completed bundle excludes model/cache/work shards. Partial resume requires the full original output folder.\n')
        os.replace(temporary,target)
    return target
