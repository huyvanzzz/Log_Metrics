"""Shared immutable selections and independent journals for paired prune locations."""
import json
from pathlib import Path

import numpy as np
from filelock import FileLock

from smolvlm_attention.sample_store import atomic_bytes, atomic_json, canonical, digest, fingerprint
from smolvlm_topk_benchmark.state import initialize_run, read_run, input_rows, _atomic_npy

CONFIGS = ('score22_prune_input', 'score22_prune_after2')
BATCH_SIZE = 64
K = 32


def _name(name):
    if name not in CONFIGS:
        raise ValueError('Unknown deferred config')
    return name


def config_dir(run, name):
    return Path(run['output_root'])/'configs'/_name(name)


def _validate_selection(indices, count):
    if (indices.dtype != np.uint16 or indices.shape != (count, K)
            or (indices >= 384).any() or (np.diff(indices.astype(np.int32), axis=1) <= 0).any()):
        raise ValueError('Invalid ordered shared Top-32 selection')


def write_selection(run, indices):
    indices = np.asarray(indices)
    _validate_selection(indices, run['count'])
    root = Path(run['output_root']); path = root/'selection.json'
    with FileLock(str(root/'selection.lock'), timeout=0):
        if path.exists():
            if not np.array_equal(load_selection(run), indices):
                raise ValueError('Selection incompatible; no overwrite')
            return path
        if (root/'configs').exists() or (root/'results').exists():
            raise ValueError('Selection missing for existing journals')
        _atomic_npy(root/'selection.npy', indices)
        atomic_json(path, dict(version=1, experiment_fingerprint=run['fingerprint'],
            score_layer_index=22, kind='shared_global_top32', shape=list(indices.shape),
            sha256=digest(root/'selection.npy')))
    return path


def load_selection(run):
    root=Path(run['output_root'])
    info=json.loads((root/'selection.json').read_text(encoding='utf-8'))
    if (info.get('version')!=1 or info.get('experiment_fingerprint')!=run['fingerprint']
            or info.get('score_layer_index')!=22 or info.get('kind')!='shared_global_top32'):
        raise ValueError('Selection identity mismatch')
    if digest(root/'selection.npy')!=info['sha256']:
        raise ValueError('Selection hash mismatch')
    indices=np.load(root/'selection.npy',mmap_mode='r',allow_pickle=False)
    _validate_selection(indices,run['count'])
    if list(indices.shape)!=info['shape']:
        raise ValueError('Selection shape mismatch')
    return indices


def config_identity(run,name):
    _name(name)
    # Checking the immutable selection before a journal is read prevents accidental reuse.
    load_selection(run)
    return fingerprint(dict(experiment=run['fingerprint'],config=name,
        selection_sha256=digest(Path(run['output_root'])/'selection.npy')))


def _validate_records(run,name,start,records,rows):
    _name(name)
    count=min(BATCH_SIZE,run['count']-start)
    if start<0 or start%BATCH_SIZE or count<1 or len(records)!=count:
        raise ValueError('Batch boundary/count differs')
    for source,record in zip(rows[start:start+count],records):
        ids=record['generated_ids']
        if (record['sample_id']!=source['sample_id'] or record['eval_id']!=source['eval_id']
                or record['reference']!=source['answer'] or record['prediction']!=record['raw_prediction'].strip()
                or record['generated_token_count']!=len(ids) or not 1<=len(ids)<=512
                or any(not isinstance(v,int) or isinstance(v,bool) or v<0 for v in ids)
                or record['stop_reason'] not in ('eos','max_new_tokens')):
            raise ValueError('Prediction mapping/continuation differs')


def save_batch(run,name,start,records,*,rows=None):
    rows=input_rows(run) if rows is None else rows
    _validate_records(run,name,start,records,rows)
    payload=dict(config_identity=config_identity(run,name),start=start,records=records)
    envelope=dict(payload,checksum=fingerprint(payload))
    path=config_dir(run,name)/'batches'/f'{start:06d}.json'
    with FileLock(str(Path(run['output_root'])/f'journal_{name}.lock'),timeout=0):
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8'))!=envelope:
                raise ValueError('Batch exists with different predictions; no overwrite')
            return path
        return atomic_json(path,envelope)


def load_batches(run,name):
    directory=config_dir(run,name); identity=config_identity(run,name)
    output={}; rows=input_rows(run)
    for path in sorted((directory/'batches').glob('*.json')):
        value=json.loads(path.read_text(encoding='utf-8'))
        checksum=value.pop('checksum')
        if checksum!=fingerprint(value):
            raise ValueError('Batch checksum mismatch')
        start=value['start']
        if value['config_identity']!=identity or path.stem!=f'{start:06d}' or start in output:
            raise ValueError('Batch identity/order differs')
        _validate_records(run,name,start,value['records'],rows)
        output[start]=value['records']
    return output


def finalize_config(run,name):
    _name(name)
    root=Path(run['output_root']); target=root/'results'/name
    with FileLock(str(root/f'finalize_{name}.lock'),timeout=0):
        if (target/'metadata.json').exists():
            validate_result(target,run,expected_config=name)
            return target
        batches=load_batches(run,name)
        if set(batches)!=set(range(0,run['count'],BATCH_SIZE)):
            raise ValueError(f'{name}: generation incomplete')
        target.mkdir(parents=True,exist_ok=True)
        records=[dict(record,config=name,score_layer_index=22,
            prune_after_layer_index=None if name==CONFIGS[0] else 2,row_index=start+i,
            selection_file='selection.npy') for start in sorted(batches) for i,record in enumerate(batches[start])]
        atomic_bytes(target/'predictions.jsonl',b''.join(canonical(r)+b'\n' for r in records))
        for file in ('selection.npy','selection.json'):
            atomic_bytes(target/file,(root/file).read_bytes())
        atomic_json(target/'metadata.json',dict(status='complete',config=name,score_layer_index=22,
            prune_after_layer_index=None if name==CONFIGS[0] else 2,k=K,count=len(records),
            experiment_identity=run['identity'],experiment_fingerprint=run['fingerprint'],
            config_identity=config_identity(run,name),source_commit=run.get('source_commit'),
            source_sha256=run.get('source_sha256'),
            sha256={f:digest(target/f) for f in ('predictions.jsonl','selection.npy','selection.json')}))
    return target


def validate_result(directory,run=None,*,expected_config=None):
    directory=Path(directory)
    info=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
    name=_name(directory.name if expected_config is None else expected_config)
    prune=None if name==CONFIGS[0] else 2
    if (info['config']!=name or info['score_layer_index']!=22 or info['prune_after_layer_index']!=prune
            or info['k']!=K or info['status']!='complete'
            or info['experiment_fingerprint']!=fingerprint(info['experiment_identity'])
            or (run and info['experiment_fingerprint']!=run['fingerprint'])):
        raise ValueError('Result identity/config differs')
    expected={'predictions.jsonl','selection.npy','selection.json'}
    if set(info['sha256'])!=expected:
        raise ValueError('Result manifest differs')
    for file,sha in info['sha256'].items():
        if digest(directory/file)!=sha:
            raise ValueError('Result hash mismatch')
    synthetic=dict(output_root=str(directory),fingerprint=info['experiment_fingerprint'],count=info['count'])
    load_selection(synthetic)
    expected_identity=fingerprint(dict(experiment=info['experiment_fingerprint'],config=name,
        selection_sha256=info['sha256']['selection.npy']))
    if expected_identity!=info['config_identity']:
        raise ValueError('Result selection/config identity mismatch')
    records=[json.loads(line) for line in (directory/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    if (len(records)!=info['count'] or len(records)!=info['experiment_identity']['count']
            or [r['row_index'] for r in records]!=list(range(len(records)))
            or len({r['sample_id'] for r in records})!=len(records)
            or len({r['eval_id'] for r in records})!=len(records)
            or any(r['config']!=name or r['score_layer_index']!=22 or r['prune_after_layer_index']!=prune for r in records)):
        raise ValueError('Result coverage mismatch')
    portable_rows=[dict(sample_id=r['sample_id'],eval_id=r['eval_id'],answer=r['reference']) for r in records]
    for start in range(0,info['count'],BATCH_SIZE):
        _validate_records(dict(count=info['count']),name,start,records[start:start+BATCH_SIZE],portable_rows)
    if run:
        rows=input_rows(run)
        for start in range(0,run['count'],BATCH_SIZE):
            _validate_records(run,name,start,records[start:start+BATCH_SIZE],rows)
        if not np.array_equal(load_selection(run),load_selection(synthetic)):
            raise ValueError('Result selection differs from run')
    return info,records
