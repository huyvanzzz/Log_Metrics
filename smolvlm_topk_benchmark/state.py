"""Atomic immutable batch journals and portable generation results; no Torch imports."""
import json
from pathlib import Path
import os
import tempfile

import numpy as np
from filelock import FileLock

from smolvlm_attention.sample_store import atomic_bytes, atomic_json, canonical, digest, fingerprint, pid_is_alive
from .config import BATCH_SIZE, CONFIGS, K, config_layer


def initialize_run(output, identity, rows, operational, *, mode='auto', source_commit=None, source_sha256=None):
    if mode not in ('auto', 'measure', 'resume'):
        raise ValueError('Invalid generation mode')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'benchmark_run.json'
    payload = b''.join(canonical(r)+b'\n' for r in rows)
    identity = dict(identity, inputs_sha256=__import__('hashlib').sha256(payload).hexdigest(), count=len(rows))
    if (identity.get('batch_size') != BATCH_SIZE or not rows
            or [r['row_index'] for r in rows] != list(range(len(rows)))
            or len({r['sample_id'] for r in rows}) != len(rows)
            or len({r['eval_id'] for r in rows}) != len(rows)):
        raise ValueError('Invalid fixed-batch input mapping')
    candidate = dict(version=1, identity=identity, fingerprint=fingerprint(identity),
        count=len(rows), operational=operational, source_commit=source_commit, source_sha256=source_sha256)
    with FileLock(str(output / 'prepare.lock'), timeout=0):
        if path.exists():
            saved = read_run(path)
            if mode == 'measure':
                raise ValueError('Run already exists; use resume or new OUTPUT')
            if saved['fingerprint'] != candidate['fingerprint']:
                raise ValueError('Run incompatible; choose a new OUTPUT')
            if saved['operational'] != operational:
                logs=output/'logs'; logs.mkdir(exist_ok=True)
                with FileLock(str(logs/'launch.lock'), timeout=0):
                    if any(pid_is_alive(json.loads(p.read_text())['pid']) for p in logs.glob('*.pid.json')):
                        raise RuntimeError('Workers running; finish before updating artifact paths')
                    previous=json.loads(path.read_text(encoding='utf-8'))
                    atomic_json(path, dict(previous, operational=operational))
            return path
        if mode == 'resume':
            raise FileNotFoundError('No existing run')
        if (output/'configs').exists() or (output/'results').exists():
            raise ValueError('Orphan state; choose a new OUTPUT')
        inputs=output/'inputs.jsonl'
        if inputs.exists() and inputs.read_bytes()!=payload:
            raise ValueError('Orphan inputs incompatible; choose new OUTPUT')
        atomic_bytes(inputs,payload)
        atomic_json(path,candidate)
    return path


def read_run(path):
    path=Path(path).resolve()
    run=json.loads(path.read_text(encoding='utf-8'))
    if run['fingerprint']!=fingerprint(run['identity']) or run['identity']['batch_size']!=BATCH_SIZE:
        raise ValueError('Run identity mismatch')
    run['output_root']=str(path.parent)
    run['config_path']=str(path)
    if digest(path.parent/'inputs.jsonl')!=run['identity']['inputs_sha256']:
        raise ValueError('Input mapping hash mismatch')
    rows=input_rows(run)
    if len(rows)!=run['count'] or len(rows)!=run['identity']['count']:
        raise ValueError('Input count mismatch')
    return run


def input_rows(run):
    rows=[json.loads(line) for line in (Path(run['output_root'])/'inputs.jsonl').read_text(encoding='utf-8').splitlines()]
    if ([r['row_index'] for r in rows]!=list(range(len(rows))) or
            len({r['sample_id'] for r in rows})!=len(rows) or len({r['eval_id'] for r in rows})!=len(rows)):
        raise ValueError('Invalid input IDs/order')
    return rows


def config_identity(run, name):
    return fingerprint(dict(experiment=run['fingerprint'],config=name,layer=config_layer(name)))


def config_dir(run, name):
    config_layer(name)
    return Path(run['output_root'])/'configs'/name


def _atomic_npy(path, array):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.npy')
    try:
        with os.fdopen(fd,'wb') as stream:
            np.save(stream,array,allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def _validate_selection(indices,positions,count,rows=None):
    if (indices.shape!=(count,K) or positions.shape!=indices.shape
            or indices.dtype!=np.uint16 or positions.dtype!=np.int32
            or (indices>=384).any() or (positions<0).any()
            or (np.diff(indices.astype(np.int32),axis=-1)<=0).any()
            or (np.diff(positions,axis=-1)<=0).any()):
        raise ValueError('Invalid ordered Top-32 selection')
    if rows is not None:
        for row,visual,absolute in zip(rows,indices,positions):
            image=np.flatnonzero(np.asarray(row['input_ids'])==row['image_token_id'])
            if len(image)!=384 or not np.array_equal(image[visual],absolute):
                raise ValueError('Selected visual/absolute prompt positions differ')


def write_selection(run,name,indices=None,positions=None):
    directory=config_dir(run,name); directory.mkdir(parents=True,exist_ok=True)
    path=directory/'selection.json'
    if name=='base':
        info=dict(config_identity=config_identity(run,name),kind='all_384',sha256={})
    else:
        indices=np.asarray(indices); positions=np.asarray(positions)
        _validate_selection(indices,positions,run['count'],input_rows(run))
        info=dict(config_identity=config_identity(run,name),kind='topk',shape=list(indices.shape))
    with FileLock(str(directory/'selection.lock'),timeout=0):
        if path.exists():
            old_indices,old_positions=load_selection(run,name)
            if name!='base' and (not np.array_equal(indices,old_indices) or not np.array_equal(positions,old_positions)):
                raise ValueError('Selection incompatible; no data overwritten')
            return path
        if (directory/'batches').exists():
            raise ValueError('Selection missing for existing batches')
        if name!='base':
            _atomic_npy(directory/'selection.npy',indices)
            _atomic_npy(directory/'prompt_positions.npy',positions)
            info['sha256']={n:digest(directory/n) for n in ('selection.npy','prompt_positions.npy')}
        atomic_json(path,info)
    return path


def load_selection(run,name):
    directory=config_dir(run,name)
    info=json.loads((directory/'selection.json').read_text(encoding='utf-8'))
    if info['config_identity']!=config_identity(run,name):
        raise ValueError('Selection identity mismatch')
    if info['kind']!=('all_384' if name=='base' else 'topk'):
        raise ValueError('Selection kind differs')
    expected=set() if name=='base' else {'selection.npy','prompt_positions.npy'}
    if set(info['sha256'])!=expected:
        raise ValueError('Selection manifest differs')
    for file,sha in info['sha256'].items():
        if digest(directory/file)!=sha:raise ValueError('Selection hash mismatch')
    if name=='base':return None,None
    arrays=tuple(np.load(directory/n,mmap_mode='r',allow_pickle=False) for n in ('selection.npy','prompt_positions.npy'))
    if arrays[0].shape!=tuple(info['shape']) or arrays[1].shape!=arrays[0].shape or arrays[0].shape[0]!=run['count']:
        raise ValueError('Selection shape mismatch')
    _validate_selection(*arrays,run['count'],input_rows(run))
    return arrays


def _validate_records(run, name, start, records, rows):
    config_layer(name)
    count=min(BATCH_SIZE,run['count']-start)
    if start<0 or start%BATCH_SIZE or count<1 or len(records)!=count:
        raise ValueError('Batch boundary/count differs')
    for source,record in zip(rows[start:start+count],records):
        if (record['sample_id']!=source['sample_id'] or record['eval_id']!=source['eval_id']
                or record['reference']!=source['answer'] or record['prediction']!=record['raw_prediction'].strip()
                or record['generated_token_count']!=len(record['generated_ids'])
                or not 1<=record['generated_token_count']<=512
                or record['stop_reason'] not in ('eos','max_new_tokens')):
            raise ValueError('Prediction mapping/continuation differs')


def save_batch(run,name,start,records,*,rows=None):
    rows=input_rows(run) if rows is None else rows
    _validate_records(run,name,start,records,rows)
    payload=dict(config_identity=config_identity(run,name),start=start,records=records)
    envelope=dict(payload,checksum=fingerprint(payload))
    path=config_dir(run,name)/'batches'/f'{start:06d}.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=envelope:
            raise ValueError('Batch exists with different predictions; no overwrite')
        return path
    return atomic_json(path,envelope)


def load_batches(run,name):
    output={}; rows=input_rows(run)
    for path in sorted((config_dir(run,name)/'batches').glob('*.json')):
        value=json.loads(path.read_text(encoding='utf-8'))
        checksum=value.pop('checksum')
        if checksum!=fingerprint(value):raise ValueError('Batch checksum mismatch')
        start=value['start']
        if value['config_identity']!=config_identity(run,name) or path.stem!=f'{start:06d}' or start in output:
            raise ValueError('Batch identity/order differs')
        _validate_records(run,name,start,value['records'],rows)
        output[start]=value['records']
    return output


def finalize_config(run,name):
    output=Path(run['output_root']); target=output/'results'/name
    with FileLock(str(output/f'finalize_{name}.lock'),timeout=0):
        if (target/'metadata.json').exists():
            validate_result(target,run)
            return target
        batches=load_batches(run,name)
        if set(batches)!=set(range(0,run['count'],BATCH_SIZE)):
            raise ValueError(f'{name}: generation incomplete')
        load_selection(run,name)
        target.mkdir(parents=True,exist_ok=True)
        records=[dict(row,config=name,layer=config_layer(name),
            selection_file='selection.json' if name=='base' else 'selection.npy',
            prompt_positions_file=None if name=='base' else 'prompt_positions.npy',row_index=start+i)
            for start in sorted(batches) for i,row in enumerate(batches[start])]
        atomic_bytes(target/'predictions.jsonl',b''.join(canonical(r)+b'\n' for r in records))
        files=['predictions.jsonl','selection.json']
        if name!='base':files+=['selection.npy','prompt_positions.npy']
        for file in files[1:]:
            atomic_bytes(target/file,(config_dir(run,name)/file).read_bytes())
        atomic_json(target/'metadata.json',dict(status='complete',config=name,layer=config_layer(name),
            k=384 if name=='base' else 32,count=len(records),experiment_identity=run['identity'],
            experiment_fingerprint=run['fingerprint'],config_identity=config_identity(run,name),
            source_commit=run.get('source_commit'),source_sha256=run.get('source_sha256'),
            sha256={file:digest(target/file) for file in files}))
    return target


def validate_result(directory,run=None,*,expected_config=None):
    directory=Path(directory)
    info=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
    expected_config=directory.name if expected_config is None else expected_config
    if (info['config']!=expected_config or info['layer']!=config_layer(expected_config)
            or info['k']!=(384 if expected_config=='base' else K)):
        raise ValueError('Result config/layer/K differs from expected config')
    if (info['status']!='complete' or info['experiment_fingerprint']!=fingerprint(info['experiment_identity'])
            or info['config_identity']!=fingerprint(dict(experiment=info['experiment_fingerprint'],
                  config=info['config'],layer=config_layer(info['config'])))
            or (run and info['experiment_fingerprint']!=run['fingerprint'])):
        raise ValueError('Result identity mismatch')
    expected={'predictions.jsonl','selection.json'}
    if info['config']!='base':expected|={'selection.npy','prompt_positions.npy'}
    if set(info['sha256'])!=expected:raise ValueError('Result manifest differs')
    for name,sha in info['sha256'].items():
        if digest(directory/name)!=sha:raise ValueError(f'Result hash mismatch: {name}')
    selection=json.loads((directory/'selection.json').read_text(encoding='utf-8'))
    selection_files=set() if info['config']=='base' else {'selection.npy','prompt_positions.npy'}
    if (selection['config_identity']!=info['config_identity'] or set(selection['sha256'])!=selection_files
            or selection['kind']!=('all_384' if info['config']=='base' else 'topk')
            or any(selection['sha256'][n]!=info['sha256'][n] for n in selection_files)):
        raise ValueError('Result selection metadata differs')
    if selection_files:
        a=np.load(directory/'selection.npy',mmap_mode='r',allow_pickle=False)
        p=np.load(directory/'prompt_positions.npy',mmap_mode='r',allow_pickle=False)
        _validate_selection(a,p,info['count'],input_rows(run) if run else None)
    records=[json.loads(line) for line in (directory/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    if (len(records)!=info['count'] or len(records)!=info['experiment_identity']['count']
            or [r['row_index'] for r in records]!=list(range(len(records)))
            or len({r['sample_id'] for r in records})!=len(records)
            or len({r['eval_id'] for r in records})!=len(records)
            or any(r['config']!=info['config'] or r['layer']!=info['layer'] for r in records)):
        raise ValueError('Result coverage mismatch')
    return info,records
