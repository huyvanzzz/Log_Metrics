"""Independent durable state for same-layer combined visual attention."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from filelock import FileLock

from .aggregate import AttentionAccumulator
from .sample_store import (atomic_bytes, atomic_json, canonical, digest, fingerprint,
                           pid_is_alive, shard_bounds, validate_samples)

LAYER_IDS = tuple(range(0, 29, 2))
BATCH_SIZE = 64
SHARD_COUNT = 2
QUERY_CHUNK_SIZE = 8
POLICY = 'same_layer_mean_heads_direct_plus_nonvisual_two_hop_v1'
GEOMETRY = (15, 384)


def write_run(output, config, rows, mode='auto'):
    if mode not in ('auto','measure','resume'):
        raise ValueError('Invalid measurement mode')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    with FileLock(str(output/'prepare.lock'),timeout=0):
        validate_samples(rows,384)
        if len(rows)<2:raise ValueError('Two nonempty shards required')
        for key,value in [('geometry',list(GEOMETRY)),('batch_size',BATCH_SIZE),('shard_count',SHARD_COUNT)]:
            if key in config and config[key]!=value:raise ValueError(f'Invalid {key}')
        payload=b''.join(canonical(row)+b'\n' for row in rows)
        identity=dict(config['identity'],inputs_sha256=hashlib.sha256(payload).hexdigest(),
            count=len(rows),geometry=list(GEOMETRY),batch_size=BATCH_SIZE,shard_count=SHARD_COUNT,
            layer_ids=list(LAYER_IDS),query_chunk_size=QUERY_CHUNK_SIZE,policy=POLICY)
        candidate=dict(config,identity=identity,fingerprint=fingerprint(identity),count=len(rows),
            geometry=list(GEOMETRY),batch_size=BATCH_SIZE,shard_count=SHARD_COUNT,
            layer_ids=list(LAYER_IDS),query_chunk_size=QUERY_CHUNK_SIZE,policy=POLICY,
            output_root=str(output),inputs=str(output/'inputs.jsonl'),inputs_sha256=identity['inputs_sha256'])
        path=output/'combined_run.json'
        if path.exists():
            existing=read_run(path)
            if mode=='measure':raise ValueError('Run already exists; use resume or new OUTPUT')
            if existing['fingerprint']!=candidate['fingerprint']:raise ValueError('Run incompatible; choose new OUTPUT')
            updates={key:candidate[key] for key in ('artifacts','cache_descriptor')
                     if key in candidate and candidate[key]!=existing.get(key)}
            if updates:
                logs=output/'logs';logs.mkdir(exist_ok=True)
                with FileLock(str(logs/'launch.lock'),timeout=0):
                    if any(pid_is_alive(json.loads(p.read_text())['pid']) for p in logs.glob('*.pid.json')):
                        raise RuntimeError('Worker still running; cannot update paths')
                    atomic_json(path,dict(existing,**updates))
            return path
        if mode=='resume':raise FileNotFoundError('No run to resume')
        if (output/'shards').exists() or (output/'data').exists():raise ValueError('Orphan state; choose new OUTPUT')
        inputs=output/'inputs.jsonl'
        if inputs.exists() and inputs.read_bytes()!=payload:raise ValueError('Orphan inputs incompatible')
        atomic_bytes(inputs,payload);atomic_json(path,candidate)
        return path


def read_run(path):
    path=Path(path).resolve()
    run=json.loads(path.read_text(encoding='utf-8'))
    identity=run['identity']
    expected=dict(geometry=list(GEOMETRY),batch_size=BATCH_SIZE,shard_count=SHARD_COUNT,
                  layer_ids=list(LAYER_IDS),query_chunk_size=QUERY_CHUNK_SIZE,policy=POLICY)
    if run['fingerprint']!=fingerprint(identity) or any(run.get(k)!=v or identity.get(k)!=v for k,v in expected.items()):
        raise ValueError('Run identity mismatch')
    run.update(output_root=str(path.parent),inputs=str(path.parent/'inputs.jsonl'),config_path=str(path))
    if digest(run['inputs'])!=run['inputs_sha256'] or run['inputs_sha256']!=identity['inputs_sha256'] or run['count']!=identity['count']:
        raise ValueError('Input mapping hash/count mismatch')
    input_rows(run)
    return run


def input_rows(run):
    rows=[json.loads(line) for line in Path(run['inputs']).read_text(encoding='utf-8').splitlines()]
    validate_samples(rows,384)
    if len(rows)!=run['count']:raise ValueError('Input count mismatch')
    return rows


def shard_path(run,shard):
    shard_bounds(run['count'],SHARD_COUNT,shard)
    return Path(run['output_root'])/'shards'/f'combined_{shard}.npz'


def _checksum(acc):
    h=hashlib.sha256(canonical(dict(fingerprint=acc.fingerprint,shape=list(acc.shape),
        sample_ids=acc.sample_ids,processed_ids=acc.processed_ids,count=acc.count)))
    h.update(np.ascontiguousarray(acc.sum,dtype='<f8').tobytes())
    return h.hexdigest()


def save_shard(acc,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(dir=path.parent,suffix='.npz')
    try:
        with os.fdopen(fd,'wb') as stream:
            np.savez(stream,sum=acc.sum,mean=acc.mean,count=acc.count,
                processed_ids=np.asarray(acc.processed_ids,dtype=str),sample_ids=np.asarray(acc.sample_ids,dtype=str),
                fingerprint=acc.fingerprint,shape=acc.shape,checksum=_checksum(acc))
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def load_shard(run,shard):
    start,end=shard_bounds(run['count'],SHARD_COUNT,shard)
    ids=[r['sample_id'] for r in input_rows(run)[start:end]]
    path=shard_path(run,shard)
    if not path.exists():return AttentionAccumulator(GEOMETRY,run['fingerprint'],ids)
    try:
        acc=AttentionAccumulator.load(path,GEOMETRY,run['fingerprint'],ids)
        with np.load(path,allow_pickle=False) as saved:
            if saved['checksum'].item()!=_checksum(acc):raise ValueError('Snapshot checksum mismatch')
        if acc.processed_ids!=ids[:acc.count]:raise ValueError('Snapshot order mismatch')
        if (acc.sum<0).any() or (acc.sum.sum(-1)>acc.count*(1+1e-6)).any():raise ValueError('Invalid snapshot scores')
        return acc
    except (OSError,KeyError,ValueError) as exc:
        raise ValueError(f'Damaged snapshot/checksum: {path}: {exc}') from exc


def merge_shards(run):
    total=np.zeros(GEOMETRY,np.float64);seen=set()
    for shard in range(SHARD_COUNT):
        acc=load_shard(run,shard)
        if acc.pending_ids:raise ValueError(f'Incomplete shard {shard}')
        if seen.intersection(acc.processed_ids):raise ValueError('Duplicate shard samples')
        seen.update(acc.processed_ids);total+=acc.sum
    if len(seen)!=run['count']:raise ValueError('Incomplete test coverage')
    return total,len(seen)
