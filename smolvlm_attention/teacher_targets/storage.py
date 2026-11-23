"""Durable normalized log-target shards, separate from probability exporters."""
from contextlib import contextmanager
import hashlib
import json
import os
import uuid
from pathlib import Path
import numpy as np
from filelock import FileLock, Timeout
from smolvlm_attention.sample_store import atomic_json, canonical, digest, fingerprint, sync_file


def validate_log_probs(values, visual_tokens=None):
    values = np.asarray(values)
    if values.ndim != 2 or values.dtype != np.dtype('<f4') or (visual_tokens is not None and values.shape[1] != visual_tokens):
        raise ValueError('Expected little-endian FP32 [rows, visual_tokens] log probabilities')
    if not np.isfinite(values).all():
        raise ValueError('Non-finite teacher targets')
    maximum = values.max(axis=1)
    normalizer = maximum + np.log(np.exp(values.astype(np.float64)-maximum[:,None]).sum(axis=1))
    if not np.all(np.abs(normalizer) <= 2e-5):
        raise ValueError('Teacher log probabilities are not visual-normalized')


def _checksum(rows, values):
    h = hashlib.sha256(np.asarray(rows,dtype='<i8').tobytes())
    h.update(np.asarray(values,dtype='<f4').tobytes())
    return h.hexdigest()


class TargetShard:
    def __init__(self, directory, identity, rows, visual_tokens=384):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self._writer_lock = FileLock(str(self.directory/'writer.lock'), timeout=0)
        try:
            with self._writer_lock:
                self._initialize(identity, rows, visual_tokens)
        except Timeout as exc:
            raise RuntimeError('Shard already has a live writer') from exc

    def _initialize(self, identity, rows, visual_tokens):
        self.identity = identity
        self.rows = np.asarray(rows,dtype='<i8')
        self.visual_tokens = visual_tokens
        if self.rows.ndim != 1 or len(set(self.rows.tolist())) != len(self.rows) or (self.rows < 0).any():
            raise ValueError('Shard rows must be distinct nonnegative integers')
        self.lookup = {int(row):i for i,row in enumerate(self.rows)}
        metadata = {'format':'teacher_log_targets_shard_v1','identity':identity,'rows':self.rows.tolist(),'visual_tokens':visual_tokens}
        metadata_path = self.directory/'shard.json'
        self.path = self.directory/'log_probs.npy'
        if metadata_path.exists():
            if json.loads(metadata_path.read_text(encoding='utf-8')) != metadata:
                raise ValueError('Shard identity or geometry mismatch')
            self.array = np.load(self.path,mmap_mode='r+',allow_pickle=False)
        else:
            intent = self.directory/'initialization.json'
            commits = self.directory/'commits'
            if intent.exists():
                if json.loads(intent.read_text(encoding='utf-8')) != metadata:
                    raise ValueError('Shard initialization identity mismatch')
                if commits.exists() and any(commits.iterdir()):
                    raise ValueError('Commit records exist without published shard metadata')
                if self.path.exists():
                    # Never erase orphan data; no receipt could have committed
                    # before shard metadata was published.
                    os.replace(self.path,self.directory/f'log_probs.uncommitted-{uuid.uuid4().hex}.npy')
            else:
                if self.path.exists() or (commits.exists() and any(commits.iterdir())):
                    raise ValueError('Unknown orphan shard payload; preserve and inspect')
                atomic_json(intent,metadata)
            self.array = np.lib.format.open_memmap(self.path,mode='w+',dtype='<f4',shape=(len(self.rows),visual_tokens))
            self.array.flush()
            sync_file(self.path)
            atomic_json(metadata_path,metadata)
        if self.array.shape != (len(self.rows),visual_tokens) or self.array.dtype != np.dtype('<f4'):
            raise ValueError('Shard payload shape/dtype mismatch')
        self.commits = self.directory/'commits'
        self.commits.mkdir(exist_ok=True)
        self.done = np.zeros(len(self.rows),dtype=bool)
        for receipt_path in sorted(self.commits.glob('*.json')):
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
            committed = receipt['rows']
            try: local = [self.lookup[int(row)] for row in committed]
            except KeyError as exc: raise ValueError('Receipt row outside shard') from exc
            if len(set(committed)) != len(committed) or self.done[local].any() or receipt.get('identity_sha256') != fingerprint(identity):
                raise ValueError('Invalid or duplicate commit receipt')
            block = np.asarray(self.array[local])
            validate_log_probs(block,visual_tokens)
            if _checksum(committed,block) != receipt['sha256']:
                raise ValueError('Committed shard checksum mismatch')
            self.done[local] = True

    @property
    def pending_rows(self):
        return self.rows[~self.done]

    @contextmanager
    def lock(self):
        # OS-backed locks release on process death and do not depend on stale
        # PID contents when artifacts move between Kaggle sessions.
        if self._writer_lock.is_locked:
            raise RuntimeError('Shard writer lock is already held')
        try:
            self._writer_lock.acquire(timeout=0)
        except Timeout as exc:
            raise RuntimeError('Shard already has a live writer') from exc
        try:
            # Another writer may have committed since this object was created.
            self._initialize(self.identity, self.rows, self.visual_tokens)
            yield self
        finally:
            self._writer_lock.release()

    def commit(self, rows, log_probs):
        if not self._writer_lock.is_locked:
            with self.lock():
                return self._commit(rows, log_probs)
        return self._commit(rows, log_probs)

    def _commit(self, rows, log_probs):
        rows = [int(row) for row in rows]
        if not rows or len(set(rows)) != len(rows): raise ValueError('Empty or duplicate commit rows')
        try: local = [self.lookup[row] for row in rows]
        except KeyError as exc: raise ValueError('Commit row outside shard') from exc
        if self.done[local].any(): raise ValueError('Cannot overwrite committed rows')
        values = np.asarray(log_probs)
        validate_log_probs(values,self.visual_tokens)
        if values.shape[0] != len(rows): raise ValueError('Commit row count mismatch')
        self.array[local] = values
        self.array.flush()
        sync_file(self.path)
        receipt = {'rows':rows,'sha256':_checksum(rows,values),'identity_sha256':fingerprint(self.identity)}
        path = self.commits/(fingerprint(receipt)+'.json')
        atomic_json(path,receipt)
        self.done[local] = True


def _sample_key(sample, row, identity):
    if not isinstance(sample, dict) or type(sample.get('row')) is not int or sample['row'] != row:
        raise ValueError('Sample mapping row/type mismatch')
    if any(key in sample for key in ('answer','ground_truth','input_ids','prompt_ids')):
        raise ValueError('Private prompt/answer data in portable mapping')
    if not isinstance(sample.get('sample_id'), str) or not sample['sample_id']:
        raise ValueError('Invalid sample ID type')
    split = sample.get('split', identity.get('split'))
    if split is not None and split not in ('train','val'):
        raise ValueError('Invalid target split')
    if identity.get('split') is not None and split != identity['split']:
        raise ValueError('Sample split does not match export identity')
    return split, sample['sample_id']


def finalize_split(directory, identity, shards, samples, count, visual_tokens=384):
    directory = Path(directory)
    if (directory/'manifest.json').exists():
        return validate_export(directory,identity)
    directory.mkdir(parents=True,exist_ok=True)
    # Reopen receipts immediately before publication; a long-lived worker object
    # does not prove its committed bytes are still intact.
    shards = [TargetShard(s.directory, identity, s.rows, visual_tokens) for s in shards]
    seen = np.zeros(count,dtype=bool)
    for shard in shards:
        if shard.identity != identity or shard.pending_rows.size: raise ValueError('Incomplete or mismatched shard')
        if (shard.rows >= count).any() or seen[shard.rows].any(): raise ValueError('Duplicate or out-of-range shard rows')
        seen[shard.rows] = True
    if not seen.all(): raise ValueError('Missing target rows')
    temporary = directory/'log_probs.partial.npy'
    output = np.lib.format.open_memmap(temporary,mode='w+',dtype='<f4',shape=(count,visual_tokens))
    for shard in shards:
        for start in range(0,len(shard.rows),1024):
            block = np.asarray(shard.array[start:start+1024])
            validate_log_probs(block,visual_tokens)
            output[shard.rows[start:start+1024]] = block
    output.flush(); sync_file(temporary); del output
    mapping_tmp = directory/'samples.partial.jsonl'
    identifiers = set()
    with mapping_tmp.open('wb') as stream:
        actual = 0
        for row,sample in enumerate(samples):
            if row >= count or sample.get('row') != row: raise ValueError('Sample mapping order/coverage mismatch')
            key = _sample_key(sample, row, identity)
            if key in identifiers: raise ValueError('Duplicate sample identity')
            identifiers.add(key)
            stream.write(canonical(sample)+b'\n'); actual += 1
        if actual != count: raise ValueError('Missing sample mapping')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,directory/'log_probs.npy')
    os.replace(mapping_tmp,directory/'samples.jsonl')
    manifest = {'format_version':'teacher_log_targets_v1','complete':True,'identity':identity,
                'shape':[count,visual_tokens],'axes':['sample','visual_token'],'dtype':'<f4',
                'normalization':'mean_heads_all_valid_keys_then_visual_normalization_log_v1',
                'checksums':{name:digest(directory/name) for name in ('log_probs.npy','samples.jsonl')}}
    atomic_json(directory/'manifest.json',manifest)
    return validate_export(directory,identity)


def validate_export(directory, expected_identity=None):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format_version') != 'teacher_log_targets_v1' or manifest.get('complete') is not True:
        raise ValueError('Incomplete or unsupported teacher export')
    if expected_identity is not None and manifest['identity'] != expected_identity: raise ValueError('Export identity mismatch')
    identity = manifest['identity']
    if identity.get('score_layer_index',22) != 22 or identity.get('score_layer_number',23) != 23:
        raise ValueError('Expected index 22, layer number 23')
    protocol = identity.get('protocol')
    if protocol is not None:
        split = identity.get('split')
        if (not isinstance(protocol,dict) or split not in ('train','val') or
                protocol.get('score_layer_index') != 22 or protocol.get('score_layer_number') != 23 or
                protocol.get('visual_tokens') != 384 or
                manifest['shape'] != [protocol.get('splits',{}).get(split),384]):
            raise ValueError('Teacher protocol split/count/layer/geometry mismatch')
        if (identity.get('axes') != ['sample','visual_token'] or identity.get('dtype') != '<f4' or
                identity.get('normalization') != protocol.get('normalization')):
            raise ValueError('Teacher identity axes/dtype/normalization mismatch')
    if manifest.get('axes') != ['sample','visual_token'] or manifest.get('dtype') != '<f4' or manifest.get('normalization') != 'mean_heads_all_valid_keys_then_visual_normalization_log_v1':
        raise ValueError('Export axes/dtype/normalization mismatch')
    for name in ('log_probs.npy','samples.jsonl'):
        if digest(directory/name) != manifest['checksums'][name]: raise ValueError('Export checksum mismatch')
    data = np.load(directory/'log_probs.npy',mmap_mode='r',allow_pickle=False)
    if list(data.shape) != manifest['shape']: raise ValueError('Export geometry mismatch')
    for start in range(0,len(data),1024): validate_log_probs(data[start:start+1024],data.shape[1])
    count=0; identifiers=set()
    with (directory/'samples.jsonl').open(encoding='utf-8') as stream:
        for row,line in enumerate(stream):
            sample=json.loads(line)
            key = _sample_key(sample, row, manifest['identity'])
            if sample.get('row') != row or key in identifiers: raise ValueError('Invalid export sample mapping')
            identifiers.add(key); count += 1
    if count != len(data): raise ValueError('Export mapping count mismatch')
    return manifest
