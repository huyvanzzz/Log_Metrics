"""Independent validation and answer-free prompt preparation for DINO caches."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import CAMERA_ORDER, CACHE_FINGERPRINT, VISION_MODEL_ID, VISION_REVISION

FEATURE_FILE = 'features.float16.memmap'
PREPARED_FILES = ('manifest.json', 'train.jsonl', 'val.jsonl', 'test.jsonl')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(path, *, progress=False):
    path = Path(path)
    result = hashlib.sha256()
    with path.open('rb') as stream, tqdm(total=path.stat().st_size,
            desc=f'SHA256 {path.name}', unit='B', unit_scale=True,
            disable=not progress) as bar:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
            bar.update(len(chunk))
    return result.hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.json')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def _stat(path):
    value = path.stat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns)


def records(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError('prepared record must be a JSON object')
                yield value


def record_key(row):
    cameras = row['camera_paths']
    return '|'.join(cameras[camera] for camera in sorted(cameras))


def _validate_manifest(manifest, expected_fingerprint):
    required = ('dataset_revision', 'model_id', 'model_revision', 'preprocessing', 'dtype',
                'image_size', 'view_order', 'shape', 'version', 'processor_policy',
                'feature_extraction', 'data_identity', 'load_dtype', 'fingerprint')
    if set(manifest) != set(required):
        raise ValueError('invalid DINO manifest fields')
    value = {key: manifest[key] for key in manifest if key != 'fingerprint'}
    if manifest['version'] != 2 or hashlib.sha256(canonical(value)).hexdigest() != manifest['fingerprint']:
        raise ValueError('invalid DINO cache fingerprint')
    shape = manifest['shape']
    if (manifest['dtype'] != 'float16' or manifest['load_dtype'] != 'float32' or
            list(manifest['view_order']) != list(CAMERA_ORDER) or len(shape) != 4 or
            any(type(x) is not int or x <= 0 for x in shape) or shape[1] != 6 or
            not all(manifest[k] for k in ('processor_policy','feature_extraction','data_identity'))):
        raise ValueError('invalid DINO cache geometry or storage policy')
    if expected_fingerprint is not None:
        if (manifest['fingerprint'] != expected_fingerprint or manifest['model_id'] != VISION_MODEL_ID or
                manifest['model_revision'] != VISION_REVISION or shape[1:] != [6,64,384]):
            raise ValueError('DINO cache differs from pinned production identity')
    return shape


def _inputs(directory, prepared, expected_fingerprint, *, progress=False):
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    shape = _validate_manifest(manifest, expected_fingerprint)
    index = json.loads((directory / 'index.json').read_text(encoding='utf-8'))
    if (not isinstance(index, dict) or any(type(v) is not int for v in index.values()) or
            sorted(index.values()) != list(range(shape[0]))):
        raise ValueError('DINO cache index must be a dense bijection')
    hashes = {name: digest(prepared / name, progress=progress) for name in PREPARED_FILES}
    identity = manifest['data_identity']
    if hashes != identity.get('prepared_sha256') or digest(directory / 'index.json') != identity.get('index_sha256'):
        raise ValueError('prepared data or index hash mismatch')
    payload = directory / FEATURE_FILE
    if payload.stat().st_size != int(np.prod(shape)) * np.dtype(np.float16).itemsize:
        raise ValueError('DINO cache payload size mismatch')
    return manifest, index, shape, hashes


def validate_cache(cache_dir, prepared_dir, output_dir, expected_fingerprint=CACHE_FINGERPRINT, *, progress=None):
    directory, prepared = Path(cache_dir).resolve(), Path(prepared_dir).resolve()
    receipt = Path(output_dir) / 'validated_cache.json'
    if receipt.exists():
        try:
            cached = json.loads(receipt.read_text(encoding='utf-8'))
            if (cached['cache_dir'] == str(directory) and cached['prepared_dir'] == str(prepared) and
                    cached['expected_fingerprint'] == expected_fingerprint):
                open_validated_cache(receipt)
                return receipt
        except (ValueError, KeyError, OSError, TypeError):
            pass
    enabled = progress is not False
    manifest, index, shape, hashes = _inputs(directory, prepared, expected_fingerprint, progress=enabled)
    counts = {}
    for split in ('train','val','test'):
        seen = set()
        count = 0
        for row in tqdm(records(prepared / f'{split}.jsonl'), desc=f'Validate {split}', unit='question', disable=not enabled):
            sample_id = row.get('sample_id')
            if not isinstance(sample_id,str) or not sample_id or sample_id in seen:
                raise ValueError(f'duplicate or invalid {split} sample ID')
            seen.add(sample_id)
            cameras = row.get('camera_paths', {})
            if set(cameras) != set(CAMERA_ORDER) or record_key(row) not in index:
                raise ValueError(f'missing camera or cache entry for {sample_id}')
            count += 1
        counts[split] = count
    array = np.memmap(directory / FEATURE_FILE, mode='r', dtype=np.float16, shape=tuple(shape))
    for start in tqdm(range(0,shape[0],32), desc='Validate cache features', unit='chunk', disable=not enabled):
        if not np.isfinite(array[start:start+32]).all():
            raise ValueError('cache features contain NaN or Inf')
    del array
    return _write(receipt, dict(version=1, cache_dir=str(directory), prepared_dir=str(prepared),
        expected_fingerprint=expected_fingerprint, fingerprint=manifest['fingerprint'],
        manifest_sha256=digest(directory/'manifest.json'), index_sha256=digest(directory/'index.json'),
        prepared_hashes=hashes, payload_stat=_stat(directory/FEATURE_FILE), shape=shape, counts=counts))


def open_validated_cache(receipt_path):
    receipt = json.loads(Path(receipt_path).read_text(encoding='utf-8'))
    directory, prepared = Path(receipt['cache_dir']), Path(receipt['prepared_dir'])
    manifest,index,shape,hashes = _inputs(directory,prepared,receipt['expected_fingerprint'])
    if (receipt['version'] != 1 or not directory.is_absolute() or not prepared.is_absolute() or
            receipt['manifest_sha256'] != digest(directory/'manifest.json') or
            receipt['index_sha256'] != digest(directory/'index.json') or receipt['shape'] != shape or
            receipt['fingerprint'] != manifest['fingerprint'] or receipt['prepared_hashes'] != hashes or
            receipt['payload_stat'] != _stat(directory/FEATURE_FILE) or
            set(receipt['counts']) != {'train','val','test'} or
            any(type(v) is not int or v < 0 for v in receipt['counts'].values())):
        raise ValueError('validated DINO cache receipt no longer matches files')
    return SimpleNamespace(manifest=manifest,index=index,counts=receipt['counts'],prepared=prepared,
        directory=directory,array=np.memmap(directory/FEATURE_FILE,mode='r',dtype=np.float16,shape=tuple(shape)))


def format_prompt(question):
    return f'Question: {question} Answer:'


def tokenizer_policy(tokenizer):
    policy = dict(tokenizer_class=type(tokenizer).__name__, vocab_sha256=hashlib.sha256(canonical(tokenizer.get_vocab())).hexdigest(),
                  special_tokens_map={k:str(v) if not isinstance(v,list) else list(map(str,v)) for k,v in tokenizer.special_tokens_map.items()},
                  pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,
                  padding_side=tokenizer.padding_side,add_special_tokens=True,truncation=False,
                  prompt_template='Question: {question} Answer:')
    backend = getattr(tokenizer,'backend_tokenizer',None)
    if backend is not None:
        state=json.loads(backend.to_str())
        state.pop('padding',None);state.pop('truncation',None)
        policy['backend_sha256']=hashlib.sha256(canonical(state)).hexdigest()
    return policy


def prepare_test(reader, tokenizer, output_dir, *, progress=True):
    output=[]
    seen=set()
    for row in tqdm(records(reader.prepared/'test.jsonl'),total=reader.counts['test'],desc='Prepare test prompts',unit='question',disable=not progress):
        sample_id=row['sample_id']
        if sample_id in seen:
            raise ValueError('duplicate test sample ID')
        seen.add(sample_id)
        if not isinstance(row.get('question'), str) or not row['question'].strip():
            raise ValueError(f'missing or invalid question for test sample {sample_id}')
        ids=tokenizer(format_prompt(row['question']),add_special_tokens=True,truncation=False)['input_ids']
        if not ids or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError('invalid tokenized test prompt')
        output.append(dict(sample_id=sample_id,eval_id=row.get('eval_id'),feature_index=reader.index[record_key(row)],input_ids=ids))
    if len(output) != reader.counts['test']:
        raise ValueError('test count mismatch')
    lengths = [len(row['input_ids']) for row in output]
    return _write(Path(output_dir)/'prepared_test.json',dict(version=1,records=output,camera_order=list(CAMERA_ORDER),
        token_lengths=dict(min=min(lengths,default=0),max=max(lengths,default=0)),
        cache_fingerprint=reader.manifest['fingerprint'],test_sha256=digest(reader.prepared/'test.jsonl'),tokenizer_policy=tokenizer_policy(tokenizer)))


def load_prepared_test(path):
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    if value.get('version') != 1 or not isinstance(value.get('records'),list):
        raise ValueError('invalid prepared test file')
    return value


def make_batch(reader, rows, tokenizer):
    if not rows or tokenizer.pad_token_id is None:
        raise ValueError('nonempty batch and tokenizer pad ID required')
    sequences=[row['input_ids'] for row in rows]
    if any(not seq for seq in sequences):
        raise ValueError('empty prompt token sequence')
    ids=torch.full((len(rows),max(map(len,sequences))),tokenizer.pad_token_id,dtype=torch.long)
    mask=torch.zeros_like(ids)
    features=torch.empty((len(rows),*reader.array.shape[1:]),dtype=torch.float32)
    for i,(row,seq) in enumerate(zip(rows,sequences)):
        index=row['feature_index']
        if type(index) is not int or not 0 <= index < len(reader.array):
            raise ValueError('feature index outside cache')
        ids[i,:len(seq)]=torch.tensor(seq,dtype=torch.long)
        mask[i,:len(seq)]=1
        np.copyto(features[i].numpy(),reader.array[index])
    return dict(input_ids=ids,attention_mask=mask,visual_features=features,sample_ids=[row['sample_id'] for row in rows])
