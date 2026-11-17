"""Validated cache receipts and pretokenized, answer-free test prompts."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from smolvlm_inference.cache import FeatureCache, FEATURE_FILE, CAMERA_ORDER, canonical, digest, records, record_key

IDENTITY_KEYS = ('backend', 'version', 'model_id', 'revision', 'processor_policy', 'geometry',
                 'storage_dtype', 'compute_dtype', 'camera_order')


def _json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.json')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def _stat(path):
    stat = path.stat()
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def _camera_order(order):
    from smolvlm_inference.model import validate_prompt_camera_order
    return validate_prompt_camera_order(order)


def validate_cache(cache_dir, prepared_dir, output_dir, expected_fingerprint=None):
    directory, prepared = Path(cache_dir).resolve(), Path(prepared_dir).resolve()
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    identity = {key: manifest[key] for key in IDENTITY_KEYS}
    if identity['storage_dtype'] != 'float16' or identity['camera_order'] != CAMERA_ORDER:
        raise ValueError('cache must use canonical FP16 storage')
    reader = FeatureCache(directory, prepared, identity)
    if expected_fingerprint is not None and reader.manifest['fingerprint'] != expected_fingerprint:
        raise ValueError('cache fingerprint differs from expected fingerprint')
    descriptor = dict(version=1, cache_dir=str(directory), prepared_dir=str(prepared),
                      fingerprint=manifest['fingerprint'], manifest_sha256=digest(directory / 'manifest.json'),
                      index_sha256=digest(directory / 'index.json'), prepared_hashes=manifest['prepared_hashes'],
                      payload_stat=_stat(directory / FEATURE_FILE), shape=list(reader.array.shape),
                      counts=reader.counts, identity=identity)
    return _json_write(Path(output_dir) / 'validated_cache.json', descriptor)


def open_validated_cache(descriptor_path):
    descriptor = json.loads(Path(descriptor_path).read_text(encoding='utf-8'))
    directory, prepared = Path(descriptor['cache_dir']), Path(descriptor['prepared_dir'])
    manifest_path, index_path = directory / 'manifest.json', directory / 'index.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    index = json.loads(index_path.read_text(encoding='utf-8'))
    identity = {key: manifest[key] for key in IDENTITY_KEYS}
    geometry = identity['geometry']
    shape = [len(index), geometry['num_views'], geometry['vision_tokens'], geometry['vision_dim']]
    payload = directory / FEATURE_FILE
    if (descriptor['version'] != 1 or not directory.is_absolute() or not prepared.is_absolute() or
            digest(manifest_path) != descriptor['manifest_sha256'] or digest(index_path) != descriptor['index_sha256'] or
            manifest['fingerprint'] != descriptor['fingerprint'] or
            hashlib.sha256(canonical({k: v for k, v in manifest.items() if k != 'fingerprint'})).hexdigest() != descriptor['fingerprint'] or
            identity != descriptor['identity'] or identity['storage_dtype'] != 'float16' or
            identity['camera_order'] != CAMERA_ORDER or descriptor['shape'] != shape or manifest['shape'] != shape or
            descriptor['counts'] != manifest['record_counts'] or manifest['num_samples'] != len(index) or
            descriptor['prepared_hashes'] != manifest['prepared_hashes'] or
            descriptor['prepared_hashes'] != {name: digest(prepared / name) for name in ('manifest.json', 'train.jsonl', 'val.jsonl', 'test.jsonl')} or
            _stat(payload) != descriptor['payload_stat'] or payload.stat().st_size != int(np.prod(shape)) * 2 or
            any(type(v) is not int for v in index.values()) or sorted(index.values()) != list(range(len(index)))):
        raise ValueError('validated cache receipt no longer matches input files')
    return SimpleNamespace(manifest=manifest, index=index, counts=descriptor['counts'], prepared=prepared,
                           directory=directory, array=np.memmap(payload, mode='r', dtype=np.float16, shape=tuple(shape)))


def prepare_test(reader, processor, prompt_camera_order, output_dir, max_length):
    from smolvlm_inference.text import CachedBatchCollator, chat_messages
    prompt_camera_order = _camera_order(prompt_camera_order)
    image_seq_len = reader.manifest['geometry']['visual_tokens_per_view']
    collator = CachedBatchCollator(processor, image_seq_len=image_seq_len, max_length=max_length,
                                   max_new_tokens=0, prompt_camera_order=prompt_camera_order)
    output, seen, eval_seen = [], set(), set()
    token = str(processor.image_token)
    image_id = processor.tokenizer.convert_tokens_to_ids(token)
    for row in records(reader.prepared / 'test.jsonl'):
        sample_id, eval_id = row['sample_id'], row['eval_id']
        if not isinstance(sample_id, str) or sample_id in seen or eval_id in eval_seen:
            raise ValueError('test sample IDs and eval IDs must be unique')
        seen.add(sample_id)
        eval_seen.add(eval_id)
        question = str(row['question'])
        if token in question:
            raise ValueError('question cannot contain image placeholders')
        messages = chat_messages(question, prompt_camera_order=prompt_camera_order)
        prompt = processor.apply_chat_template(messages,
                                               tokenize=False, add_generation_prompt=True)
        ids = processor.tokenizer(collator._expand(prompt), add_special_tokens=True,
                                  truncation=False)['input_ids']
        if len(ids) > max_length or ids.count(image_id) != 6 * image_seq_len:
            raise ValueError('test prompt exceeds context or has incorrect image token count')
        output.append(dict(sample_id=sample_id, eval_id=eval_id, feature_index=reader.index[record_key(row)], input_ids=ids))
    if len(output) != reader.counts['test']:
        raise ValueError('test count mismatch')
    value = dict(version=1, records=output, camera_order=list(prompt_camera_order), prompt_camera_order=list(prompt_camera_order),
                 manifest_fingerprint=reader.manifest['fingerprint'], test_sha256=digest(reader.prepared / 'test.jsonl'),
                 max_length=max_length, max_new_tokens=0, image_seq_len=image_seq_len,
                 tokenizer_class=type(processor.tokenizer).__name__, tokenizer_name=str(processor.tokenizer.name_or_path),
                 tokenizer_model_max_length=processor.tokenizer.model_max_length, image_token_id=image_id)
    return _json_write(Path(output_dir) / 'prepared_test.json', value)


def load_prepared_test(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if value.get('version') != 1 or not isinstance(value.get('records'), list):
        raise ValueError('invalid prepared test file')
    return value


def make_batch(reader, prepared_rows, processor, camera_order):
    order = _camera_order(camera_order)
    if not prepared_rows:
        raise ValueError('cannot build an empty batch')
    sequences = [row['input_ids'] for row in prepared_rows]
    if any(not ids for ids in sequences) or processor.tokenizer.pad_token_id is None:
        raise ValueError('nonempty sequences and tokenizer pad ID required')
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), processor.tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    features = torch.empty((len(sequences), *reader.array.shape[1:]), dtype=torch.float32)
    for i, (row, sequence) in enumerate(zip(prepared_rows, sequences)):
        index = row['feature_index']
        if type(index) is not int or not 0 <= index < len(reader.array):
            raise ValueError('feature index outside cache')
        ids[i, -len(sequence):] = torch.tensor(sequence, dtype=torch.long)
        mask[i, -len(sequence):] = 1
        for j, camera in enumerate(order):
            np.copyto(features[i, j].numpy(), reader.array[index, CAMERA_ORDER.index(camera)])
    return dict(input_ids=ids, attention_mask=mask, visual_features=features,
                sample_ids=[row['sample_id'] for row in prepared_rows])
