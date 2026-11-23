"""Streaming, answer-free native prompts in a compact relocatable mmap store."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from smolvlm_attention.sample_store import atomic_json, canonical, digest, fingerprint
from smolvlm_inference.cache import records, record_key, CAMERA_ORDER

FILES = ('input_ids.bin', 'offsets.npy', 'lengths.npy', 'row_offsets.npy', 'rows.jsonl')


def _tokenizer_identity(processor):
    tokenizer = processor.tokenizer
    return dict(tokenizer_class=type(tokenizer).__name__,
                tokenizer_backend_sha256=hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
                chat_template=str(getattr(processor, 'chat_template', None) or tokenizer.chat_template),
                image_token=str(processor.image_token), pad_token_id=tokenizer.pad_token_id)


class PromptStore:
    """Only row-local token IDs become Python lists; all index arrays remain mmap-backed."""
    def __init__(self, directory):
        self.directory = Path(directory)
        manifest = json.loads((self.directory / 'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('version') != 1 or manifest.get('complete') is not True:
            raise ValueError('incomplete or unsupported prompt store')
        if set(manifest.get('checksums', {})) != set(FILES) or any(
                digest(self.directory / name) != expected for name, expected in manifest['checksums'].items()):
            raise ValueError('prompt store checksum mismatch')
        self.identity = manifest['identity']
        if fingerprint(self.identity) != manifest.get('identity_sha256'):
            raise ValueError('prompt store identity checksum mismatch')
        self.offsets = np.load(self.directory / 'offsets.npy', mmap_mode='r', allow_pickle=False)
        self.lengths = np.load(self.directory / 'lengths.npy', mmap_mode='r', allow_pickle=False)
        self.row_offsets = np.load(self.directory / 'row_offsets.npy', mmap_mode='r', allow_pickle=False)
        count = manifest['count']
        if (count < 1 or self.offsets.shape != (count+1,) or self.lengths.shape != (count,) or
                self.row_offsets.shape != (count+1,) or self.offsets.dtype != np.dtype('<u8') or
                self.lengths.dtype != np.dtype('<u4') or self.row_offsets.dtype != np.dtype('<u8') or
                self.offsets[0] != 0 or self.row_offsets[0] != 0 or
                np.any(self.lengths == 0) or not np.array_equal(np.diff(self.offsets), self.lengths) or
                np.any(self.row_offsets[1:] <= self.row_offsets[:-1]) or
                int(self.offsets[-1])*4 != (self.directory / 'input_ids.bin').stat().st_size or
                int(self.row_offsets[-1]) != (self.directory / 'rows.jsonl').stat().st_size):
            raise ValueError('invalid prompt store geometry')
        self.ids = np.memmap(self.directory / 'input_ids.bin', mode='r', dtype='<u4')

    def __len__(self):
        return len(self.lengths)

    def portable_row(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self):
            raise IndexError(index)
        with (self.directory / 'rows.jsonl').open('rb') as stream:
            stream.seek(int(self.row_offsets[index]))
            row = json.loads(stream.read(int(self.row_offsets[index+1]-self.row_offsets[index])))
        if (row['row'] != index or row['split'] != self.identity['split'] or
                row['prompt_length'] != int(self.lengths[index]) or 'input_ids' in row or 'answer' in row):
            raise ValueError('invalid prompt row metadata')
        return row

    def row(self, index):
        row = self.portable_row(index)
        row['input_ids'] = self.ids[int(self.offsets[index]):int(self.offsets[index+1])].astype(np.int64).tolist()
        if (fingerprint(row['input_ids']) != row['prompt_ids_sha256'] or
                row['input_ids'][-1] != row['query_token_id'] or row['query_position'] != len(row['input_ids'])-1):
            raise ValueError('prompt IDs do not match mapping')
        return row


def prepare_prompt_store(reader, processor, split, directory, max_length=8192, *, progress=None):
    """Build once atomically; source content identities allow relocating operational paths."""
    from smolvlm_inference.text import CachedBatchCollator, chat_messages
    if split not in ('train', 'val'):
        raise ValueError('teacher prompt split must be train or val')
    if type(max_length) is not int or max_length <= 0:
        raise ValueError('invalid context limit')
    if list(reader.manifest['camera_order']) != list(CAMERA_ORDER):
        raise ValueError('teacher prompts require canonical camera order')
    directory = Path(directory)
    source = Path(reader.prepared) / f'{split}.jsonl'
    image_count = 6*reader.manifest['geometry']['visual_tokens_per_view']
    identity = dict(version=1, split=split, prepared_sha256=digest(source),
                    cache_fingerprint=reader.manifest['fingerprint'], cache_index_sha256=fingerprint(reader.index),
                    camera_order=list(CAMERA_ORDER), visual_tokens=image_count,
                    max_length=max_length, count=reader.counts[split], tokenizer=_tokenizer_identity(processor),
                    query_policy='last_valid_native_prompt_token')
    if directory.exists():
        existing = PromptStore(directory)
        if existing.identity != identity:
            raise ValueError('existing prompt store identity differs; use a separate directory')
        return existing
    count = reader.counts[split]
    if type(count) is not int or count < 1:
        raise ValueError('prompt split must be nonempty')
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=directory.name+'.partial-', dir=directory.parent))
    offsets = np.lib.format.open_memmap(temporary/'offsets.npy', mode='w+', dtype='<u8', shape=(count+1,))
    lengths = np.lib.format.open_memmap(temporary/'lengths.npy', mode='w+', dtype='<u4', shape=(count,))
    row_offsets = np.lib.format.open_memmap(temporary/'row_offsets.npy', mode='w+', dtype='<u8', shape=(count+1,))
    offsets[0] = row_offsets[0] = 0
    collator = CachedBatchCollator(processor, image_seq_len=image_count//6, max_length=max_length,
                                  max_new_tokens=0, prompt_camera_order=CAMERA_ORDER)
    image_id = processor.tokenizer.convert_tokens_to_ids(str(processor.image_token))
    seen = set()
    written = 0
    started = last_report = time.perf_counter()
    with (temporary/'input_ids.bin').open('wb') as ids_file, (temporary/'rows.jsonl').open('wb') as rows_file:
        for index, record in enumerate(records(source)):
            if index >= count:
                raise ValueError('prepared split count mismatch')
            sample_id = record['sample_id']
            if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
                raise ValueError('invalid or duplicate sample ID within split')
            seen.add(sample_id)
            question = str(record['question'])
            if str(processor.image_token) in question:
                raise ValueError('question contains image placeholder')
            prompt = processor.apply_chat_template(chat_messages(question, prompt_camera_order=CAMERA_ORDER),
                                                   tokenize=False, add_generation_prompt=True)
            ids = processor.tokenizer(collator._expand(prompt), add_special_tokens=True, truncation=False)['input_ids']
            if not ids or len(ids) > max_length or ids.count(image_id) != image_count:
                raise ValueError('prompt exceeds context or has incorrect image token count')
            if min(ids) < 0 or max(ids) > np.iinfo(np.uint32).max:
                raise ValueError('token ID outside storage range')
            key = record_key(record)
            if key not in reader.index:
                raise ValueError('sample is missing from feature cache')
            row = dict(row=index, prepared_row=index, split=split, sample_id=sample_id,
                       cache_key=key, feature_index=reader.index[key], prompt_length=len(ids),
                       query_token_id=ids[-1], query_position=len(ids)-1,
                       query_token_decoded=processor.tokenizer.decode([ids[-1]]),
                       question_sha256=hashlib.sha256(question.encode('utf-8')).hexdigest(),
                       prompt_ids_sha256=fingerprint(ids))
            np.asarray(ids, dtype='<u4').tofile(ids_file)
            rows_file.write(canonical(row)+b'\n')
            lengths[index] = len(ids)
            offsets[index+1] = offsets[index]+len(ids)
            row_offsets[index+1] = rows_file.tell()
            written += 1
            now = time.perf_counter()
            if progress is not None and (now-last_report >= 10 or written == count):
                elapsed = now-started
                speed = written/elapsed if elapsed > 0 else 0.0
                progress(dict(phase='prepare_prompts', split=split, processed=written, total=count,
                              elapsed_seconds=elapsed, samples_per_second=speed,
                              eta_seconds=(count-written)/speed if speed > 0 else None))
                last_report = now
        if written != count:
            raise ValueError('prepared split count mismatch')
        for stream in (ids_file, rows_file):
            stream.flush(); os.fsync(stream.fileno())
    if digest(source) != identity['prepared_sha256']:
        raise ValueError('prepared split changed while building prompt store')
    for array in (offsets, lengths, row_offsets):
        array.flush()
        array._mmap.close()
    for name in FILES:
        with (temporary/name).open('r+b') as stream:
            os.fsync(stream.fileno())
    atomic_json(temporary/'manifest.json', dict(version=1, complete=True, count=count, identity=identity,
                identity_sha256=fingerprint(identity), checksums={name:digest(temporary/name) for name in FILES}))
    os.rename(temporary, directory)
    return PromptStore(directory)
