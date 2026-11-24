"""Validated, read-only MAP training inputs and CPU collation."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from smolvlm_attention.sample_store import atomic_json, digest, fingerprint
from smolvlm_attention.teacher_targets.loader import file_stat
from smolvlm_attention.teacher_targets.runtime import validate_teacher_cache
from smolvlm_attention.teacher_targets.storage import validate_export, validate_log_probs
from smolvlm_inference.cache import record_key, CAMERA_ORDER, canonical as cache_canonical
from smolvlm_inference.pruning_text import PruningBatchCollator


def read_receipt(path, *, include_targets=True):
    receipt = json.loads(Path(path).read_text(encoding='utf-8'))
    if receipt.get('version') == 2:
        from smolvlm_predictor_io.compact import read_store
        return read_store(path, include_targets=include_targets)
    if receipt.get('version') != 1:
        raise ValueError('unsupported receipt')
    try:
        changed = any(file_stat(p) != stat for p, stat in receipt['files'].items()
                      if include_targets or p not in receipt.get('target_files', []))
    except OSError as exc:
        raise ValueError('validated input changed') from exc
    if changed:
        raise ValueError('validated input changed')
    return receipt


def prepare_inputs(targets_root, cache_dir, prepared_dir, model, output_dir, *, expected_counts=None,
                   expected_cache_fingerprint=None, checkpoint_sha256=None, projected_cache=None):
    targets_root, prepared_dir, output = map(lambda p: Path(p).resolve(),
        (targets_root, prepared_dir, output_dir))
    output.mkdir(parents=True, exist_ok=True)
    signature = model.architecture_signature
    projected = None
    if projected_cache is not None:
        from smolvlm_predictor_io.projected import ProjectedCache
        projected = projected_cache if isinstance(projected_cache, ProjectedCache) else ProjectedCache(projected_cache)
        origin = projected.manifest['identity']
        descriptor, manifest = origin['raw_cache_descriptor'], origin['raw_cache_manifest']
        if origin.get('checkpoint_sha256') != checkpoint_sha256 or origin.get('model') != signature:
            raise ValueError('Projected checkpoint/connector identity mismatch')
        if expected_cache_fingerprint and manifest['fingerprint'] != expected_cache_fingerprint:
            raise ValueError('Projected source cache fingerprint mismatch')
        for name, sha in descriptor['prepared_hashes'].items():
            if digest(prepared_dir/name) != sha:
                raise ValueError('Projected prepared data identity mismatch')
        if descriptor['prepared_hashes'] != manifest['prepared_hashes']:
            raise ValueError('Projected source prepared hashes mismatch')
        expected_shape = [len(projected.index), 6, model.geometry['visual_tokens_per_view'], model.geometry['text_dim']]
        if projected.manifest['shape'] != expected_shape or manifest['fingerprint'] != hashlib.sha256(cache_canonical({k:v for k,v in manifest.items() if k != 'fingerprint'})).hexdigest():
            raise ValueError('Projected geometry/source manifest mismatch')
        index = projected.index
        cache_dir = None
    else:
        cache_dir = Path(cache_dir).resolve()
        descriptor = json.loads(validate_teacher_cache(cache_dir, prepared_dir, output,
                                  expected_cache_fingerprint).read_text(encoding='utf-8'))
        manifest = json.loads((cache_dir/'manifest.json').read_text(encoding='utf-8'))
        index = json.loads((cache_dir/'index.json').read_text(encoding='utf-8'))
    counts = expected_counts or dict(train=341381, val=19785, test=16817)
    if descriptor['counts'] != counts:
        raise ValueError('prepared counts differ')
    if (model.training or any(p.requires_grad for p in model.parameters()) or
            manifest['geometry'] != model.geometry or manifest['model_id'] != model.model_id or
            manifest['revision'] != model.model_revision or manifest['processor_policy'] != signature['processor'] or
            list(model.prompt_camera_order) != CAMERA_ORDER):
        raise ValueError('model/cache identity mismatch')
    visual_tokens = 6 * model.geometry['visual_tokens_per_view']
    encoder = PruningBatchCollator(model.processor, image_seq_len=visual_tokens//6,
        max_length=model.native.config.text_config.max_position_embeddings, max_new_tokens=0)
    files = [prepared_dir/'manifest.json']
    if projected is None:
        files += [cache_dir/'manifest.json', cache_dir/'index.json', cache_dir/'features.float16.memmap']
    else:
        files += [projected.directory/name for name in ('manifest.json','index.json','projected.npy')]
    target_hashes, split_paths, all_target_files = {}, {}, []
    for split in ('train', 'val', 'test'):
        source = prepared_dir/f'{split}.jsonl'
        files.append(source)
        target_dir = targets_root/split
        has_target = split != 'test'
        if has_target:
            target = validate_export(target_dir)
            identity, protocol = target['identity'], target['identity'].get('protocol', {})
            expected = dict(checkpoint_sha256=checkpoint_sha256, native_model=model.model_id,
                            native_revision=model.model_revision, cache_fingerprint=manifest['fingerprint'],
                            cache_manifest_sha256=descriptor['manifest_sha256'], cache_index_sha256=descriptor['index_sha256'],
                            cache_payload_sha256=descriptor['payload_sha256'], prepared_hashes=descriptor['prepared_hashes'])
            if (any(identity.get(k) != v for k,v in expected.items()) or identity.get('split') != split or
                    protocol.get('score_layer_index') != 22 or protocol.get('score_layer_number') != 23 or
                    protocol.get('query_policy') != 'last_valid_native_generation_prompt_token' or
                    protocol.get('normalization') != 'mean_heads_all_valid_keys_then_visual_normalization_log_v1' or
                    target['shape'] != [counts[split], visual_tokens] or
                    identity.get('model') != dict(model_id=model.model_id, revision=model.model_revision,
                        geometry=model.geometry, processor_policy=signature['processor'], camera_order=CAMERA_ORDER)):
                raise ValueError('teacher target identity mismatch')
            if counts == dict(train=341381, val=19785, test=16817):
                from smolvlm_attention.teacher_targets.runtime import load_config
                pinned = load_config()
                for key in ('checkpoint_repo', 'checkpoint_revision', 'checkpoint_filename', 'camera_order'):
                    if protocol.get(key) != pinned[key]: raise ValueError('teacher pinned identity mismatch')
            target_files = [target_dir/name for name in ('manifest.json', 'samples.jsonl', 'log_probs.npy')]
            files.extend(target_files)
            all_target_files.extend(map(str, target_files))
            target_hashes[split] = {p.name: digest(p) for p in target_files}
        lookup = {}
        if has_target:
            with (target_dir/'samples.jsonl').open('rb') as stream:
                for target_row in range(counts[split]):
                    start = stream.tell(); meta = json.loads(stream.readline()); end = stream.tell()
                    sid = meta['sample_id']
                    if sid in lookup or meta['row'] != target_row: raise ValueError('duplicate target sample')
                    lookup[sid] = (target_row, start, end, meta)
        offsets = np.lib.format.open_memmap(output/f'{split}_offsets.npy', mode='w+', dtype='<u8', shape=(counts[split]+1,))
        target_offsets = np.lib.format.open_memmap(output/f'{split}_target_offsets.npy', mode='w+', dtype='<u8', shape=(counts[split], 2))
        target_rows = np.lib.format.open_memmap(output/f'{split}_target_rows.npy', mode='w+', dtype='<u8', shape=(counts[split],))
        lengths = np.lib.format.open_memmap(output/f'{split}_lengths.npy', mode='w+', dtype='<u4', shape=(counts[split],))
        feature_indices = np.lib.format.open_memmap(output/f'{split}_features.npy', mode='w+', dtype='<u8', shape=(counts[split],))
        offsets[0] = 0
        seen = set()
        with source.open('rb') as src:
            target_stream = None
            try:
                for i in range(counts[split]):
                    line = src.readline()
                    if not line: raise ValueError('prepared row count mismatch')
                    row = json.loads(line); sid = row['sample_id']; key = record_key(row)
                    if sid in seen or key not in index: raise ValueError('duplicate or missing sample')
                    seen.add(sid); feature_indices[i] = index[key]; offsets[i+1] = src.tell()
                    ids = encoder._encode_question(row)[0]['input_ids']; lengths[i] = len(ids)
                    if has_target:
                        if sid not in lookup: raise ValueError('missing target sample')
                        target_row, start, end, meta = lookup.pop(sid)
                        target_rows[i] = target_row; target_offsets[i] = (start, end)
                        if (meta['row'] != target_row or meta['prepared_row'] != i or meta['sample_id'] != sid or
                            meta['split'] != split or meta['cache_key'] != key or meta['feature_index'] != index[key] or
                            meta['question_sha256'] != hashlib.sha256(str(row['question']).encode()).hexdigest() or
                            meta['prompt_ids_sha256'] != fingerprint(ids) or meta['prompt_length'] != len(ids) or
                            meta['query_position'] != len(ids)-1 or meta['query_token_id'] != ids[-1]):
                            raise ValueError('teacher sample alignment mismatch')
                    else: target_offsets[i] = (0, 0); target_rows[i] = 0
                if src.readline() or lookup: raise ValueError('row count mismatch')
            finally:
                if target_stream: target_stream.close()
        for array in (offsets, target_offsets, feature_indices, lengths, target_rows): array.flush(); array._mmap.close()
        paths = {name: str((output/f'{split}_{name}.npy').resolve()) for name in ('offsets','target_offsets','features','lengths','target_rows')}
        files.extend(map(Path, paths.values()))
        split_paths[split] = dict(source=str(source), target=str(target_dir) if has_target else None, **paths)
    from smolvlm_attention.config import MODELS
    identity = dict(target_hashes=target_hashes, cache_payload_sha256=descriptor['payload_sha256'],
                    cache_manifest_sha256=descriptor['manifest_sha256'], cache_index_sha256=descriptor['index_sha256'],
                    cache_fingerprint=manifest['fingerprint'], prepared_hashes=descriptor['prepared_hashes'],
                    checkpoint_sha256=checkpoint_sha256, model=signature,
                    base_checkpoint=dict(repo=MODELS['base']['repo'], revision=MODELS['base']['revision'], filename='finetune_best.pt'),
                    teacher=dict(layer_index=22, layer_number=23,
                        query_policy='last_valid_native_generation_prompt_token',
                        normalization='mean_heads_all_valid_keys_then_visual_normalization_log_v1'),
                    predictor_query_policy='question_tokens_visual_softmax_then_masked_query_mean_v1')
    receipt = dict(version=1, identity=identity, geometry=model.geometry, counts=counts,
                   cache_dir=str(cache_dir), shape=descriptor['shape'], splits=split_paths,
                   max_length=model.native.config.text_config.max_position_embeddings,
                   target_files=all_target_files, files={str(p): file_stat(p) for p in files})
    if projected is not None:
        receipt['projected_cache_dir'] = str(projected.directory.resolve())
        receipt['identity']['projected_cache'] = dict(format=projected.manifest['format'],
            fingerprint=projected.manifest['fingerprint'], payload_sha256=projected.manifest['payload_sha256'])
        receipt['identity']['pipeline'] = 'projected_compact_single_pool_v1'
    path = output/'receipt.json'; atomic_json(path, receipt); return path


class PredictorDataset(Dataset):
    def __init__(self, receipt, split):
        self.receipt = str(receipt); self.split = split
        self.info = read_receipt(receipt, include_targets=split != 'test')
        if split not in self.info['splits']: raise ValueError('invalid split')
        self._arrays = None
    @property
    def prompt_lengths(self):
        return np.load(self.info['splits'][self.split]['lengths'], mmap_mode='r', allow_pickle=False)
    def __len__(self): return self.info['counts'][self.split]
    def __getstate__(self): return dict(receipt=self.receipt, split=self.split, info=self.info, _arrays=None)
    def __getitem__(self, item):
        if not isinstance(item, (int, np.integer)) or not 0 <= item < len(self): raise IndexError(item)
        if self._arrays is None:
            read_receipt(self.receipt, include_targets=self.split != 'test'); p = self.info['splits'][self.split]
            self._arrays = {k: np.load(p[k], mmap_mode='r', allow_pickle=False) for k in ('offsets','target_offsets','features','target_rows')}
            if p['target']: self._arrays['target'] = np.load(Path(p['target'])/'log_probs.npy', mmap_mode='r', allow_pickle=False)
        p, a = self.info['splits'][self.split], self._arrays
        def read(path, offsets):
            with Path(path).open('rb') as stream:
                stream.seek(int(offsets[item])); return json.loads(stream.read(int(offsets[item+1]-offsets[item])))
        row = read(p['source'], a['offsets'])
        meta = None
        if p['target']:
            with (Path(p['target'])/'samples.jsonl').open('rb') as stream:
                start, end = map(int, a['target_offsets'][item]); stream.seek(start); meta = json.loads(stream.read(end-start))
        return dict(row=row, metadata=meta, feature_index=int(a['features'][item]), index=int(item), split=self.split,
                    log_target=np.array(a['target'][a['target_rows'][item]], copy=True) if p['target'] else None)


class PredictorCollator:
    def __init__(self, receipt, processor, *, max_new_tokens=0):
        self.receipt = str(receipt); self.info = read_receipt(receipt, include_targets=False); self.processor = processor
        self.encoder = PruningBatchCollator(processor, image_seq_len=self.info['geometry']['visual_tokens_per_view'],
            max_length=self.info['max_length'], max_new_tokens=max_new_tokens)
        self._features = None
    def __getstate__(self): return {**self.__dict__, '_features': None}
    def __call__(self, items):
        if not items or len({x['split'] for x in items}) != 1: raise ValueError('empty or mixed batch')
        if self._features is None:
            read_receipt(self.receipt, include_targets=items[0]['split'] != 'test')
            self._features = np.memmap(Path(self.info['cache_dir'])/'features.float16.memmap', mode='r', dtype='<f2', shape=tuple(self.info['shape']))
        encoded = [self.encoder._encode_question(x['row']) for x in items]
        width = max(len(x[0]['input_ids']) for x in encoded); n = len(items)
        ids = torch.full((n,width), self.processor.tokenizer.pad_token_id, dtype=torch.long)
        attention = torch.zeros_like(ids); question = torch.zeros_like(ids, dtype=torch.bool)
        features = torch.empty((n,*self._features.shape[1:]), dtype=torch.float16)
        image_id = self.processor.tokenizer.convert_tokens_to_ids(str(self.processor.image_token))
        for i,(item,(enc,overlap)) in enumerate(zip(items,encoded)):
            tokens = enc['input_ids']; start = width-len(tokens)
            if item['metadata'] and fingerprint(tokens) != item['metadata']['prompt_ids_sha256']: raise ValueError('prompt hash mismatch')
            ids[i,start:] = torch.tensor(tokens); attention[i,start:] = 1; question[i,start:] = torch.tensor(overlap)
            np.copyto(features[i].numpy(), self._features[item['feature_index']])
        if not torch.isfinite(features).all(): raise ValueError('nonfinite feature batch')
        image_positions = torch.stack([(ids[i] == image_id).nonzero().flatten() for i in range(n)])
        result = dict(input_ids=ids, attention_mask=attention, visual_features=features, question_mask=question,
            image_positions=image_positions, source_position_ids=(attention.cumsum(1)-1).clamp_min(0),
            sample_ids=[x['row']['sample_id'] for x in items], rows=torch.tensor([x['index'] for x in items]))
        if items[0]['log_target'] is not None:
            targets = np.stack([x['log_target'] for x in items])
            validate_log_probs(targets, self.info['geometry']['visual_tokens_per_view'] * 6)
            result['log_target'] = torch.from_numpy(targets).float()
        if items[0]['split'] == 'test':
            result['answers'] = [x['row'].get('answer','') for x in items]
            result['eval_ids'] = [x['row'].get('eval_id', x['row']['sample_id']) for x in items]
        return result
