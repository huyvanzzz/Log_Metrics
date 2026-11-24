"""Compact prompt stores: tokenize once, retain no tokenizer in loader workers."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from smolvlm_attention.sample_store import atomic_json, fingerprint, digest
from smolvlm_attention.teacher_targets.loader import file_stat
from smolvlm_attention.teacher_targets.storage import validate_log_probs


def prepare_store(receipt, processor, directory, projected_manifest=None, *, projected_path=None):
    """Consume an already validated receipt; verify prompt hashes before publication."""
    from smolvlm_inference.pruning_text import PruningBatchCollator
    info = json.loads(Path(receipt).read_text(encoding='utf-8')) if not isinstance(receipt, dict) else receipt
    if projected_path is not None: projected_manifest=Path(projected_path)/'manifest.json'
    root = Path(directory).resolve(); root.mkdir(parents=True, exist_ok=True)
    path = root/'store.json'
    source_identity = fingerprint(info['identity'])
    if path.exists():
        existing = read_store(path)
        if existing['source_identity'] != source_identity or existing.get('projected_manifest') != (str(projected_manifest) if projected_manifest else None): raise ValueError('compact store identity mismatch')
        return path
    encoder = PruningBatchCollator(processor, image_seq_len=info['geometry']['visual_tokens_per_view'],
        max_length=info['max_length'], max_new_tokens=0)
    image_id = processor.tokenizer.convert_tokens_to_ids(str(processor.image_token))
    splits = {}; files = {}
    for split, p in info['splits'].items():
        count = info['counts'][split]; prefix = root/split
        # Streams bound construction memory independently of the split token count.
        token_file = prefix.with_suffix('.tokens.bin'); question_file = prefix.with_suffix('.questions.bin')
        offsets = np.lib.format.open_memmap(root/f'{split}_tokens_offsets.npy', mode='w+', dtype='<i8', shape=(count+1,))
        qoffsets = np.lib.format.open_memmap(root/f'{split}_question_offsets.npy', mode='w+', dtype='<i8', shape=(count+1,))
        images = np.lib.format.open_memmap(root/f'{split}_images.npy', mode='w+', dtype='<i8', shape=(count,6*info['geometry']['visual_tokens_per_view']))
        features = np.load(p['features'], mmap_mode='r'); targets = np.load(p['target_rows'], mmap_mode='r')
        target_offsets = np.load(p['target_offsets'], mmap_mode='r') if p.get('target') else None
        metadata_file = root/f'{split}_metadata.jsonl'; metadata_offsets = [0]
        offsets[0] = qoffsets[0] = 0
        target_stream = (Path(p['target'])/'samples.jsonl').open('rb') if p.get('target') else None
        try:
            with Path(p['source']).open('rb') as source, token_file.open('wb') as tok, question_file.open('wb') as ques, metadata_file.open('wb') as meta:
                for row in range(count):
                    raw = source.readline()
                    if not raw: raise ValueError('prepared row count mismatch')
                    item = json.loads(raw); enc, overlap = encoder._encode_question(item)
                    ids = np.asarray(enc['input_ids'], dtype='<i8'); qc = np.flatnonzero(overlap).astype('<i8')
                    cols = np.flatnonzero(ids == image_id)
                    if len(cols) != images.shape[1] or not len(qc): raise ValueError('invalid compact prompt mapping')
                    if target_stream:
                        start,end = map(int,target_offsets[row]); target_stream.seek(start); teacher = json.loads(target_stream.read(end-start))
                        if teacher['sample_id'] != item['sample_id'] or teacher['prompt_ids_sha256'] != fingerprint(ids.tolist()):
                            raise ValueError('compact teacher prompt mismatch')
                    tok.write(ids.tobytes()); ques.write(qc.tobytes()); offsets[row+1] = offsets[row]+len(ids); qoffsets[row+1] = qoffsets[row]+len(qc); images[row] = cols
                    compact = dict(sample_id=item['sample_id'])
                    if split == 'test': compact.update(answer=item.get('answer',''), eval_id=item.get('eval_id',item['sample_id']), question=item['question'])
                    meta.write((json.dumps(compact,ensure_ascii=False)+'\n').encode()); metadata_offsets.append(meta.tell())
                if source.readline(): raise ValueError('prepared row count mismatch')
        finally:
            if target_stream: target_stream.close()
        for a in (offsets,qoffsets,images): a.flush()
        np.save(root/f'{split}_features.npy', np.asarray(features)); np.save(root/f'{split}_target_rows.npy',np.asarray(targets)); np.save(root/f'{split}_metadata_offsets.npy',np.asarray(metadata_offsets,dtype='<i8'))
        names = dict(tokens=str(token_file),questions=str(question_file),metadata=str(metadata_file),
            **{k:str(root/f'{split}_{suffix}.npy') for k,suffix in [('token_offsets','tokens_offsets'),('question_offsets','question_offsets'),('images','images'),('features','features'),('target_rows','target_rows'),('metadata_offsets','metadata_offsets')]})
        if p.get('target'): names['target'] = str(Path(p['target'])/'log_probs.npy')
        splits[split] = names
        for name in names.values(): files[name] = file_stat(name)
    result = dict(version=2, source_identity=source_identity, identity=info['identity'], counts=info['counts'],
        geometry=info['geometry'], max_length=info['max_length'], pad_token_id=processor.tokenizer.pad_token_id,
        splits=splits,files=files,projected_manifest=str(projected_manifest) if projected_manifest else None)
    if projected_manifest:
        manifest_path=Path(projected_manifest)
        projected_info=json.loads(manifest_path.read_text(encoding='utf-8'))
        payload=Path(projected_info.get('payload',projected_info.get('features_path','projected.npy')))
        if not payload.is_absolute(): payload=manifest_path.parent/payload
        for value in (manifest_path,payload): files[str(value)]=file_stat(value)
    atomic_json(path,result); return path


def read_store(path, *, include_targets=True):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if value.get('version') != 2: raise ValueError('unsupported compact store')
    target_paths={p['target'] for p in value['splits'].values() if p.get('target')}
    for p,stat in value['files'].items():
        if not include_targets and p in target_paths: continue
        if file_stat(p) != stat: raise ValueError('compact store changed')
    return value


class CompactDataset(Dataset):
    def __init__(self, store, split=None):
        self.path = str(store); full=read_store(store,include_targets=split!='test')
        self.info = {k:v for k,v in full.items() if k not in ('identity','files')}
        excluded={p['target'] for p in full['splits'].values() if p.get('target')} if split=='test' else set()
        self._stats={p:s for p,s in full['files'].items() if p not in excluded}
        self.split = split; self._arrays = {}; self._streams = {}; self._checked=False
    def __len__(self): return self.info['counts'][self.split] if self.split else sum(self.info['counts'].values())
    @property
    def prompt_lengths(self):
        p=self.info['splits'][self.split]; return np.diff(np.load(p['token_offsets'],mmap_mode='r'))
    def __getstate__(self): return {**self.__dict__,'_arrays':{},'_streams':{},'_checked':False}
    def __getitem__(self, request):
        split,row = (self.split,request) if isinstance(request,(int,np.integer)) else request
        row=int(row)
        if split not in self.info['counts'] or not 0 <= row < self.info['counts'][split]: raise IndexError(request)
        if not self._checked:
            for path,stat in self._stats.items():
                if split=='test' and path in {p.get('target') for p in self.info['splits'].values()}: continue
                if file_stat(path)!=stat: raise ValueError('compact worker input changed')
            self._checked=True
        if split not in self._arrays:
            p=self.info['splits'][split]
            a={k:np.load(p[k],mmap_mode='r',allow_pickle=False) for k in ('token_offsets','question_offsets','images','features','target_rows','metadata_offsets')}
            a['tokens']=np.memmap(p['tokens'],mode='r',dtype='<i8'); a['questions']=np.memmap(p['questions'],mode='r',dtype='<i8')
            if p.get('target'): a['target']=np.load(p['target'],mmap_mode='r',allow_pickle=False)
            self._arrays[split]=a; self._streams[split]=Path(p['metadata']).open('rb')
        a=self._arrays[split]; stream=self._streams[split]; start,end=map(int,a['metadata_offsets'][row:row+2]); stream.seek(start); metadata=json.loads(stream.read(end-start))
        result=dict(split=split,index=row,row=metadata,feature_index=int(a['features'][row]),
            tokens=np.array(a['tokens'][int(a['token_offsets'][row]):int(a['token_offsets'][row+1])],copy=True),
            question_columns=np.array(a['questions'][int(a['question_offsets'][row]):int(a['question_offsets'][row+1])],copy=True),
            image_columns=np.array(a['images'][row],copy=True))
        result['log_target']=np.array(a['target'][int(a['target_rows'][row])],copy=True) if 'target' in a else None
        return result


class CompactCollator:
    def __init__(self,store,*,max_new_tokens=0):
        self.info={k:v for k,v in read_store(store,include_targets=False).items() if k not in ('identity','files')}; self.max_new_tokens=max_new_tokens; self._features=None; self._scratch=None
    def __getstate__(self): return {**self.__dict__,'_features':None,'_scratch':None}
    def __call__(self,items):
        started = time.perf_counter()
        feature_seconds = 0.
        if not items or len({x['split'] for x in items})!=1: raise ValueError('empty or mixed batch')
        if self._features is None:
            manifest=json.loads(Path(self.info['projected_manifest']).read_text(encoding='utf-8')); payload=manifest.get('payload',manifest.get('features_path','projected.npy'))
            path=Path(payload); path=path if path.is_absolute() else Path(self.info['projected_manifest']).parent/path
            self._features=np.load(path,mmap_mode='r',allow_pickle=False) if path.suffix=='.npy' else np.memmap(path,mode='r',dtype='<f4',shape=tuple(manifest['shape']))
            self._scratch=np.empty(self._features.shape[1:],dtype=bool)
        n=len(items); width=max(len(x['tokens']) for x in items); qwidth=max(len(x['question_columns']) for x in items)
        if width+self.max_new_tokens>self.info['max_length']: raise ValueError('expanded chat exceeds context budget')
        ids=torch.full((n,width),self.info['pad_token_id'],dtype=torch.long); mask=torch.zeros_like(ids); qm=torch.zeros_like(ids,dtype=torch.bool)
        features=torch.empty((n,*self._features.shape[1:]),dtype=torch.float32); images=torch.empty((n,len(items[0]['image_columns'])),dtype=torch.long)
        questions=torch.zeros((n,qwidth),dtype=torch.long); qvalid=torch.zeros((n,qwidth),dtype=torch.bool)
        for i,item in enumerate(items):
            start=width-len(item['tokens']); ids[i,start:]=torch.from_numpy(item['tokens']); mask[i,start:]=1
            cols=torch.from_numpy(item['question_columns'])+start; qm[i,cols]=True; questions[i,:len(cols)]=cols; qvalid[i,:len(cols)]=True
            images[i]=torch.from_numpy(item['image_columns'])+start
            feature_started = time.perf_counter()
            source=self._features[item['feature_index']]; np.isfinite(source,out=self._scratch)
            if not self._scratch.all(): raise ValueError('nonfinite projected features')
            np.copyto(features[i].numpy(),source)
            feature_seconds += time.perf_counter() - feature_started
        result=dict(input_ids=ids,attention_mask=mask,projected_visual_features=features.flatten(1,2),question_mask=qm,image_positions=images,
            source_position_ids=(mask.cumsum(1)-1).clamp_min(0),question_columns=questions,question_valid=qvalid,
            query_columns=torch.full((n,),width-1,dtype=torch.long),sample_ids=[x['row']['sample_id'] for x in items],rows=torch.tensor([x['index'] for x in items]))
        result['last_query_columns']=result['query_columns']; result['position_ids']=result['source_position_ids']
        if items[0]['log_target'] is not None:
            target=np.stack([x['log_target'] for x in items]); validate_log_probs(target,images.shape[1]); result['log_target']=torch.from_numpy(target)
        if items[0]['split']=='test':
            result.update(answers=[x['row'].get('answer','') for x in items],eval_ids=[x['row'].get('eval_id',x['row']['sample_id']) for x in items])
        result['cpu_collate_seconds'] = time.perf_counter() - started
        result['cpu_feature_validation_copy_seconds'] = feature_seconds
        return result
