"""Read existing native projected caches; never build or open raw features."""
import json
from pathlib import Path
import numpy as np
import torch
from smolvlm_inference.model import CachedSmolVLM, CAMERA_ORDER
from smolvlm_attention.config import CACHE_FINGERPRINT
from smolvlm_attention.sample_store import atomic_json, digest, fingerprint
from smolvlm_predictor_io.projected import ProjectedCache, FORMAT, GROUPED_PROJECTION_POLICY


def discover_projected_cache(explicit='', *, search_roots=None):
    def complete(path):
        try:
            value=json.loads((path/'manifest.json').read_text(encoding='utf-8'))
            return value.get('format')==FORMAT and value.get('complete') is True and all((path/n).is_file() for n in ('projected.npy','index.json'))
        except (OSError,ValueError): return False
    if explicit:
        path=Path(explicit).resolve()
        if not complete(path): raise ValueError(f'Not a complete projected cache: {path}')
        return path
    roots=search_roots if search_roots is not None else ['/kaggle/input','/kaggle/working/smolvlm_projected_visual_cache']
    found=set()
    for root in map(Path,roots):
        if not root.is_dir(): continue
        candidates=[root,*[p.parent for p in root.rglob('manifest.json')]]
        found.update(p.resolve() for p in candidates if complete(p))
    if not found: raise FileNotFoundError('No complete projected cache found; set PROJECTED_CACHE_INPUT explicitly. This notebook never builds projected cache.')
    if len(found)!=1: raise ValueError(f'Found multiple complete projected caches; set PROJECTED_CACHE_INPUT explicitly: {sorted(map(str,found))}')
    return next(iter(found))


def _stats(path):
    stat=Path(path).stat()
    return dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)


def validate_projected_cache(directory, model, checkpoint, prepared_dir, receipt_path):
    """Parent-only full payload verification, then a content/stat bound receipt."""
    from tqdm.auto import tqdm
    directory=Path(directory).resolve()
    with tqdm(total=4,desc='Projected cache validation',unit='stage') as bar:
        cache=ProjectedCache(directory,full=False); bar.update(1)
        origin=cache.manifest['identity']; raw=origin.get('raw_cache_manifest',{})
        if origin.get('projection_policy')!=GROUPED_PROJECTION_POLICY:
            raise ValueError('Projected projection policy mismatch')
        if origin.get('checkpoint_sha256')!=digest(checkpoint) or origin.get('model')!=model.architecture_signature:
            raise ValueError('Projected checkpoint/architecture mismatch')
        if (raw.get('fingerprint')!=CACHE_FINGERPRINT or raw.get('geometry')!=model.geometry
                or model.prompt_camera_order!=CAMERA_ORDER):
            raise ValueError('Projected raw-cache fingerprint/geometry/camera order mismatch')
        if model.visual_token_len!=384 or len(model.native.model.text_model.layers)!=30:
            raise ValueError('Expected base model with 384 visual tokens and 30 decoder layers')
        if tuple(cache.array.shape[1:])!=(6,64,576): raise ValueError('Projected production geometry must be [6,64,576]')
        # Origin signatures contain the canonical processor camera policy. Reject any
        # explicit raw order as well; old manifests may encode it only in signature.
        for key in ('camera_order','camera_names'):
            if key in raw and raw[key]!=CAMERA_ORDER: raise ValueError('Projected camera order mismatch')
        hashes=origin.get('raw_cache_descriptor',{}).get('prepared_hashes',{})
        if not hashes or 'test.jsonl' not in hashes: raise ValueError('Projected prepared hashes missing test.jsonl')
        for name,sha in hashes.items():
            if digest(Path(prepared_dir)/name)!=sha: raise ValueError(f'Projected prepared identity mismatch: {name}')
        bar.update(1)
        if digest(directory/'projected.npy')!=cache.manifest['payload_sha256']: raise ValueError('Projected payload checksum mismatch')
        bar.update(1)
        for start in tqdm(range(0,len(cache.array),32),desc='Scan projected features',leave=False):
            if not np.isfinite(cache.array[start:start+32]).all(): raise ValueError('Nonfinite projected features')
        value=dict(format='combined_topk_projected_receipt_v1',directory=str(directory),
            manifest_sha256=digest(directory/'manifest.json'),fingerprint=cache.manifest['fingerprint'],
            checkpoint_sha256=origin['checkpoint_sha256'],prepared_hashes=hashes,
            files={n:_stats(directory/n) for n in ('manifest.json','index.json','projected.npy')})
        value['receipt_fingerprint']=fingerprint(value)
        atomic_json(receipt_path,value); bar.update(1)
    return Path(receipt_path)


def open_projected_receipt(path):
    """Workers reopen read-only; parent already hashed the full payload."""
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    if value.get('format')!='combined_topk_projected_receipt_v1' or value.get('receipt_fingerprint')!=fingerprint({k:v for k,v in value.items() if k!='receipt_fingerprint'}):
        raise ValueError('Invalid projected validation receipt')
    directory=Path(value['directory'])
    if any(_stats(directory/n)!=stat for n,stat in value['files'].items()): raise ValueError('Projected files changed after validation')
    if digest(directory/'manifest.json')!=value['manifest_sha256']: raise ValueError('Projected manifest changed after validation')
    cache=ProjectedCache(directory,full=False)
    if cache.manifest['fingerprint']!=value['fingerprint']: raise ValueError('Projected receipt fingerprint mismatch')
    return cache


class ProjectedSmolVLM(CachedSmolVLM):
    """Explicit input adapter; native model/weights are shared, connector is bypassed."""
    @classmethod
    def from_model(cls, model):
        adapter=cls(model.native,model.processor,model_id=model.model_id,model_revision=model.model_revision,
                    prompt_camera_order=model.prompt_camera_order)
        adapter.checkpoint_metadata=getattr(model,'checkpoint_metadata',{})
        return adapter

    def _project(self, visual_features, batch):
        per=self.geometry['visual_tokens_per_view']; dim=self.geometry['text_dim']
        allowed=((batch,6*per,dim),(batch,6,per,dim))
        if visual_features.dtype!=torch.float32 or tuple(visual_features.shape) not in allowed:
            raise ValueError(f'Projected features must be FP32 with shape {allowed}')
        if not torch.isfinite(visual_features).all(): raise ValueError('Nonfinite projected features')
        device=self.native.get_input_embeddings().weight.device
        return visual_features.to(device=device,non_blocking=True).reshape(batch*6,per,dim)
