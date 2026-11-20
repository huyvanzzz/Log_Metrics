"""Verified attention/data alignment and deterministic single-layer selections."""
import json
from pathlib import Path

import numpy as np

from smolvlm_attention.sample_store import open_prompt_attention, digest
from .config import K, TEST_COUNT


def topk_indices(values,k=K):
    values=np.asarray(values)
    if (values.ndim!=2 or values.shape[1]!=384 or values.dtype!=np.float32
            or not np.isfinite(values).all() or (values<0).any()
            or (values.sum(-1,dtype=np.float64)>1+1e-6).any() or not 0<k<=384):
        raise ValueError('Expected finite FP32 probabilities over 384 visual positions')
    return np.sort(np.argsort(-values,axis=-1,kind='stable')[:,:k],axis=-1).astype(np.uint16)


def find_attention(attention_input=None,roots=None):
    roots=[Path(attention_input)] if attention_input else list(map(Path,roots or (
        '/kaggle/input','/kaggle/working/smolvlm_prompt_attention_outputs/export')))
    candidates=set()
    for root in roots:
        if root.is_file():root=root.parent
        if (root/'attention.npy').is_file():candidates.add(root.resolve())
        if root.is_dir():
            candidates.update(p.parent.resolve() for p in root.rglob('attention.npy')
                              if (p.parent/'manifest.json').is_file() and (p.parent/'samples.jsonl').is_file())
    if len(candidates)!=1:
        raise ValueError(f'Expected one complete per-question attention export; found {len(candidates)}. Set ATTENTION_INPUT to its folder.')
    directory=candidates.pop()
    return directory,open_prompt_attention(directory)


def aligned_rows(expected,exported):
    lookup={r['sample_id']:r for r in exported}
    if len(lookup)!=len(exported) or set(lookup)!={r['sample_id'] for r in expected}:
        raise ValueError('Attention test IDs differ')
    fields=('eval_id','question','answer','camera_paths','input_ids','image_token_id','feature_key',
            'feature_index','prompt_length','query_position','query_token_id')
    rows=[]
    for source in expected:
        saved=lookup[source['sample_id']]
        for key in fields:
            if source[key]!=saved[key]:raise ValueError(f'Attention/sample mismatch: {source["sample_id"]} {key}')
        rows.append(dict(source,attention_row=saved['row_index']))
    return rows


def check_export(export,candidate_rows,data_identity,checkpoint_sha256,*,expected_count=TEST_COUNT):
    from smolvlm_attention.config import MODELS,QUERY_POLICY
    from smolvlm_inference.cache import CAMERA_ORDER
    meta=export.metadata; identity=meta['identity']
    if (export.attention.shape!=(expected_count,30,384) or len(candidate_rows)!=expected_count
            or identity['model']!=MODELS['base']['repo'] or identity['revision']!=MODELS['base']['revision']
            or identity['checkpoint_sha256']!=checkpoint_sha256 or identity['policy']!=QUERY_POLICY
            or identity['reduction']!='mean_heads_only_per_sample_v1'
            or identity['camera_order']!=CAMERA_ORDER or meta['camera_order']!=CAMERA_ORDER):
        raise ValueError('Attention export is not the pinned base full-test last-prompt-query measurement')
    for key in ('cache_fingerprint','prepared_hashes','cache_manifest_sha256','cache_index_sha256',
                'model_id','native_revision','processor_policy','geometry','image_token_id',
                'tokenizer_class','tokenizer_model_max_length'):
        if identity['data'][key]!=data_identity[key]:
            raise ValueError(f'Attention model/data identity differs: {key}')
    return aligned_rows(candidate_rows,export.samples)


def prepare_selection(run,name,export):
    from .config import config_layer
    from .state import input_rows,write_selection
    if name=='base':return write_selection(run,name)
    rows=input_rows(run); layer=config_layer(name)
    indices=np.empty((len(rows),K),np.uint16)
    positions=np.empty((len(rows),K),np.int32)
    from tqdm.auto import tqdm
    for start in tqdm(range(0,len(rows),256),desc=f'Top-32 {name}',unit='block'):
        selected=rows[start:start+256]
        values=np.asarray(export.attention[[r['attention_row'] for r in selected],layer,:])
        keep=topk_indices(values)
        indices[start:start+len(selected)]=keep
        for offset,(row,visual) in enumerate(zip(selected,keep)):
            image_positions=np.flatnonzero(np.asarray(row['input_ids'])==row['image_token_id'])
            if len(image_positions)!=384:raise ValueError('Expected 384 valid image positions')
            positions[start+offset]=image_positions[visual]
    return write_selection(run,name,indices,positions)
