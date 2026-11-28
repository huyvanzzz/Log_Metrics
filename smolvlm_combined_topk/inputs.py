"""Tokenize test once; preserve full native-prompt and six-camera row mapping."""
import json
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from smolvlm_attention.sample_store import atomic_json,atomic_bytes,digest
from smolvlm_inference.cache import record_key
from smolvlm_predictor_io.compact import prepare_store,read_store,CompactDataset,CompactCollator


def prepare_inputs(model, projected, prepared_dir, output, identity=None, *, expected_count=16817):
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    source=Path(prepared_dir).resolve()/'test.jsonl'
    spec=dict(identity=identity or {},source_sha256=digest(source),count=expected_count,
              projected_manifest_sha256=digest(projected.directory/'manifest.json'))
    spec_path=output/'input_identity.json'; samples_path=output/'samples.jsonl'; store=output/'compact/store.json'
    if spec_path.is_file():
        previous=json.loads(spec_path.read_text(encoding='utf-8'))
        if previous['spec']!=spec: raise ValueError('Existing prepared inputs identity mismatch')
        if digest(samples_path)!=previous['samples_sha256']: raise ValueError('Prepared sample mapping checksum mismatch')
        for relative,sha in previous.get('compact_content_sha256',{}).items():
            if digest(output/relative)!=sha: raise ValueError(f'Compact prompt content changed: {relative}')
        read_store(store,include_targets=False)
        rows=[json.loads(line) for line in samples_path.read_text(encoding='utf-8').splitlines()]
        return dict(store=str(store),samples_path=str(samples_path),rows=rows)
    originals=[]; indices=[]; seen=set()
    with source.open(encoding='utf-8') as stream:
        for line in tqdm(stream,total=expected_count,desc='Join test projected features',unit='question'):
            item=json.loads(line); key=record_key(item); sid=item['sample_id']
            if sid in seen or key not in projected.index: raise ValueError('Duplicate sample ID or missing projected six-camera key')
            if item.get('eval_id')!=len(originals): raise ValueError('Prepared test eval_id order must be dense 0..N-1')
            seen.add(sid); originals.append(item); indices.append(projected.index[key])
    if len(originals)!=expected_count: raise ValueError('Prepared test count mismatch')
    np.save(output/'features.npy',np.asarray(indices,dtype='<i8'))
    np.save(output/'unused_target_rows.npy',np.zeros(len(indices),dtype='<i8'))
    receipt=dict(identity=spec,geometry=model.geometry,counts={'test':len(indices)},
        max_length=model.native.config.text_config.max_position_embeddings,
        splits={'test':dict(source=str(source),features=str(output/'features.npy'),
            target_rows=str(output/'unused_target_rows.npy'),target=None)})
    # The neutral compact store tokenizes each native prompt exactly once and has
    # no teacher targets. Reopening it never invokes a tokenizer.
    with tqdm(total=1,desc=f'Tokenize {expected_count} native test prompts',unit='stage') as bar:
        store=prepare_store(receipt,model.processor,output/'compact',projected_path=projected.directory)
        bar.update(1)
    dataset=CompactDataset(store,'test'); rows=[]
    image_id=model.native.config.image_token_id
    for i,original in enumerate(tqdm(originals,desc='Save native prompt mapping',unit='question')):
        item=dataset[i]; ids=item['tokens'].tolist(); images=item['image_columns'].tolist()
        row=dict(original,row_index=i,feature_key=record_key(original),feature_index=indices[i],
            input_ids=ids,prompt_length=len(ids),query_position=len(ids)-1,query_token_id=ids[-1],
            image_token_id=image_id,image_positions=images)
        if len(images)!=model.visual_token_len or any(ids[j]!=image_id for j in images): raise ValueError('Invalid native visual position mapping')
        rows.append(row)
    atomic_bytes(samples_path,''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode('utf-8'))
    content_files=[p for p in (output/'compact').iterdir() if p.is_file() and p.name!='store.json']
    content_files.extend((output/'features.npy',output/'unused_target_rows.npy'))
    compact_hashes={p.relative_to(output).as_posix():digest(p) for p in sorted(content_files)}
    atomic_json(spec_path,dict(spec=spec,samples_sha256=digest(samples_path),compact_content_sha256=compact_hashes))
    return dict(store=str(store),samples_path=str(samples_path),rows=rows)


def make_batch_builder(store, *, max_new_tokens=0):
    dataset=CompactDataset(store,'test'); collator=CompactCollator(store,max_new_tokens=max_new_tokens)
    def build(rows):
        indices=[int(r['row_index']) if isinstance(r,dict) else int(r) for r in rows]
        batch=collator([dataset[i] for i in indices])
        batch['visual_features']=batch.pop('projected_visual_features')
        return batch
    return build
