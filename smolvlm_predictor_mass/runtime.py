"""Independent test-only lifecycle; no targets, optimizer, training, or teacher calls."""
import json
import os
from pathlib import Path
import sys
import time
import uuid
import zipfile

import numpy as np
import torch
from filelock import FileLock
from torch.utils.data import DataLoader
from smolvlm_attention.sample_store import atomic_json, digest, fingerprint
from smolvlm_predictor_io.compact import CompactDataset, CompactCollator, prepare_store
from smolvlm_map_predictor.evaluation import _read_commit, _stress_gate
from smolvlm_map_predictor.training import precision, move_batch, memory, close_loader, ProgressWriter

QUERY = dict(before='question_tokens_visual_softmax_then_masked_query_mean_v1',
             after2='last_valid_native_generation_prompt_token')

def execution_identity():
    from smolvlm_map_predictor.runtime import versions
    root=Path(__file__).resolve().parents[1]
    packages=('smolvlm_predictor_mass','smolvlm_predictor_io','smolvlm_map_predictor',
              'smolvlm_map_predictor_after2','smolvlm_topk_deferred','smolvlm_topk_benchmark','smolvlm_inference')
    return dict(versions=versions(),sources={str(p.relative_to(root)).replace('\\','/'):digest(p)
        for name in packages for p in sorted((root/name).glob('*.py'))})

def make_predictor(backbone, cfg):
    if cfg['variant'] == 'before':
        from smolvlm_map_predictor.training import make_predictor as make
    else:
        from smolvlm_map_predictor_after2.training import make_predictor as make
    return make(backbone, cfg)


def read_predictor(path, predictor, expected_identity, variant):
    """Verify semantic input identity, not training epochs/libraries/RNG locations."""
    value = torch.load(path, map_location='cpu', weights_only=False)
    identity = value.get('identity', {})
    # Targets are provenance only at inference: never open or recreate them.
    keys = ('checkpoint_sha256','model','base_checkpoint','prepared_hashes',
            'cache_payload_sha256','cache_manifest_sha256','cache_index_sha256','cache_fingerprint')
    if (value.get('format') != 'map_predictor_checkpoint_v1' or
            value.get('architecture_signature') != predictor.architecture_signature or
            any(identity.get(key) != expected_identity.get(key) for key in keys) or
            identity.get('predictor_query_policy') != QUERY[variant] or
            expected_identity.get('predictor_query_policy') != QUERY[variant]):
        raise ValueError('Predictor architecture/method/base/prepared/cache identity mismatch')
    if 'teacher' in expected_identity and identity.get('teacher') != expected_identity['teacher']:
        raise ValueError('Predictor teacher provenance semantics mismatch')
    if 'teacher' in expected_identity:
        provenance=identity.get('target_hashes',{})
        if set(provenance) != {'train','val'} or any(
                set(provenance[split]) != {'manifest.json','samples.jsonl','log_probs.npy'} or
                any(not isinstance(sha,str) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha)
                    for sha in provenance[split].values()) for split in ('train','val')):
            raise ValueError('Predictor teacher target checksum provenance missing or malformed')
    if identity.get('projected_cache') is not None and identity['projected_cache'] != expected_identity.get('projected_cache'):
        raise ValueError('Predictor projected cache identity mismatch')
    if variant == 'after2':
        from smolvlm_map_predictor_after2.config import METHOD
        expected = dict(method=METHOD,predictor_input_boundary='decoder_block_output',
                        predictor_input_layer_index=2,production_prune_after_layer_index=2)
        if any(identity.get(k) != v or expected_identity.get(k) != v for k,v in expected.items()):
            raise ValueError('Predictor after2 input boundary/method mismatch')
    predictor.load_state_dict(value['model_state_dict'], strict=True)
    return value


def prepare_test_store(model, projected, prepared, output, identity, *, expected_count=16817):
    """Only tokenize test, join six-camera keys, and use compact neutral reader."""
    from smolvlm_inference.cache import record_key
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    existing=output/'compact/store.json'
    if existing.is_file():
        from smolvlm_predictor_io.compact import read_store
        value=read_store(existing,include_targets=False)
        if (value['identity'] != identity or value['counts'] != {'test':expected_count} or
                value.get('projected_manifest') != str(projected.directory/'manifest.json')):
            raise ValueError('Existing test-only store identity/count/projected path mismatch')
        return existing
    source = Path(prepared)/'test.jsonl'; indices=[]; seen=set()
    with source.open(encoding='utf-8') as stream:
        for line in stream:
            item=json.loads(line); sid=item['sample_id']; key=record_key(item)
            if sid in seen or key not in projected.index:
                raise ValueError('Duplicate test sample or missing projected six-camera cache key')
            seen.add(sid); indices.append(projected.index[key])
    if len(indices) != expected_count: raise ValueError('Prepared test count mismatch')
    np.save(output/'features.npy',np.asarray(indices,dtype='<i8'))
    np.save(output/'unused_target_rows.npy',np.zeros(len(indices),dtype='<i8'))
    receipt=dict(identity=identity,geometry=model.geometry,counts={'test':len(indices)},
        max_length=model.native.config.text_config.max_position_embeddings,
        splits={'test':dict(source=str(source),features=str(output/'features.npy'),
                            target_rows=str(output/'unused_target_rows.npy'),target=None)})
    return prepare_store(receipt,model.processor,output/'compact',projected_path=projected.directory)


def prepare(cfg, output, predictor_checkpoint, *, projected_input='', projected_dir='',
            cache_input='', prepared_dir=None, base_checkpoint=None, resume=False):
    from smolvlm_predictor_io.setup import resolve_artifacts
    from smolvlm_predictor_io.orchestration import resolve_projected
    from smolvlm_predictor_io.projected import ProjectedCache
    from smolvlm_inference.model import load_model,CAMERA_ORDER
    from smolvlm_attention.config import MODELS
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    with FileLock(str(output/'parent.lock'),timeout=0):
        previous=json.loads((output/'run.json').read_text(encoding='utf-8')) if (output/'run.json').exists() else None
        if bool(previous) != bool(resume): raise ValueError('Use new output or explicitly resume existing run')
        precision()
        artifacts=resolve_artifacts(output/'internal/artifacts',cache_input,prepared_dir,base_checkpoint,
            projected_input=projected_input,projected_dir=projected_dir)
        model=load_model(artifacts['model']['checkpoint'],device='cpu',expected_prompt_camera_order=CAMERA_ORDER)
        for parameter in model.parameters(): parameter.requires_grad_(False)
        model.eval()
        projected_path=resolve_projected(cfg,artifacts,model,output,projected_input,projected_dir)
        projected=ProjectedCache(projected_path,full=False)
        origin=projected.manifest['identity']; raw=origin['raw_cache_descriptor']; manifest=origin['raw_cache_manifest']
        from smolvlm_attention.config import CACHE_FINGERPRINT
        if (manifest['fingerprint'] != CACHE_FINGERPRINT or manifest['geometry'] != model.geometry or
                origin['model'] != model.architecture_signature or origin['checkpoint_sha256'] != digest(artifacts['model']['checkpoint'])):
            raise ValueError('Projected base/model/cache geometry identity mismatch')
        identity=dict(checkpoint_sha256=digest(artifacts['model']['checkpoint']),model=model.architecture_signature,
            base_checkpoint=dict(repo=MODELS['base']['repo'],revision=MODELS['base']['revision'],filename='finetune_best.pt'),
            prepared_hashes=raw['prepared_hashes'],cache_payload_sha256=raw['payload_sha256'],
            cache_manifest_sha256=raw['manifest_sha256'],cache_index_sha256=raw['index_sha256'],
            cache_fingerprint=manifest['fingerprint'],predictor_query_policy=QUERY[cfg['variant']],
            teacher=dict(layer_index=22,layer_number=23,query_policy='last_valid_native_generation_prompt_token',
                         normalization='mean_heads_all_valid_keys_then_visual_normalization_log_v1'),
            projected_cache=dict(format=projected.manifest['format'],fingerprint=projected.manifest['fingerprint'],
                                 payload_sha256=projected.manifest['payload_sha256']))
        if cfg['variant']=='after2':
            from smolvlm_map_predictor_after2.config import METHOD
            identity.update(method=METHOD,predictor_input_boundary='decoder_block_output',
                            predictor_input_layer_index=2,production_prune_after_layer_index=2)
        with torch.random.fork_rng():
            predictor=make_predictor(model,cfg)
        read_predictor(predictor_checkpoint,predictor,identity,cfg['variant'])
        selected=Path(predictor_checkpoint).resolve()
        stable=dict(config=cfg,identity=identity,predictor_sha256=digest(selected),
                    projected_fingerprint=projected.manifest['fingerprint'],execution=execution_identity())
        if previous and previous['stable'] != stable: raise ValueError('Test-only resume identity/config mismatch')
        store=prepare_test_store(model,projected,artifacts['prepared_dir'],output/'internal/test',identity)
        run=dict(version=1,stable=stable,config=cfg,identity=identity,receipt=str(store),receipt_sha256=digest(store),
            output_dir=str(output),backbone_checkpoint=artifacts['model']['checkpoint'],predictor_checkpoint=str(selected))
        atomic_json(output/'runtime_config.json',cfg); atomic_json(output/'run.json',run)
        return output/'run.json'


def read_run(path):
    run=json.loads(Path(path).read_text(encoding='utf-8'))
    from smolvlm_predictor_io.compact import read_store
    if (run.get('version') != 1 or run['stable']['execution'] != execution_identity() or
            digest(run['receipt']) != run['receipt_sha256'] or
            digest(run['predictor_checkpoint']) != run['stable']['predictor_sha256'] or
            digest(run['backbone_checkpoint']) != run['identity']['checkpoint_sha256'] or
            read_store(run['receipt'],include_targets=False)['identity'] != run['identity']):
        raise ValueError('Test-only inputs/checkpoint/receipt changed')
    return run


def evaluation_identity(run):
    variant=run['config']['variant']; block=0 if variant=='before' else 3
    return dict(version='predictor_preserve_mass_hard_prune_v1',inputs=run['identity'],
        predictor_sha256=run['stable']['predictor_sha256'],architecture=run['config']['architecture'],
        projected_fingerprint=run['stable']['projected_fingerprint'],
        intervention_layer_index=block,gather_after_layer_index=block,
        query_scope='first_question_through_native_generation_prompt_prefill_only',
        mass_policy='per_head_query_visual_mass_keep_text_eps_1e-8',
        cache_policy='full_prefix_compact_suffix_original_rope_positions',query_chunk_size=64,
        generation=dict(run['config']['evaluation'],do_sample=False,num_beams=1,repetition_penalty=1.0),
        precision='fp32',tf32=False,autocast=False,partition='test_original_rows_rank_stride_v1')


def worker(run,rank,*,device='cuda:0',backbone=None):
    directory=Path(run['output_dir'])/'evaluation'; directory.mkdir(exist_ok=True)
    with FileLock(str(directory/f'rank_{rank}.lock'),timeout=0):
        return _worker(run,rank,device=device,backbone=backbone)


def _worker(run,rank,*,device='cuda:0',backbone=None):
    from .generation import Decoder
    from smolvlm_predictor_io.progress import StatusWriter
    from smolvlm_inference.model import load_model,CAMERA_ORDER
    precision(); device=torch.device(device)
    backbone=backbone or load_model(run['backbone_checkpoint'],device=str(device),expected_prompt_camera_order=CAMERA_ORDER)
    backbone=backbone.to(device=device,dtype=torch.float32).eval()
    for parameter in backbone.parameters(): parameter.requires_grad_(False)
    with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
        predictor=make_predictor(backbone,run['config']).to(device).eval()
    read_predictor(run['predictor_checkpoint'],predictor,run['identity'],run['config']['variant'])
    identity=evaluation_identity(run); root=Path(run['output_dir']); directory=root/'evaluation'; directory.mkdir(exist_ok=True)
    with FileLock(str(directory/'identity.lock')):
        path=directory/'identity.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) != identity: raise ValueError('Evaluation identity mismatch')
        atomic_json(path,identity)
    world=int(run.get('_evaluation_world_size',2))
    if not 0<=rank<world: raise ValueError('Invalid rank')
    dataset=CompactDataset(run['receipt'],'test'); size=run['config']['evaluation']['batch_size']; keep=run['config']['evaluation']['keep_tokens']
    rows=list(range(rank,len(dataset),world)); schedules=[rows[i:i+size] for i in range(0,len(rows),size)]
    if not rows: raise ValueError('Empty test shard')
    shard=directory/f'rank_{rank}'; shard.mkdir(exist_ok=True); pending=[]
    for index,scheduled in enumerate(schedules):
        path=shard/f'batch_{index:06d}.json'
        if path.exists(): _read_commit(path,identity,rank,index,scheduled,[dataset[i] for i in scheduled],keep)
        else: pending.append(index)
    lengths=dataset.prompt_lengths; stress=sorted(rows,key=lambda row:(-int(lengths[row]),row))[:size]
    loader=DataLoader(dataset,batch_sampler=[stress]+[schedules[i] for i in pending],
        collate_fn=CompactCollator(run['receipt'],max_new_tokens=run['config']['evaluation']['max_new_tokens']),
        num_workers=2,persistent_workers=True,pin_memory=True,prefetch_factor=1,multiprocessing_context='spawn',
        generator=torch.Generator().manual_seed(42))
    decoder=Decoder(backbone,predictor,variant=run['config']['variant'],keep_tokens=keep)
    progress=ProgressWriter(root,filename=f'evaluate_rank_{rank}.jsonl'); live=StatusWriter(root,rank)
    try:
        iterator=iter(loader)
        def generate(batch,debug=False):
            with torch.inference_mode():
                return decoder.generate(move_batch(batch,device),max_new_tokens=run['config']['evaluation']['max_new_tokens'],debug=debug)
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        generate(next(iterator),True)
        atomic_json(root/'logs'/f'first_batch_rank_{rank}.json',decoder.last_debug)
        from smolvlm_map_predictor.runtime import resource_snapshot
        atomic_json(root/'logs'/f'stress_resources_rank_{rank}.json',dict(memory=memory(device),resources=resource_snapshot()))
        _stress_gate(run,directory,rank,world,identity,dict(ok=True,rows=stress,samples=len(stress),**memory(device)))
        processed=len(rows)-sum(len(schedules[i]) for i in pending); resumed=processed; started=time.perf_counter()
        for index,batch in zip(pending,iterator):
            scheduled=schedules[index]
            if batch['rows'].tolist()!=scheduled: raise ValueError('Loader changed fixed test schedule')
            outputs=generate(batch); items=[dataset[i] for i in scheduled]
            if len(outputs)!=len(items): raise ValueError('Decoder output batch count mismatch')
            records=[dict(result,prepared_row=row,sample_id=item['row']['sample_id'],eval_id=item['row'].get('eval_id',item['row']['sample_id']),
                          question=item['row']['question'],reference=item['row'].get('answer','')) for result,item,row in zip(outputs,items,scheduled)]
            commit=dict(version=1,identity_sha256=fingerprint(identity),rank=rank,batch_index=index,rows=scheduled,
                        sample_ids=batch['sample_ids'],records=records,records_sha256=fingerprint(records))
            temporary=shard/f'.batch_{index:06d}.check.json'; atomic_json(temporary,commit)
            _read_commit(temporary,identity,rank,index,scheduled,items,keep)
            os.replace(temporary,shard/f'batch_{index:06d}.json')
            processed+=len(scheduled); live.update('test',processed,len(rows))
            elapsed=time.perf_counter()-started; rate=(processed-resumed)/elapsed
            progress.write('evaluation_batch',rank=rank,processed=processed,total=len(rows),elapsed_seconds=elapsed,
                           samples_per_second=rate,estimated_remaining_seconds=(len(rows)-processed)/rate,**memory(device))
        atomic_json(shard/'complete.json',dict(identity_sha256=fingerprint(identity),rank=rank,count=len(rows),
            batch_checksums={f'batch_{i:06d}.json':digest(shard/f'batch_{i:06d}.json') for i in range(len(schedules))}))
        live.update('test',processed,len(rows),force=True)
    finally: close_loader(loader)


def finalize(run):
    """Existing strict record/coverage/metrics protocol with new method identity."""
    from smolvlm_topk_benchmark.scoring import score_rows
    identity=evaluation_identity(run); root=Path(run['output_dir']); directory=root/'evaluation'
    if json.loads((directory/'identity.json').read_text(encoding='utf-8'))!=identity: raise ValueError('Evaluation identity mismatch')
    dataset=CompactDataset(run['receipt'],'test'); merged={}; world=int(run.get('_evaluation_world_size',2))
    size=run['config']['evaluation']['batch_size']; keep=run['config']['evaluation']['keep_tokens']
    for rank in range(world):
        rows=list(range(rank,len(dataset),world)); schedules=[rows[i:i+size] for i in range(0,len(rows),size)]; shard=directory/f'rank_{rank}'
        expected=dict(identity_sha256=fingerprint(identity),rank=rank,count=len(rows),
            batch_checksums={f'batch_{i:06d}.json':digest(shard/f'batch_{i:06d}.json') for i in range(len(schedules))})
        if json.loads((shard/'complete.json').read_text(encoding='utf-8'))!=expected: raise ValueError('Shard coverage/checksum mismatch')
        for index,scheduled in enumerate(schedules):
            value=_read_commit(shard/f'batch_{index:06d}.json',identity,rank,index,scheduled,[dataset[i] for i in scheduled],keep)
            for record in value['records']:
                if record['prepared_row'] in merged: raise ValueError('Duplicate test row')
                merged[record['prepared_row']]=record
    if set(merged)!=set(range(len(dataset))): raise ValueError('Incomplete test coverage')
    records=[merged[i] for i in range(len(dataset))]; result=score_rows(records,expected_count=len(dataset))
    temporary=root/'predictions.jsonl.tmp'
    with temporary.open('w',encoding='utf-8') as stream:
        for record in records: stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,root/'predictions.jsonl')
    for key in ('metrics','metrics_display','scoring_identity'): atomic_json(root/f'{key}.json',result[key])
    atomic_json(root/'evaluation_summary.json',dict(identity=identity,metrics_display=result['metrics_display'],
        diagnostics=result['diagnostics'],prediction_sha256=digest(root/'predictions.jsonl'),scope='greedy_QA_quality_no_latency_or_FLOPs'))


def launch(run_path):
    from smolvlm_map_predictor.runtime import _parallel
    run=read_run(run_path)
    if torch.cuda.device_count()<2: raise RuntimeError('Two CUDA GPUs required; no automatic fallback')
    root=Path(run['output_dir']); logs=root/'logs'; logs.mkdir(exist_ok=True)
    from smolvlm_topk_benchmark.scoring import metric_preflight
    atomic_json(logs/'metric_preflight.json',metric_preflight())
    visible=[x.strip() for x in os.environ.get('CUDA_VISIBLE_DEVICES','0,1').split(',') if x.strip()]
    if len(visible)<2: raise RuntimeError('Two visible devices required')
    with FileLock(str(root/'parent.lock'),timeout=0):
        env=dict(os.environ,USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                 SMOLVLM_MAP_EVALUATION_SESSION=uuid.uuid4().hex)
        _parallel([([sys.executable,'-m','smolvlm_predictor_mass','worker','--run',str(Path(run_path).resolve()),'--rank',str(rank)],
            dict(env,CUDA_VISIBLE_DEVICES=visible[rank]),logs/f'evaluate_gpu_{rank}.log') for rank in (0,1)])
        finalize(run)


def bundle(run_path):
    run=read_run(run_path); root=Path(run['output_dir']); target=root/'predictor_preserve_mass_artifacts.zip'
    with FileLock(str(root/'parent.lock'),timeout=0):
        return _bundle(run,target)


def _bundle(run,target):
    root=Path(run['output_dir'])
    if not (root/'evaluation_summary.json').is_file(): raise ValueError('Complete evaluation before bundling')
    summary=json.loads((root/'evaluation_summary.json').read_text(encoding='utf-8'))
    if summary.get('identity')!=evaluation_identity(run) or summary.get('prediction_sha256')!=digest(root/'predictions.jsonl'):
        raise ValueError('Evaluation summary/predictions changed before bundle')
    names=('runtime_config.json','run.json','predictions.jsonl','metrics.json','metrics_display.json','scoring_identity.json','evaluation_summary.json')
    with zipfile.ZipFile(target.with_suffix('.zip.tmp'),'w',zipfile.ZIP_DEFLATED) as archive:
        for name in names: archive.write(root/name,name)
        for path in (root/'logs').glob('*'):
            if path.is_file(): archive.write(path,'logs/'+path.name)
        archive.write(run['predictor_checkpoint'],'checkpoints/predictor_best.pt')
    os.replace(target.with_suffix('.zip.tmp'),target)
    atomic_json(root/'predictor_preserve_mass_artifacts.sha256.json',dict(filename=target.name,sha256=digest(target)))
    return target
