"""MVPruner with immutable base layer-22 scores; native attention stays unchanged."""
import argparse
import json
import os
from pathlib import Path
import zipfile
import subprocess
import sys
import time

import numpy as np
import torch

from . import pruning_run as common
from .cache import CAMERA_ORDER, digest
from .decoder import PrunedGreedyDecoder
from .model import load_model
from .pruning_text import PruningBatchCollator
from smolvlm_attention.sample_inputs import prepare_inputs
from smolvlm_topk_benchmark.data import find_attention, check_export


def validate_config(config):
    expected = dict(common.FIXED, stage2_layer_index=2,
                    stage2_score_source='saved_base_attention', score_layer_index=22)
    if config != expected or any(type(config[k]) is not type(v) for k,v in expected.items()):
        raise ValueError('Expected fixed prune-index2 score-index22 configuration')
    return config


def validate_export(export, rows, identity, checkpoint_sha256, expected_count=16817):
    if (export.metadata.get('axes') != ['sample','layer','visual_token']
            or export.metadata.get('layer_ids') != list(range(30))):
        raise ValueError('Export axes/layer order differs')
    return check_export(export, rows, identity, checkpoint_sha256, expected_count=expected_count)


def batch_scores(export, rows):
    values = np.asarray(export.attention[[r['attention_row'] for r in rows],22,:])
    if (values.shape != (len(rows),384) or values.dtype != np.float32
            or not np.isfinite(values).all() or (values < 0).any()
            or (values.sum(1,dtype=np.float64) > 1+1e-6).any()):
        raise ValueError('Invalid saved layer-22 attention probabilities')
    return torch.from_numpy(np.array(values, copy=True))


class SavedScoreCollator:
    def __init__(self, collator, export):
        self.collator, self.export = collator, export
    def prompt_length(self, row):
        return self.collator.prompt_length(row)
    def __call__(self, items):
        result = self.collator(items)
        result['stage2_visual_scores'] = batch_scores(self.export, items)
        return result


class SavedScoreDecoder(PrunedGreedyDecoder):
    def generate(self, batch, **kwargs):
        result = super().generate(batch, stage2_visual_scores=batch['stage2_visual_scores'], **kwargs)
        if kwargs.get('debug'):
            result['debug'].update(stage2_score_source='saved_base_attention',
                                   score_layer_index=22,prune_after_layer_index=2)
        return result


def run(args):
    config = validate_config(json.loads(Path(args.config).read_text()))
    if args.worker_rank is None:
        common.preflight_metrics()
    common.configure_fp32()
    out = Path(args.output)
    if any((out/name).exists() for name in ('predictions.jsonl','metadata.json')):
        raise ValueError('Output already contains a run; choose a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    directory, export = find_attention(args.attention_input)
    rows, _, identity = prepare_inputs(dict(cache_dir=args.cache, prepared_dir=args.prepared), out)
    checkpoint_sha = digest(args.checkpoint)
    rows = validate_export(export, rows, identity, checkpoint_sha)
    total_count = len(rows)
    if args.worker_rank is not None:
        rows = shard_rows(rows, args.worker_rank, 2)
    cache = common.open_validated_cache(args.cache,args.prepared,common.cache_identity(),args.receipt)
    model = load_model(args.checkpoint,device='cuda:0',expected_prompt_camera_order=CAMERA_ORDER)
    collator = SavedScoreCollator(PruningBatchCollator(model.processor,
        image_seq_len=model.geometry['visual_tokens_per_view'],
        max_length=model.native.config.text_config.max_position_embeddings,
        max_new_tokens=512,prompt_camera_order=CAMERA_ORDER),export)
    decoder = SavedScoreDecoder(model,2,stage1_keep=196,stage2_keep=32)
    import transformers
    metadata = dict(implementation_version='mvpruner_saved_base_attention_v1',config=config,
        stage2_score_source='saved_base_attention',score_layer_index=22,prune_after_layer_index=2,
        checkpoint_sha256=checkpoint_sha,checkpoint_revision=common.CHECKPOINT_REVISION,
        prepared_revision=common.PREPARED_REVISION,camera_order=CAMERA_ORDER,dtype='float32',
        autocast=False,tf32=False,torch_version=torch.__version__,transformers_version=transformers.__version__,
        model_id=model.model_id,model_revision=model.model_revision,
        cache_fingerprint=cache.manifest['fingerprint'],test_count=len(rows),
        eos_token_id=model.native.generation_config.eos_token_id,
        pad_token_id=model.native.generation_config.pad_token_id,
        allocator={k:os.environ.get(k) for k in ('PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF')},
        export_directory=str(directory),export_fingerprint=export.metadata['fingerprint'],
        export_manifest=json.loads((directory/'manifest.json').read_text()),
        query_policy=export.metadata['identity']['policy'],
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        worker_rank=args.worker_rank,world_size=2 if args.worker_rank is not None else 1,
        full_test_count=total_count,
        timing_scope='inference_with_precomputed_scores_excludes_export_creation')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    common.evaluate_batches(model,cache,rows,collator,decoder,out/'stress','stress')
    if args.worker_rank is not None:
        (out/'stress.ready').touch()
        while not (out.parent/'test.go').exists():
            time.sleep(1)
    common.evaluate_batches(model,cache,rows,collator,decoder,out,'test')
    if args.worker_rank is not None:
        return
    common.score_predictions(out/'predictions.jsonl')
    archive_outputs(out)


def archive_outputs(out):
    with zipfile.ZipFile(out/'mvpruner_saved_attention_artifacts.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.suffix in ('.json','.jsonl','.log') and 'validation' not in path.relative_to(out).parts:
                archive.write(path,path.relative_to(out))


def shard_rows(rows, rank, world_size=2):
    if world_size != 2 or rank not in (0,1):
        raise ValueError('Expected two independent GPU shards')
    return rows[rank::world_size]


def merge_shards(out, expected):
    out=Path(out)
    merged=[]
    metadata=[]
    for rank in range(2):
        folder=out/f'rank_{rank}'
        values=list(common.records(folder/'predictions.jsonl'))
        common.validate_coverage(values,shard_rows(expected,rank))
        merged.extend(values)
        metadata.append(json.loads((folder/'metadata.json').read_text()))
    for key in ('config','checkpoint_sha256','export_fingerprint','camera_order','dtype','query_policy'):
        if metadata[0][key] != metadata[1][key]:
            raise ValueError(f'GPU shard identity differs: {key}')
    common.validate_coverage(merged,expected)
    merged.sort(key=lambda row:int(row['eval_id']))
    with (out/'predictions.jsonl').open('w',encoding='utf-8') as stream:
        for row in merged:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    meta=dict(metadata[0],worker_rank=None,world_size=2,test_count=len(merged),
              execution='two_gpu_data_shards',shard_counts=[len(shard_rows(expected,r)) for r in range(2)])
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    return merged


def launch(args):
    if torch.cuda.device_count() < 2:
        raise RuntimeError('Two GPUs required for --gpus 2')
    validate_config(json.loads(Path(args.config).read_text()))
    common.preflight_metrics()
    out=Path(args.output)
    if any((out/name).exists() for name in ('rank_0','rank_1','test.go','metadata.json','predictions.jsonl')):
        raise ValueError('Output already contains a run; choose a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    # Respect physical GPU mapping supplied by the notebook/container.
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    devices=visible.split(',')[:2] if visible else ['0','1']
    jobs=[]
    try:
        for rank in range(2):
            folder=out/f'rank_{rank}'
            folder.mkdir()
            log=(folder/'worker.log').open('w',encoding='utf-8')
            argv=[sys.executable,'-m','smolvlm_inference.saved_attention_run']
            for name in ('cache','prepared','receipt','checkpoint','config','attention_input'):
                argv += ['--'+name.replace('_','-'),str(getattr(args,name))]
            argv += ['--output',str(folder),'--worker-rank',str(rank)]
            process=subprocess.Popen(argv,env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[rank]),
                stdout=log,stderr=subprocess.STDOUT)
            jobs.append((process,log,rank))
        last_report=0
        while True:
            for process,_,rank in jobs:
                status=process.poll()
                if status is not None and status != 0:
                    raise RuntimeError(f'GPU shard {rank} failed ({status}); inspect rank_{rank}/worker.log. No retries.')
                if status == 0 and not (out/'test.go').exists():
                    raise RuntimeError(f'GPU shard {rank} exited before stress barrier')
            if not (out/'test.go').exists() and all((out/f'rank_{r}'/'stress.ready').exists() for r in range(2)):
                (out/'test.go').touch()
                common.emit(stage='stress',status='both_passed_start_test')
            if all(p.poll() is not None for p,_,_ in jobs):
                break
            now=time.monotonic()
            if now-last_report>=30:
                for process,_,rank in jobs:
                    progress=out/f'rank_{rank}'/'progress.jsonl'
                    latest=None
                    if progress.exists():
                        for line in progress.read_text().splitlines():
                            try:latest=json.loads(line)
                            except json.JSONDecodeError:pass
                    common.emit(stage='two_gpu_saved_attention',rank=rank,status=process.poll(),latest_progress=latest)
                last_report=now
            time.sleep(1)
        expected=list(common.records(Path(args.prepared)/'test.jsonl'))
        merge_shards(out,expected)
        common.score_predictions(out/'predictions.jsonl')
        archive_outputs(out)
    finally:
        for process,log,_ in jobs:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait()
            log.close()


def main(argv=None):
    parser=argparse.ArgumentParser(__doc__)
    for name in ('cache','prepared','receipt','checkpoint','config','attention-input','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--gpus',type=int,choices=(1,2),default=1)
    parser.add_argument('--worker-rank',type=int,choices=(0,1),default=None,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.gpus == 2:
        if args.worker_rank is not None:
            parser.error('worker cannot launch workers')
        launch(args)
    else:
        run(args)

if __name__ == '__main__':
    main()
