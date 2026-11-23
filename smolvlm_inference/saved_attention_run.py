"""MVPruner with immutable base layer-22 scores; native attention stays unchanged."""
import argparse
import json
import os
from pathlib import Path
import zipfile

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
        timing_scope='inference_with_precomputed_scores_excludes_export_creation')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    common.evaluate_batches(model,cache,rows,collator,decoder,out/'stress','stress')
    common.evaluate_batches(model,cache,rows,collator,decoder,out,'test')
    common.score_predictions(out/'predictions.jsonl')
    with zipfile.ZipFile(out/'mvpruner_saved_attention_artifacts.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.suffix in ('.json','.jsonl','.log') and 'validation' not in path.relative_to(out).parts:
                archive.write(path,path.relative_to(out))


def main(argv=None):
    parser=argparse.ArgumentParser(__doc__)
    for name in ('cache','prepared','receipt','checkpoint','config','attention-input','output'):
        parser.add_argument('--'+name,required=True)
    run(parser.parse_args(argv))

if __name__ == '__main__':
    main()
