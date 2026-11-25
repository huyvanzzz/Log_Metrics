"""Dedicated predictor recipe; no changes to the existing model profiles."""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs/smolvlm_map_predictor_index22.json'
SOURCE_TRAINING_COMMIT = '4d7dbd238c7960d305c882b75220fb95beb37644'


def load_config(path=DEFAULT_CONFIG):
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    required = {'version', 'architecture', 'training', 'evaluation', 'inputs', 'output_dir'}
    if set(cfg) != required or cfg['version'] != 1:
        raise ValueError('Unsupported predictor configuration')
    if cfg['architecture'] != {'hidden_dim': 128, 'projection_dim': 64, 'visual_tokens': 384}:
        raise ValueError('Production predictor requires hidden 128, projection 64, visual tokens 384')
    train = cfg['training']
    for name in ('batch_size', 'gradient_accumulation_steps', 'epochs', 'num_workers', 'prefetch_factor', 'progress_log_every_steps'):
        if type(train.get(name)) is not int or train[name] < 1:
            raise ValueError(f'Invalid training {name}')
    if any(train.get(k) != v for k,v in dict(batch_size=32,gradient_accumulation_steps=1,num_workers=2,prefetch_factor=1).items()):
        raise ValueError('Production training requires batch 32, accumulation 1, workers 2 and prefetch 1')
    if train.get('precision') != 'fp32' or train.get('seed') != 42:
        raise ValueError('Predictor uses FP32 and seed 42')
    if train.get('world_size') != 2 or train.get('multiprocessing_context') != 'spawn':
        raise ValueError('Kaggle predictor requires two ranks and spawn loader workers')
    if not train.get('pin_memory') or not train.get('persistent_workers'):
        raise ValueError('Pinned persistent loading is required')
    for name in ('learning_rate', 'weight_decay', 'scheduler_gamma', 'max_grad_norm'):
        value = train.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < float('inf'):
            raise ValueError(f'Invalid training {name}')
    if cfg['evaluation'] != {'batch_size': 64, 'max_new_tokens': 512, 'keep_tokens': 32}:
        raise ValueError('Evaluation requires batch 64, greedy budget 512 and Top-32')
    return cfg


def recipe(cfg):
    return {'backend': 'map_predictor_v1', 'training': deepcopy(cfg['training']),
            'objective': 'mean_microbatch_kl_teacher_to_predictor',
            'validation': 'sample_weighted_kl', 'accumulation_version': 2,
            'sampler': 'accelerate_seedable_even_batches_v1',
            'rng_policy': 'post_prepare_seed_plus_rank_v1'}
