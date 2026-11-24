"""Production recipe for direct prediction from decoder block index two."""
from copy import deepcopy
import json
from pathlib import Path
from smolvlm_map_predictor.config import SOURCE_TRAINING_COMMIT, recipe as original_recipe

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs/smolvlm_map_predictor_after2_lastquery_index22.json'
METHOD = dict(id='map_after2_lastprompt_direct_v1', input_layer_index=2,
              query_policy='last_valid_native_generation_prompt_token',
              prune_after_layer_index=2, teacher_layer_index=22)
_INPUTS = dict(checkpoint_repo='huyvanzzz/smolvlm256M_base',
    checkpoint_revision='f9d4e9a4d704078f9e7f39f061d13602cf251c78',
    prepared_repo='huyvanzzz/prepared_data', prepared_revision='d2a6914f681dbb9148c4ac11e6fa92452010506e',
    cache_dataset='huyhuy12334/cache-smolvlm/versions/1',
    cache_fingerprint='0632954bab4d7a8e5af5934f3476610a0547db8f3083495c1207c3da87b15585')


def load_config(path=DEFAULT_CONFIG):
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    required = {'version','method','architecture','training','evaluation','inputs','output_dir'}
    if set(cfg) != required or cfg['version'] != 1:
        raise ValueError('Unsupported after2 predictor configuration')
    if cfg['method'] != METHOD:
        raise ValueError('Unsupported after2 method')
    if cfg['inputs'] != _INPUTS:
        raise ValueError('Production input pins differ')
    if cfg['architecture'] != dict(hidden_dim=128, projection_dim=64, visual_tokens=384):
        raise ValueError('Production predictor requires hidden 128, projection 64, visual tokens 384')
    expected = dict(precision='fp32', batch_size=32, gradient_accumulation_steps=1, epochs=10,
        world_size=2, seed=42, learning_rate=1e-4, weight_decay=.05, scheduler_gamma=.9,
        max_grad_norm=1., num_workers=2, prefetch_factor=1, persistent_workers=True,
        pin_memory=True, multiprocessing_context='spawn', progress_log_every_steps=500)
    if cfg['training'] != expected:
        raise ValueError('Production training recipe differs')
    if cfg['evaluation'] != dict(batch_size=64, max_new_tokens=512, keep_tokens=32):
        raise ValueError('Evaluation requires batch 64, greedy budget 512 and Top-32')
    return cfg


def recipe(cfg):
    result = original_recipe(cfg)
    from .runtime import sources, versions
    result.update(backend=METHOD['id'], method=deepcopy(METHOD),
                  execution=dict(sources=sources(), versions=versions()))
    return result
