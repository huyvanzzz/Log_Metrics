"""Independent inference settings; training recipe is intentionally absent."""
import json
from pathlib import Path

def load_config(path=None, *, variant='before'):
    root = Path(__file__).resolve().parents[1]
    path = path or root / 'configs' / f'smolvlm_predictor_{variant}_mmr.json'
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    if set(cfg) != {'version','variant','architecture','evaluation','output_dir'} or cfg['version'] != 1:
        raise ValueError('Unsupported test-only MMR config')
    if cfg['variant'] not in ('before','after2') or cfg['architecture'] != dict(hidden_dim=128,projection_dim=64,visual_tokens=384):
        raise ValueError('Unsupported MMR predictor architecture')
    if cfg['evaluation'] != dict(batch_size=64,keep_tokens=32,max_new_tokens=512):
        raise ValueError('MMR test requires batch 64, MMR-32, greedy budget 512')
    return cfg
