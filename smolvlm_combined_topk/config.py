"""Fixed combined-score Top-32 experiment configuration."""
LAYER_IDS = (10,15,16,17,21,22,23,25,26,27)
LAYERS = LAYER_IDS
BATCH_SIZE = 64
K = 32
MAX_NEW_TOKENS = 512
TEST_COUNT = 16817
CONFIGS = tuple(f'layer_{layer:02d}' for layer in LAYER_IDS)
FORMAT = 'smolvlm_combined_visual_per_sample_fp32_v1'
FORMULA = 'mean_heads(A)[q,v] + sum(nonvisual valid t != q, mean_heads(A)[q,t] * mean_heads(A)[t,v])'

def worker_configs(worker):
    if worker not in (0,1): raise ValueError('Exactly two GPU workers are required')
    return CONFIGS[worker*5:(worker+1)*5]
