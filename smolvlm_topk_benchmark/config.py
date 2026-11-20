"""Fixed experiment: fifteen single-layer rankings and one unpruned baseline."""
BATCH_SIZE = 32
K = 32
MAX_NEW_TOKENS = 512
TEST_COUNT = 16817
LAYERS = (0, 1, 2, 9, 10, 11, 15, 16, 17, 21, 22, 23, 25, 26, 27)
CONFIGS = tuple(f'layer_{i:02d}' for i in LAYERS) + ('base',)
CONFIG_SETS = {'first': CONFIGS[:8], 'second': CONFIGS[8:]}
POLICY = 'single_layer_global_top32_keep_original_rope_compact_physical_cache_greedy512_v1'


def config_layer(name):
    if name not in CONFIGS:
        raise ValueError(f'Unknown config: {name}')
    return None if name == 'base' else int(name.split('_')[1])


def worker_configs(config_set, worker):
    if config_set not in CONFIG_SETS or worker not in (0, 1):
        raise ValueError('Select first/second and worker 0/1')
    return CONFIG_SETS[config_set][worker*4:worker*4+4]
