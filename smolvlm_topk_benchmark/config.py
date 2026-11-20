"""Fixed experiment: fifteen single-layer rankings and one unpruned baseline."""
BATCH_SIZE = 64
K = 32
MAX_NEW_TOKENS = 512
TEST_COUNT = 16817
LAYERS = (0, 1, 2, 9, 10, 11, 15, 16, 17, 21, 22, 23, 25, 26, 27)
CONFIGS = tuple(f'layer_{i:02d}' for i in LAYERS) + ('base',)
CONFIG_SETS = {'layers': CONFIGS[:-1], 'base': ('base',)}
POLICY = 'single_layer_global_top32_keep_original_rope_compact_physical_cache_greedy512_v1'


def config_layer(name):
    if name not in CONFIGS:
        raise ValueError(f'Unknown config: {name}')
    return None if name == 'base' else int(name.split('_')[1])


def worker_configs(config_set, worker):
    if config_set not in CONFIG_SETS or worker not in (0, 1):
        raise ValueError('Select layers/base and worker 0/1')
    if config_set == 'base':
        return ('base',) if worker == 0 else ()
    return CONFIG_SETS[config_set][:7] if worker == 0 else CONFIG_SETS[config_set][7:]


def active_workers(config_set):
    if config_set not in CONFIG_SETS:
        raise ValueError('Select layers/base')
    return (0, 1) if config_set == 'layers' else (0,)
