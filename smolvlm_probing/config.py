"""Versioned measurement policies, independent of figure styling."""
MODEL_REPO = 'huyvanzzz/smolvlm256M_base'
MODEL_REVISION = 'f9d4e9a4d704078f9e7f39f061d13602cf251c78'
PREPARED_REVISION = 'd2a6914f681dbb9148c4ac11e6fa92452010506e'
DATASET_FINGERPRINT = '563561514fb257360a40a3eb9ba7846b6e8829411ae8e42b68a2e723cd7a9156'
IMAGE_REPO = 'minhdang0901/drivelm-nuscenes-v1-0'
IMAGE_REVISION = '6c4336a5279eaba3f3dfc5b525cc6ba57d55d61a'
CAMERA_ORDER = ['Front', 'Front-Left', 'Front-Right', 'Back', 'Back-Left', 'Back-Right']
RANKING_LAYERS = (1, 7, 13, 19, 25)
LAYER_GROUPS = tuple(tuple(range(i, i + 5)) for i in range(0, 30, 5))
MAX_NEW_TOKENS = 64
K = 32
SEED = 42
BATCH_SIZE = 16
POLICY = 'base_original_generated_positions_including_self_geomean_exact_literal_v1'
