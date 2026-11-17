"""Pinned public artifacts used by the Kaggle attention notebook."""
from smolvlm_inference.model import CAMERA_ORDER

MODELS = {
    'base': {
        'repo': 'huyvanzzz/smolvlm256M_base',
        'revision': 'f9d4e9a4d704078f9e7f39f061d13602cf251c78',
        'camera_order': list(CAMERA_ORDER),
        'gpu': 0,
    },
    'reverse': {
        'repo': 'huyvanzzz/256M_reverse',
        'revision': 'd13e6d97849b85fe6367ad75a027b6b3696788b4',
        'camera_order': list(reversed(CAMERA_ORDER)),
        'gpu': 1,
    },
}
PREPARED_REPO = 'huyvanzzz/prepared_data'
PREPARED_REVISION = 'd2a6914f681dbb9148c4ac11e6fa92452010506e'
CACHE_DATASET = 'huyhuy12334/cache-smolvlm/versions/1'
CACHE_FINGERPRINT = '0632954bab4d7a8e5af5934f3476610a0547db8f3083495c1207c3da87b15585'
TEST_COUNT = 16817
QUERY_POLICY = 'last_valid_prompt_position_softmax_over_all_valid_keys_v1'
