"""Pinned base checkpoint, tokenizer and feature-cache contracts."""
MODEL_REPO = 'huyvanzzz/dinov2_base'
MODEL_REVISION = '84f863d54c0f1526b10b6c51d6496ecb29aea5cf'
CHECKPOINT_SHA256 = 'c8d640939cf9c1bb106814eeffbd8fe33c7b16e716bc6a5c8f5a23f811cc16b8'
TEXT_MODEL_ID = 'google/t5-efficient-mini'
TEXT_REVISION = 'af3816c2dbd386f7786f7757c536f0314084ff8b'
VISION_MODEL_ID = 'facebook/dinov2-small'
VISION_REVISION = 'ed25f3a31f01632728cabb09d1542f84ab7b0056'
REFERENCE_COMMIT = '11d1d431176d744b53ddeab61c7f669cc390f63a'
PREPARED_REPO = 'huyvanzzz/prepared_data'
PREPARED_REVISION = 'd2a6914f681dbb9148c4ac11e6fa92452010506e'
CACHE_REPO = 'huyvanzzz/cache_dinov2'
CACHE_REVISION = '051552392528900ce55bbaef1707b74fd5dd4419'
CACHE_FINGERPRINT = '39546b7a46379be3ac39e9b7b907d80a7291d0869adbdfc413649c83f47c79bc'
CAMERA_ORDER = ['Front', 'Front-Left', 'Front-Right', 'Back', 'Back-Left', 'Back-Right']
COUNTS = dict(train=341381, val=19785, test=16817)
TEST_COUNT = COUNTS['test']
QUERY_POLICY = 't5_decoder_start_id0_valid_mask1_native_cross_attention_mean_heads_samples_v1'
MEASUREMENT_FILES = ('config.py', 'model.py', 'collector.py', 'data.py')
BATCH_SIZE = 32
