import copy

import pytest
import torch

from smolvlm_inference import model as module
from smolvlm_inference.text import CachedBatchCollator
from test_smolvlm_model import checkpoint, tiny_components


def test_reverse_checkpoint_retains_canonical_cache_identity(tmp_path, monkeypatch):
    _, original, data, path = checkpoint(tmp_path, monkeypatch)
    order = list(reversed(module.CAMERA_ORDER))
    original.prompt_camera_order = order
    data['architecture_signature'] = original.architecture_signature
    torch.save(data, path)
    loaded = module.load_model(path, device='cpu', expected_prompt_camera_order=order)
    assert loaded.prompt_camera_order == order
    assert loaded.architecture_signature['processor']['camera_order'] == module.CAMERA_ORDER
    assert loaded.architecture_signature == data['architecture_signature']
    with pytest.raises(ValueError, match='order'):
        module.load_model(path, device='cpu', expected_prompt_camera_order=module.CAMERA_ORDER)


def test_reverse_collator_reorders_features_and_prompt_together():
    native, processor = tiny_components()
    order = list(reversed(module.CAMERA_ORDER))
    model = module.CachedSmolVLM(native, processor, prompt_camera_order=order)
    item = dict(question='What is visible?', visual_features=torch.stack([
        torch.full((16, 8), float(i)) for i in range(6)]))
    batch = CachedBatchCollator(processor, image_seq_len=4, max_length=512,
                               max_new_tokens=0, prompt_camera_order=order)([item])
    assert batch['visual_features'][0, :, 0, 0].tolist() == [5, 4, 3, 2, 1, 0]
    prompt = processor.tokenizer.decode(batch['input_ids'][0])
    assert prompt.index('Back-Right:') < prompt.index('Front:')
    assert model.visual_token_len == 24


@pytest.mark.parametrize('order', [[], ['Front'] * 6,
    ['Front-Left', 'Front', 'Front-Right', 'Back', 'Back-Left', 'Back-Right']])
def test_other_camera_permutations_rejected(order):
    with pytest.raises(ValueError):
        module.validate_prompt_camera_order(order)
