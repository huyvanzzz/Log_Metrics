import pytest
import torch

from test_smolvlm_model import tiny_components
from smolvlm_inference.model import CachedSmolVLM, CAMERA_ORDER
from smolvlm_inference.text import CachedBatchCollator


def fixture(order=CAMERA_ORDER, questions=('What is visible?',)):
    native, processor = tiny_components()
    model = CachedSmolVLM(native, processor, prompt_camera_order=order)
    items = [dict(question=q, visual_features=torch.randn(6, 16, 8)) for q in questions]
    collator = CachedBatchCollator(processor, image_seq_len=4, max_length=512,
                                  max_new_tokens=1, prompt_camera_order=order)
    return model, collator(items), items, collator


@pytest.mark.parametrize('reverse', [False, True])
def test_all_layer_reference_parity(reverse):
    from smolvlm_attention.collector import collect_last_query_attention, reference_last_query_attention
    model, batch, _, _ = fixture(list(reversed(CAMERA_ORDER)) if reverse else CAMERA_ORDER)
    actual = collect_last_query_attention(model, batch)
    expected = reference_last_query_attention(model, batch)
    assert actual.shape == (1, 2, 24)
    assert actual.device.type == 'cpu'
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-4)
    assert (actual >= 0).all() and (actual.sum(-1) <= 1).all()


def test_collector_runs_in_inference_mode():
    from smolvlm_attention.collector import collect_last_query_attention
    model, batch, _, _ = fixture()
    observed = []
    handle = model.native.model.text_model.layers[0].self_attn.register_forward_pre_hook(
        lambda *args: observed.append(torch.is_inference_mode_enabled()))
    try:
        collect_last_query_attention(model, batch)
    finally:
        handle.remove()
    assert observed == [True]


def test_left_padding_matches_singles_and_ignores_pad_identity():
    from smolvlm_attention.collector import collect_last_query_attention, reference_last_query_attention
    model, batch, items, collator = fixture(questions=('What?', 'What is visible? A road. A car.'))
    actual = collect_last_query_attention(model, batch)
    torch.testing.assert_close(actual, reference_last_query_attention(model, batch), atol=1e-6, rtol=1e-4)
    for i, item in enumerate(items):
        torch.testing.assert_close(actual[i:i+1], collect_last_query_attention(model, collator([item])), atol=1e-6, rtol=1e-4)
    batch['input_ids'][batch['attention_mask'] == 0] = model.processor.tokenizer.unk_token_id
    torch.testing.assert_close(actual, collect_last_query_attention(model, batch), atol=1e-6, rtol=1e-4)


def test_last_query_uses_mask_positions_with_right_padding():
    from smolvlm_attention.collector import collect_last_query_attention, reference_last_query_attention
    model, batch, _, _ = fixture()
    expected = collect_last_query_attention(model, batch)
    batch['input_ids'] = torch.nn.functional.pad(batch['input_ids'], (0, 3), value=model.processor.tokenizer.pad_token_id)
    batch['attention_mask'] = torch.nn.functional.pad(batch['attention_mask'], (0, 3), value=0)
    torch.testing.assert_close(collect_last_query_attention(model, batch), expected, atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(reference_last_query_attention(model, batch), expected, atol=1e-6, rtol=1e-4)


def test_native_path_once_and_cleanup(monkeypatch):
    from smolvlm_attention.collector import collect_last_query_attention
    model, batch, _, _ = fixture()
    calls = []
    handle = model.native.model.connector.register_forward_hook(lambda *args: calls.append('connector'))
    merger = model.native.model.inputs_merger
    monkeypatch.setattr(model.native.model, 'inputs_merger', lambda **kw: (calls.append('merger'), merger(**kw))[1])
    def forbidden(*args, **kwargs):
        raise AssertionError('vision and lm_head must not run')
    monkeypatch.setattr(model.native.model.vision_model, 'forward', forbidden)
    monkeypatch.setattr(model.native.lm_head, 'forward', forbidden)
    collect_last_query_attention(model, batch)
    assert calls == ['connector', 'merger']
    layers = model.native.model.text_model.layers
    assert all(not layer.self_attn._forward_pre_hooks for layer in layers)
    monkeypatch.setattr(layers[0].self_attn, 'forward', forbidden)
    with pytest.raises(AssertionError):
        collect_last_query_attention(model, batch)
    assert all(not layer.self_attn._forward_pre_hooks for layer in layers)
    handle.remove()


@pytest.mark.parametrize('invalid', ['empty', 'images', 'train', 'dtype'])
def test_invalid_input(invalid):
    from smolvlm_attention.collector import collect_last_query_attention
    model, batch, _, _ = fixture()
    if invalid == 'empty':
        batch['attention_mask'].zero_()
    elif invalid == 'images':
        batch['input_ids'][batch['input_ids'] == model.native.config.image_token_id] = 0
    elif invalid == 'train':
        model.train()
    else:
        model.double()
    with pytest.raises(ValueError):
        collect_last_query_attention(model, batch)


def test_native_pixel_shuffle_preserves_row_major_spatial_blocks():
    native, _ = tiny_components()
    coordinates = torch.arange(32 * 32).reshape(1, 32 * 32, 1).float()
    shuffled = native.model.connector.pixel_shuffle(coordinates, scale_factor=4)
    assert shuffled.shape == (1, 64, 16)
    for row in range(8):
        for col in range(8):
            expected = torch.tensor([(row*4+r)*32 + col*4+c for r in range(4) for c in range(4)])
            # The native channel order is immaterial; token membership proves orientation.
            assert torch.equal(shuffled[0, row*8+col].sort().values, expected.float())
