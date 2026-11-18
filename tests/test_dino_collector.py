import pytest
import torch

from test_dino_model import tiny_model
from dinov2_attention.collector import collect_pad_cross_attention, pad_cross_forward


def batch():
    return dict(input_ids=torch.tensor([[4, 5, 6], [7, 0, 0]]),
                attention_mask=torch.tensor([[1, 1, 1], [1, 0, 0]]),
                visual_features=torch.randn(2, 2, 4, 8))


def test_native_cross_attention_and_first_generated_token():
    torch.manual_seed(42)
    model = tiny_model()
    data = batch()
    attention, logits = pad_cross_forward(model, data)
    assert attention.shape == (2, 2, 8)
    assert attention.dtype == torch.float32 and attention.device.type == 'cpu'
    with torch.inference_mode():
        embeds, mask, *_ = model._native_inputs(**data)
        oracle = model.text_model(inputs_embeds=embeds, attention_mask=mask,
                                  decoder_input_ids=torch.zeros(2, 1, dtype=torch.long),
                                  decoder_attention_mask=torch.ones(2, 1, dtype=torch.long),
                                  output_attentions=True, use_cache=False, return_dict=True)
        expected = torch.stack([a[:, :, 0, :8].mean(1) for a in oracle.cross_attentions], 1)
        torch.testing.assert_close(attention, expected)
        generated = model.text_model.generate(inputs_embeds=embeds, attention_mask=mask,
                                              max_new_tokens=1, do_sample=False, num_beams=1)
        assert torch.equal(logits.argmax(-1), generated[:, 1])
        assert (attention.sum(-1) <= 1.00001).all()
        for a in oracle.cross_attentions:
            torch.testing.assert_close(a.sum(-1), torch.ones(2, 2, 1))
            assert not a[1, :, :, -2:].any()


def test_batch_matches_individual_and_encoder_does_not_return_attentions():
    model = tiny_model()
    data = batch()
    calls = []
    hook = model.text_model.encoder.register_forward_hook(lambda m, a, o: calls.append(o.attentions))
    together = collect_pad_cross_attention(model, data)
    separate = torch.cat([collect_pad_cross_attention(model, {k: v[i:i+1] for k, v in data.items()})
                          for i in range(2)])
    hook.remove()
    torch.testing.assert_close(together, separate, atol=1e-6, rtol=1e-5)
    assert calls == [None, None, None]


def test_collector_rejects_training_and_non_fp32():
    model = tiny_model().train()
    with pytest.raises(ValueError, match='eval'):
        collect_pad_cross_attention(model, batch())
    model.eval().double()
    with pytest.raises(ValueError, match='FP32'):
        collect_pad_cross_attention(model, batch())
