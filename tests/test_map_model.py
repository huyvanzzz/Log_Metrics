import pytest
import torch


def test_predictor_distribution_gradient_and_ties():
    from transformers import LlamaConfig
    from smolvlm_map_predictor.model import MapPredictor, kl_per_sample, topk_indices
    config = LlamaConfig(hidden_size=8, num_attention_heads=2)
    model = MapPredictor(config, hidden_dim=8, projection_dim=4, visual_tokens=3).double()
    visual = torch.randn(2, 3, 8, dtype=torch.double, requires_grad=True)
    question = torch.randn(2, 2, 8, dtype=torch.double, requires_grad=True)
    mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.bool)
    vp = torch.tensor([[0, 1, 2], [1, 2, 3]])
    qp = torch.tensor([[4, 5], [4, 0]])
    output = model(visual, question, mask, vp, qp)
    torch.testing.assert_close(output.exp().sum(-1), torch.ones(2, dtype=torch.double))
    assert torch.autograd.gradcheck(lambda a, b: model(a, b, mask, vp, qp), (visual, question))
    torch.testing.assert_close(kl_per_sample(output, output), torch.zeros(2, dtype=torch.double))
    assert topk_indices(torch.tensor([[3., 1., 4., 2.]]), 3).tolist() == [[0, 2, 3]]
    assert topk_indices(torch.tensor([[1., 3., 3., 2.]]), 1).tolist() == [[1]]
    with pytest.raises(ValueError):
        model(visual, question, torch.zeros_like(mask), vp, qp)


def test_frozen_features_uses_original_positions_and_all_visual_tokens():
    from test_smolvlm_pruned_decoder import fixture
    from smolvlm_map_predictor.model import frozen_features, MapPredictor
    model, batch = fixture()
    features = frozen_features(model, batch)
    assert features['visual'].shape == (2, 24, model.geometry['text_dim'])
    assert not features['visual'].requires_grad
    assert torch.equal(features['visual_positions'], batch['source_position_ids'].gather(1, batch['image_positions']))
    for i in range(2):
        assert torch.equal(features['question_positions'][i, features['question_mask'][i]], batch['source_position_ids'][i, batch['question_mask'][i]])
    predictor = MapPredictor(model.native.config.text_config, hidden_dim=8, projection_dim=4, visual_tokens=24)
    predictor(**features).sum().backward()
    assert all(p.grad is None for p in model.parameters())


def test_rope_softmax_before_mean_and_kl_match_independent_reference():
    import math
    from transformers import LlamaConfig
    from smolvlm_map_predictor.model import MapPredictor, kl_per_sample
    torch.manual_seed(9)
    cfg = LlamaConfig(hidden_size=8, num_attention_heads=2, rope_theta=12345.)
    model = MapPredictor(cfg, hidden_dim=8, projection_dim=4, visual_tokens=3)
    visual, question = torch.randn(2,3,8), torch.randn(2,2,8)
    vp = torch.tensor([[2,6,9],[1,3,12]])
    qp = torch.tensor([[11,14],[13,0]])
    mask = torch.tensor([[True,True],[True,False]])
    q, k = model.Wq(model.question_mlp(question)), model.Wk(model.visual_mlp(visual))
    def manual_rope(x, positions):
        frequencies = cfg.rope_theta ** (-torch.arange(0,4,2).float()/4)
        angle = positions.float().unsqueeze(-1)*frequencies
        real, imag = x[...,:2], x[...,2:]
        return torch.cat([real*angle.cos()-imag*angle.sin(), real*angle.sin()+imag*angle.cos()], -1)
    logits = manual_rope(q,qp) @ manual_rope(k,vp).transpose(-1,-2)/math.sqrt(4)
    expected = torch.stack([logits[b,mask[b]].softmax(-1).mean(0) for b in range(2)])
    actual = model(visual, question, mask, vp, qp)
    torch.testing.assert_close(actual.exp(), expected)
    target = torch.tensor([[.1,.2,.7],[.5,.4,.1]])
    reference_kl = (target*(target.log()-expected.log())).sum(-1)
    torch.testing.assert_close(kl_per_sample(actual,target.log()), reference_kl)
    # The unqueried padded position must not contribute to the mixture.
    question[1,1] = 1e4
    torch.testing.assert_close(model(visual,question,mask,vp,qp), actual)


def test_production_parameter_count_and_extreme_fp32_logits():
    from transformers import LlamaConfig
    from smolvlm_map_predictor.model import MapPredictor
    model = MapPredictor(LlamaConfig(hidden_size=576,num_attention_heads=9))
    assert sum(p.numel() for p in model.parameters()) == 197120
    visual, question = torch.randn(1,384,576)*1e3, torch.randn(1,2,576)*1e3
    output = model(visual,question,torch.ones(1,2,dtype=torch.bool),
                   torch.arange(384)[None], torch.tensor([[390,400]]))
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.logsumexp(-1), torch.zeros(1), atol=1e-5, rtol=0)
