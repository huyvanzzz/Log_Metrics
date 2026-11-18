import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

from dinov2_attention.model import CachedDinoT5, validate_checkpoint_metadata


def tiny_model():
    config = T5Config(vocab_size=32, d_model=16, d_ff=24, d_kv=4,
                      num_layers=2, num_decoder_layers=2, num_heads=2,
                      dropout_rate=0., decoder_start_token_id=0, pad_token_id=0,
                      eos_token_id=1)
    return CachedDinoT5(T5ForConditionalGeneration(config), num_views=2,
                        vision_dim=8, seq_len=4).eval()


def test_native_inputs_exact_spatial_camera_modal_order():
    model = tiny_model()
    visual = torch.randn(2, 2, 4, 8)
    ids = torch.tensor([[4, 5], [6, 0]])
    mask = ids.ne(0).long()
    embeds, combined, spatial, projected, text = model._native_inputs(ids, mask, visual)
    rows = torch.tensor([0, 0, 1, 1])
    cols = torch.tensor([0, 1, 0, 1])
    expected = visual + (model.row_embeddings(rows) + model.col_embeddings(cols))[None, None]
    torch.testing.assert_close(spatial, expected)
    enriched = expected + model.visual_fusion.camera_embeddings.weight[None, :, None]
    torch.testing.assert_close(projected, model.projector.proj(enriched.flatten(1, 2)))
    torch.testing.assert_close(embeds[:, :8], projected + model.modal_embeddings.weight[1])
    torch.testing.assert_close(embeds[:, 8:], text + model.modal_embeddings.weight[0])
    assert combined.tolist() == [[1] * 10, [1] * 9 + [0]]
    assert 'projector.proj.weight' in model.state_dict()
    assert 'visual_fusion.camera_embeddings.weight' in model.state_dict()


def test_geometry_errors_are_explicit():
    model = tiny_model()
    with pytest.raises(ValueError, match='geometry'):
        model._native_inputs(torch.ones(1, 2, dtype=torch.long),
                             torch.ones(1, 2, dtype=torch.long), torch.zeros(1, 2, 3, 8))


def test_metadata_rejects_pruning_and_missing_identity():
    with pytest.raises(ValueError, match='architecture_signature'):
        validate_checkpoint_metadata({})
    with pytest.raises(ValueError, match='fusion.name'):
        validate_checkpoint_metadata({'architecture_signature': {'fusion.name': 'mvpruner'}})


@pytest.fixture
def loader_fixture(monkeypatch, tmp_path):
    import dinov2_attention.model as module

    class Tokenizer:
        pad_token_id, eos_token_id = 0, 1
        def __init__(self):
            self.vocab = {'<pad>': 0, '</s>': 1}
        def get_vocab(self):
            return self.vocab
        def add_tokens(self, token):
            self.vocab[token] = 32100
        def convert_tokens_to_ids(self, token):
            return self.vocab[token]
        def __len__(self):
            return 32101

    config = T5Config(vocab_size=32128, d_model=384, d_ff=24, d_kv=64,
                      num_layers=4, num_decoder_layers=4, num_heads=8,
                      decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    monkeypatch.setattr(module.AutoConfig, 'from_pretrained', lambda *a, **k: config)
    monkeypatch.setattr(module.AutoTokenizer, 'from_pretrained', lambda *a, **k: Tokenizer())
    # Small native fixture isolates checkpoint policy without allocating full T5.
    native = tiny_model().text_model
    wrapper = module.CachedDinoT5(native)
    monkeypatch.setattr(module, 'T5ForConditionalGeneration', lambda *a, **k: tiny_model().text_model)
    payload = dict(model={k: v.clone() for k, v in wrapper.state_dict().items()}, metadata={
        'architecture_signature': module.expected_signature(),
        'tokenizer': {'less_than_token': {'token': '<', 'token_id': 32100, 'normal_token': True}},
        'epoch': 10})
    path = tmp_path / 'best.pt'
    return module, payload, path


def test_loader_strict_policy_eval_fp32_and_tied_weights(loader_fixture):
    module, payload, path = loader_fixture
    torch.save(payload, path)
    with pytest.raises(ValueError, match='SHA-256'):
        module.load_model(path, 'cpu')
    model, tokenizer = module.load_model(path, 'cpu', verify_checkpoint_hash=False)
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert model.text_model.shared.weight is model.text_model.lm_head.weight
    assert tokenizer.convert_tokens_to_ids('<') == 32100


@pytest.mark.parametrize('problem', ['tied', 'missing', 'tokenizer', 'pruned'])
def test_loader_rejects_incompatible_checkpoint(loader_fixture, problem):
    module, payload, path = loader_fixture
    if problem == 'tied':
        payload['model']['text_model.lm_head.weight'][0, 0] += 1
    elif problem == 'missing':
        payload['model'].pop('projector.proj.bias')
    elif problem == 'tokenizer':
        payload['metadata']['tokenizer']['less_than_token']['token_id'] = 32099
    else:
        payload['metadata']['architecture_signature']['fusion.name'] = 'mvpruner'
    torch.save(payload, path)
    with pytest.raises((ValueError, RuntimeError)):
        module.load_model(path, 'cpu', verify_checkpoint_hash=False)
