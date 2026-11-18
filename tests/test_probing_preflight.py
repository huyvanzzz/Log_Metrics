import pytest
import torch

from test_attention_collector import fixture


def run(model, batch):
    from smolvlm_probing.preflight import numerical_preflight
    return numerical_preflight(model, batch, ranking_layers=(0, 1), layer_groups=((0,), (1,)))


def test_preflight_preserves_middle_specials_and_uses_native_eos(monkeypatch):
    model, batch, _, _ = fixture(questions=('What?', 'What is visible? A road. A car.'))
    tokens = model.processor.tokenizer.encode('A car.', add_special_tokens=False)
    special = model.processor.tokenizer.bos_token_id
    native_eos = model.processor.tokenizer.eos_token_id + 20
    model.native.generation_config.eos_token_id = [native_eos, native_eos + 1]
    emitted = [tokens[0], special, tokens[1], native_eos]
    calls = []
    def generated(**kwargs):
        calls.append(kwargs['generation_config'])
        return torch.tensor([emitted, emitted])
    monkeypatch.setattr(model, 'generate_from_features', generated)
    import smolvlm_probing.measurement as measurement
    original = measurement.collect_baseline
    replayed = []
    def collect(*args, **kwargs):
        replayed.append(args[2])
        return original(*args, **kwargs)
    monkeypatch.setattr(measurement, 'collect_baseline', collect)
    receipt = run(model, batch)
    assert replayed[0] == [emitted[:3], emitted[:3]]
    assert replayed[1] == [emitted[:3]]
    assert calls[0].max_new_tokens == 4 and calls[0].eos_token_id == [native_eos, native_eos+1]
    assert receipt['samples'] == 2 and receipt['fallback_samples'] == []
    for key in ('native_attention_parity', 'native_predecessor_scoring_parity',
                'fused_zero_ablation_parity', 'cached_decode_attention_parity', 'cached_decode_scoring_parity'):
        assert receipt[key] is True


@pytest.mark.parametrize('kind', ['empty', 'invalid'])
def test_preflight_fallback_is_explicit_and_never_replays_reserved_image(monkeypatch, kind):
    model, batch, _, _ = fixture()
    eos = model.native.generation_config.eos_token_id
    emitted = [eos] if kind == 'empty' else [model.native.config.image_token_id, eos]
    monkeypatch.setattr(model, 'generate_from_features', lambda **kwargs: torch.tensor([emitted]))
    receipt = run(model, batch)
    assert receipt['fallback_samples'] == [{'sample_index': 0, 'reason': 'empty' if kind == 'empty' else 'invalid_multimodal_generation'}]
    assert receipt['fallback_policy'] == 'regular_A_car_tokens_for_numeric_check_only'


def test_preflight_actual_tiny_greedy_runs_without_downloads():
    model, batch, _, _ = fixture()
    receipt = run(model, batch)
    assert receipt['samples'] == 1
    assert receipt['cached_decode_attention_parity'] is True
