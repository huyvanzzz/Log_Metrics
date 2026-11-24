import json


def test_diagnostic_runs_real_stress_update_and_restores_rng(tmp_path):
    import copy
    import torch
    from map_support import make_inputs
    from smolvlm_map_predictor.config import load_config
    from smolvlm_map_predictor.diagnostics import diagnose
    model, receipt = make_inputs(tmp_path)
    cfg = copy.deepcopy(load_config())
    cfg['training'].update(world_size=1, batch_size=2, num_workers=0, pin_memory=False)
    state = torch.get_rng_state().clone()
    result = diagnose(cfg, receipt, None, tmp_path/'out', cpu=True, backbone=model)
    assert torch.equal(state, torch.get_rng_state())
    assert result['gradients_finite'] and result['optimizer_state_count'] > 0
    assert result['optimizer_parameter_ids_unique'] and result['decoder_calls'] == 0
    assert result['batch_size'] == 2
    assert json.loads((tmp_path/'out'/'diagnose_rank_0.json').read_text()) == result


def test_diagnostic_uses_spawn_prefetch_workers(tmp_path):
    import copy
    from map_support import make_inputs
    from smolvlm_map_predictor.config import load_config
    from smolvlm_map_predictor.diagnostics import diagnose
    model, receipt = make_inputs(tmp_path)
    cfg = copy.deepcopy(load_config())
    cfg['training'].update(world_size=1, batch_size=2, num_workers=2, prefetch_factor=1, pin_memory=False)
    result = diagnose(cfg, receipt, None, tmp_path/'out', cpu=True, backbone=model)
    assert result['loader_num_workers'] == 2
    assert result['loader_prefetch_factor'] == 1
