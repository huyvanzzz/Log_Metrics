import json
import pytest


def test_production_config_and_recipe():
    from smolvlm_map_predictor_after2.config import load_config, recipe, METHOD
    from smolvlm_map_predictor.config import load_config as baseline
    cfg = load_config()
    assert cfg['method'] == METHOD
    assert cfg['inputs'] == baseline()['inputs']
    assert cfg['training'] == baseline()['training']
    assert recipe(cfg)['backend'] == METHOD['id']
    assert recipe(cfg)['method'] == METHOD


@pytest.mark.parametrize('key,value', [('method', {}), ('inputs', {}), ('architecture', {})])
def test_rejects_changed_production_identity(tmp_path, key, value):
    from smolvlm_map_predictor_after2.config import load_config
    cfg = load_config(); cfg[key] = value
    path = tmp_path/'config.json'; path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError): load_config(path)
