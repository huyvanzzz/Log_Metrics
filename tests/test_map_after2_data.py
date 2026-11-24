import json
import numpy as np
import pytest


def test_new_receipt_reuses_real_inputs_without_mutation(tmp_path):
    import torch
    from map_support import make_inputs
    from smolvlm_map_predictor_after2.data import prepare_inputs, read_receipt, PredictorDataset, PredictorCollator
    from smolvlm_map_predictor_after2.config import METHOD
    model, original = make_inputs(tmp_path, reorder=True)
    old = json.loads(original.read_text(encoding='utf-8'))
    protected = {p: __import__('pathlib').Path(p).read_bytes() for p in old['files']}
    original_bytes = original.read_bytes()
    receipt = prepare_inputs(tmp_path/'targets', old['cache_dir'], tmp_path/'prepared', model,
        tmp_path/'after2_inputs', expected_counts=old['counts'], checkpoint_sha256=old['identity']['checkpoint_sha256'])
    info = read_receipt(receipt)
    assert receipt.name == 'after2_receipt.json'
    assert info['identity']['method'] == METHOD
    assert 'question_tokens' not in json.dumps(info['identity'])
    assert info['identity']['target_hashes'] == old['identity']['target_hashes']
    assert original.read_bytes() == original_bytes
    assert all(__import__('pathlib').Path(p).read_bytes() == content for p, content in protected.items())
    for split in old['splits']:
        for key in ('offsets','target_offsets','features','lengths','target_rows'):
            np.testing.assert_array_equal(np.load(info['splits'][split][key]), np.load(old['splits'][split][key]))
    dataset = PredictorDataset(receipt, 'train')
    item = dataset[0]
    assert item['metadata']['row'] == 1
    assert isinstance(dataset._arrays['target'], np.memmap)
    assert not dataset._arrays['target'].flags.writeable
    batch = PredictorCollator(receipt, model.processor)([item, dataset[1]])
    assert batch['visual_features'].dtype == torch.float16
    assert batch['log_target'].shape == (2,384)
    target = __import__('pathlib').Path(info['target_files'][0]); target.unlink()
    assert read_receipt(receipt, include_targets=False)
    with pytest.raises(ValueError): read_receipt(receipt)
    assert PredictorCollator(receipt, model.processor)([PredictorDataset(receipt, 'test')[0]])['visual_features'].dtype == torch.float16


def test_rejects_old_method_receipt(tmp_path):
    from smolvlm_map_predictor_after2.data import read_receipt, PredictorDataset, PredictorCollator
    path = tmp_path/'receipt.json'
    path.write_text(json.dumps({'version':1,'files':{},'identity':{'predictor_query_policy':'question_mean'}}))
    with pytest.raises(ValueError, match='method'): read_receipt(path)
    with pytest.raises(ValueError, match='method'): PredictorDataset(path, 'test')
    with pytest.raises(ValueError, match='method'): PredictorCollator(path, None)
