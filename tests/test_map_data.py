import json
import pytest


def test_receipt_rejects_changed_input(tmp_path):
    from smolvlm_map_predictor.data import read_receipt
    from smolvlm_attention.teacher_targets.loader import file_stat
    source = tmp_path / 'source'; source.write_text('ok')
    receipt = tmp_path / 'receipt.json'
    receipt.write_text(json.dumps({'version': 1, 'files': {str(source): file_stat(source)}}))
    assert read_receipt(receipt)['version'] == 1
    source.write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        read_receipt(receipt)


def test_receipt_can_skip_target_stats_for_test_evaluation(tmp_path):
    from smolvlm_map_predictor.data import read_receipt
    from smolvlm_attention.teacher_targets.loader import file_stat
    source = tmp_path / 'source'; source.write_text('ok')
    target = tmp_path / 'target'; target.write_text('ok')
    receipt = tmp_path / 'receipt.json'
    receipt.write_text(json.dumps({'version': 1, 'files': {str(source): file_stat(source), str(target): file_stat(target)}, 'target_files': [str(target)]}))
    target.unlink()
    assert read_receipt(receipt, include_targets=False)['version'] == 1
    with pytest.raises(ValueError): read_receipt(receipt)


def test_collator_preserves_native_prompt_and_fp16_cache(tmp_path):
    import numpy as np
    import torch
    from test_smolvlm_pruned_decoder import fixture
    from test_teacher_loader import loader_inputs
    from smolvlm_attention.teacher_targets.loader import file_stat
    from smolvlm_map_predictor.data import PredictorDataset, PredictorCollator
    model, _ = fixture()
    reader, processor, stores, _ = loader_inputs(tmp_path, model)
    source = reader.prepared / 'test.jsonl'
    raw = source.read_bytes().splitlines(keepends=True)
    offsets = np.cumsum([0] + [len(x) for x in raw], dtype=np.uint64)
    paths = {}
    for key, value in {'offsets': offsets, 'target_offsets': np.zeros(4, dtype=np.uint64),
                       'features': np.array([0, 1, 0], dtype=np.uint64), 'target_rows': np.zeros(3, dtype=np.uint64), 'lengths': np.ones(3, dtype=np.uint32)}.items():
        path = tmp_path / (key + '.npy'); np.save(path, value); paths[key] = str(path)
    receipt = tmp_path / 'map.json'
    receipt.write_text(json.dumps(dict(version=1, files={str(source): file_stat(source)}, target_files=[],
        geometry=model.geometry, max_length=2048, cache_dir=str(reader.directory), shape=list(reader.array.shape),
        counts={'test': 3}, splits={'test': dict(source=str(source), target=None, **paths)})))
    dataset = PredictorDataset(receipt, 'test')
    batch = PredictorCollator(receipt, processor)([dataset[0], dataset[1]])
    assert batch['visual_features'].dtype == torch.float16
    np.testing.assert_array_equal(batch['visual_features'][0].numpy(), reader.array[0])
    assert batch['image_positions'].shape == (2, 24)
    assert batch['question_mask'].any(1).all()
    assert 'log_target' not in batch



def test_dataset_reads_target_in_prepared_order_after_join(tmp_path):
    import numpy as np
    from smolvlm_map_predictor.data import PredictorDataset
    source = tmp_path/'source.jsonl'; source.write_text('{"sample_id":"a"}\n{"sample_id":"b"}\n')
    target = tmp_path/'target'; target.mkdir()
    lines = [b'{"sample_id":"b"}\n', b'{"sample_id":"a"}\n']
    (target/'samples.jsonl').write_bytes(b''.join(lines))
    np.save(target/'log_probs.npy', np.array([[10, 11], [20, 21]], dtype=np.float32))
    arrays = {'offsets': [0, 18, 36], 'target_offsets': [[len(lines[0]), sum(map(len,lines))], [0, len(lines[0])]],
              'features': [0, 1], 'target_rows': [1, 0], 'lengths': [2, 2]}
    paths = {}
    for name, data in arrays.items():
        path = tmp_path/(name+'.npy'); np.save(path, np.array(data, dtype=np.uint64)); paths[name] = str(path)
    receipt = tmp_path/'receipt.json'
    receipt.write_text(json.dumps(dict(version=1, files={}, counts={'train': 2},
        splits={'train': dict(source=str(source), target=str(target), **paths)})))
    dataset = PredictorDataset(receipt, 'train')
    assert dataset[0]['metadata']['sample_id'] == 'a'
    np.testing.assert_array_equal(dataset[0]['log_target'], [20, 21])


def test_prepare_real_teacher_fixture_and_reordered_targets(tmp_path):
    from map_support import make_inputs
    from smolvlm_map_predictor.data import PredictorDataset, PredictorCollator
    model, receipt = make_inputs(tmp_path, reorder=True)
    dataset = PredictorDataset(receipt, 'train')
    batch = PredictorCollator(receipt, model.processor)([dataset[0], dataset[1]])
    assert batch['log_target'].shape == (2, 384)
    assert len(dataset.prompt_lengths) == 3
    assert dataset[0]['metadata']['sample_id'] == dataset[0]['row']['sample_id']
    assert dataset[0]['metadata']['row'] == 1


@pytest.mark.parametrize('problem', ['layer','checkpoint','camera','query','question_hash','duplicate','checksum'])
def test_parent_rejects_misaligned_or_corrupted_teacher(tmp_path,problem):
    from map_support import make_inputs
    from smolvlm_map_predictor.data import prepare_inputs
    from smolvlm_attention.sample_store import digest
    model, receipt = make_inputs(tmp_path)
    description = json.loads(receipt.read_text(encoding='utf-8'))
    directory = tmp_path/'targets/train'
    path = directory/'manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if problem == 'layer': manifest['identity']['protocol']['score_layer_index'] = 21
    elif problem == 'checkpoint': manifest['identity']['checkpoint_sha256'] = 'f'*64
    elif problem == 'camera': manifest['identity']['model']['camera_order'].reverse()
    elif problem == 'query': manifest['identity']['protocol']['query_policy'] = 'question_mean'
    else:
        samples = directory/'samples.jsonl'
        rows = [json.loads(line) for line in samples.read_text(encoding='utf-8').splitlines()]
        if problem == 'question_hash': rows[0]['question_sha256'] = 'f'*64
        elif problem == 'duplicate': rows[1]['sample_id'] = rows[0]['sample_id']
        else: rows[0]['prompt_length'] += 1
        samples.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
        if problem != 'checksum': manifest['checksums']['samples.jsonl'] = digest(samples)
    path.write_text(json.dumps(manifest),encoding='utf-8')
    with pytest.raises(ValueError):
        prepare_inputs(tmp_path/'targets',tmp_path/'cache',tmp_path/'prepared',model,tmp_path/'rejected',
                       expected_counts={'train':3,'val':3,'test':3},checkpoint_sha256=description['identity']['checkpoint_sha256'])


def test_answers_do_not_enter_predictor_inputs_or_loss_and_bad_batch_is_rejected(tmp_path):
    import copy
    import torch
    import numpy as np
    from map_support import make_inputs
    from smolvlm_map_predictor.data import PredictorDataset,PredictorCollator
    from smolvlm_map_predictor.model import frozen_features,kl_per_sample
    from smolvlm_map_predictor.training import make_predictor
    from test_map_training import tiny_recipe
    model,receipt = make_inputs(tmp_path)
    data = PredictorDataset(receipt,'train'); collate = PredictorCollator(receipt,model.processor)
    original = data[0]; changed = copy.deepcopy(original)
    changed['row']['answer'] = 'A completely different answer that must never be tokenized.'
    a,b = collate([original]),collate([changed])
    for key in ('input_ids','attention_mask','question_mask','visual_features','image_positions','log_target'):
        assert torch.equal(a[key],b[key])
    predictor = make_predictor(model,tiny_recipe())
    torch.testing.assert_close(kl_per_sample(predictor(**frozen_features(model,a)),a['log_target']),
                               kl_per_sample(predictor(**frozen_features(model,b)),b['log_target']),rtol=0,atol=0)
    changed['log_target'][0] = np.nan
    with pytest.raises(ValueError,match='Non-finite teacher'):
        collate([changed])
