import json
from pathlib import Path

import numpy as np
import pytest

from smolvlm_topk_benchmark.config import CONFIG_SETS, BATCH_SIZE, config_layer, worker_configs
from smolvlm_topk_benchmark.state import (initialize_run, read_run, input_rows, save_batch,
    load_batches, finalize_config, validate_result, write_selection, load_selection)


def samples(count=67):
    return [dict(row_index=i, sample_id=f'test-{i}', eval_id=i, answer='A car.',
                 input_ids=[9]*384+[3], image_token_id=9, attention_row=i) for i in range(count)]


def run_fixture(tmp_path, count=67, policy='fixture'):
    path = initialize_run(tmp_path, {'policy': policy, 'batch_size': BATCH_SIZE}, samples(count),
                          {'checkpoint': 'fixture'})
    return read_run(path)


def predictions(rows):
    return [dict(sample_id=r['sample_id'], eval_id=r['eval_id'], reference=r['answer'],
        generated_ids=[4, 5], generated_token_count=2, stop_reason='eos', raw_prediction=' A car. ',
        prediction='A car.') for r in rows]


def test_registry_and_same_dataset_for_every_config():
    assert BATCH_SIZE == 64
    assert [config_layer(c) for c in CONFIG_SETS['layers']] == [0,1,2,9,10,11,15,16,17,21,22,23,25,26,27]
    assert CONFIG_SETS['base'] == ('base',)
    assert len(worker_configs('layers', 0)) == 7
    assert len(worker_configs('layers', 1)) == 8
    assert worker_configs('layers', 0) + worker_configs('layers', 1) == CONFIG_SETS['layers']
    assert worker_configs('base', 0) == ('base',)
    assert worker_configs('base', 1) == ()


def test_initialize_rejects_mismatch_before_overwrite_and_part_is_not_identity(tmp_path):
    run = run_fixture(tmp_path)
    before = (tmp_path / 'benchmark_run.json').read_bytes()
    initialize_run(tmp_path, {'policy': 'fixture', 'batch_size': BATCH_SIZE}, samples(), {'checkpoint': 'fixture'})
    assert before == (tmp_path / 'benchmark_run.json').read_bytes()
    with pytest.raises(ValueError, match='incompatible'):
        run_fixture(tmp_path, policy='changed')
    assert before == (tmp_path / 'benchmark_run.json').read_bytes()
    with pytest.raises(ValueError):
        initialize_run(tmp_path, run['identity'], samples(), {}, mode='measure')


def test_batch_atomic_resume_corruption_and_final_batch(tmp_path):
    run = run_fixture(tmp_path)
    rows = input_rows(run)
    cfg = CONFIG_SETS['layers'][0]
    write_selection(run, cfg, np.tile(np.arange(32,dtype=np.uint16), (67,1)),
                    np.tile(np.arange(32,dtype=np.int32), (67,1)))
    save_batch(run, cfg, 0, predictions(rows[:64]))
    assert set(load_batches(run, cfg)) == {0}
    with pytest.raises(ValueError, match='incomplete'):
        finalize_config(run, cfg)
    with pytest.raises(ValueError):
        save_batch(run, cfg, 64, predictions(rows[64:66]))
    save_batch(run, cfg, 64, predictions(rows[64:]))
    # Idempotent publication must not add duplicates.
    save_batch(run, cfg, 64, predictions(rows[64:]))
    result = finalize_config(run, cfg)
    metadata, records = validate_result(result, run)
    assert len(records) == 67 and metadata['config'] == cfg
    assert records[-1]['eval_id'] == 66
    batch = tmp_path / 'configs' / cfg / 'batches' / '000000.json'
    payload = json.loads(batch.read_text())
    payload['records'][0]['prediction'] = 'tamper'
    batch.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='checksum'):
        load_batches(run, cfg)


def test_selection_hash_and_config_identity(tmp_path):
    run = run_fixture(tmp_path, 2)
    cfg = CONFIG_SETS['layers'][0]
    indices = np.tile(np.arange(32,dtype=np.uint16),(2,1))
    positions = np.tile(np.arange(32,dtype=np.int32),(2,1))
    write_selection(run, cfg, indices, positions)
    a,b = load_selection(run, cfg)
    np.testing.assert_array_equal(a,indices)
    with pytest.raises(ValueError):
        write_selection(run, cfg, indices[:,::-1],positions)
    path = tmp_path / 'configs' / cfg / 'selection.npy'
    x=np.load(path,mmap_mode='r+'); x[0,0]=1; x.flush(); del x
    with pytest.raises(ValueError, match='hash'):
        load_selection(run,cfg)


def test_resume_keeps_identity_when_output_directory_moves(tmp_path):
    old=tmp_path/'old'; new=tmp_path/'new'
    run=run_fixture(old,2)
    import shutil
    shutil.copytree(old,new)
    moved=read_run(new/'benchmark_run.json')
    assert moved['fingerprint']==run['fingerprint']
    assert len(input_rows(moved))==2


def test_cannot_publish_wrong_k_or_mislabeled_result(tmp_path):
    run=run_fixture(tmp_path,2); cfg=CONFIG_SETS['layers'][0]
    with pytest.raises(ValueError):
        write_selection(run,cfg,np.tile(np.arange(2,dtype=np.uint16),(2,1)),np.tile(np.arange(2,dtype=np.int32),(2,1)))
    write_selection(run,'base')
    save_batch(run,'base',0,predictions(input_rows(run)))
    result=finalize_config(run,'base')
    import shutil
    renamed=tmp_path/'results'/cfg
    shutil.copytree(result,renamed)
    with pytest.raises(ValueError,match='config'):
        validate_result(renamed,run)
