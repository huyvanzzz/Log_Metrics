import json
import shutil

import numpy as np
import pytest

from smolvlm_topk_deferred.state import (CONFIGS, initialize_run, read_run, input_rows,
    write_selection, load_selection, save_batch, load_batches, finalize_config, validate_result)


def fixture(path, count=67):
    rows = [dict(row_index=i, sample_id=f's-{i}', eval_id=i, answer='Car.',
                 input_ids=[9]*384+[3], image_token_id=9) for i in range(count)]
    run = read_run(initialize_run(path, {'batch_size':64,'experiment':'deferred'}, rows, {}))
    indices = np.tile(np.arange(32,dtype=np.uint16),(count,1))
    write_selection(run,indices)
    return run,rows,indices


def predictions(rows):
    return [dict(sample_id=r['sample_id'],eval_id=r['eval_id'],reference=r['answer'],
        generated_ids=[3],generated_token_count=1,stop_reason='eos',raw_prediction=' Car. ',
        prediction='Car.') for r in rows]


def test_shared_selection_and_independent_resume_tail(tmp_path):
    run,rows,indices=fixture(tmp_path)
    np.testing.assert_array_equal(load_selection(run),indices)
    assert not load_selection(run).flags.writeable
    for name in CONFIGS:
        save_batch(run,name,0,predictions(rows[:64]))
    save_batch(run,CONFIGS[0],64,predictions(rows[64:]))
    assert set(load_batches(run,CONFIGS[1]))=={0}
    with pytest.raises(ValueError,match='incomplete'):
        finalize_config(run,CONFIGS[1])
    target=finalize_config(run,CONFIGS[0])
    info,records=validate_result(target,run)
    assert len(records)==67 and info['score_layer_index']==22
    assert info['prune_after_layer_index'] is None
    with pytest.raises(ValueError):
        validate_result(target,run,expected_config=CONFIGS[1])
    save_batch(run,CONFIGS[1],64,predictions(rows[64:]))
    info,_=validate_result(finalize_config(run,CONFIGS[1]),run)
    assert info['prune_after_layer_index']==2


def test_corruption_selection_and_immutable_batch(tmp_path):
    run,rows,indices=fixture(tmp_path,2)
    name=CONFIGS[0]
    save_batch(run,name,0,predictions(rows))
    changed=predictions(rows); changed[0]['raw_prediction']='Other'; changed[0]['prediction']='Other'
    with pytest.raises(ValueError,match='overwrite'):
        save_batch(run,name,0,changed)
    path=tmp_path/'configs'/name/'batches'/'000000.json'
    value=json.loads(path.read_text()); value['records'][0]['prediction']='bad'; path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='checksum'):
        load_batches(run,name)
    array=np.load(tmp_path/'selection.npy',mmap_mode='r+'); array[0,0]=3; array.flush(); del array
    with pytest.raises(ValueError,match='hash'):
        load_selection(run)


def test_selection_orphan_and_move_portable(tmp_path):
    run,rows,indices=fixture(tmp_path/'run',2)
    with pytest.raises(ValueError):
        write_selection(run,indices[:,::-1])
    with pytest.raises(ValueError):
        load_batches(run,'layer_22')
    # A partial atomic write is ignored; only committed .json files count.
    batches=tmp_path/'run'/'configs'/CONFIGS[0]/'batches'; batches.mkdir(parents=True)
    (batches/'partial.tmp').write_text('incomplete')
    assert load_batches(run,CONFIGS[0])=={}
    save_batch(run,CONFIGS[0],0,predictions(rows))
    result=finalize_config(run,CONFIGS[0])
    shutil.copytree(result,tmp_path/CONFIGS[0])
    info,records=validate_result(tmp_path/CONFIGS[0])
    assert info['count']==2 and len(records)==2
    with pytest.raises(ValueError):
        save_batch(run,CONFIGS[0],1,predictions(rows[1:]))

@pytest.mark.parametrize('change', ['range','dtype','duplicate','shape'])
def test_invalid_shared_selection(tmp_path,change):
    rows=[dict(row_index=0,sample_id='s',eval_id=1,answer='a')]
    run=read_run(initialize_run(tmp_path,{'batch_size':64},rows,{}))
    indices=np.arange(32,dtype=np.uint16)[None,:]
    if change=='range': indices[0,-1]=384
    if change=='dtype': indices=indices.astype(np.int32)
    if change=='duplicate': indices[0,1]=0
    if change=='shape': indices=indices[:,:31]
    with pytest.raises(ValueError): write_selection(run,indices)


def test_committed_selection_cannot_change_and_orphan_can_resume(tmp_path):
    run,rows,indices=fixture(tmp_path,2)
    changed=indices+np.uint16(1)
    with pytest.raises(ValueError,match='overwrite'): write_selection(run,changed)
    # Interrupted publication before selection.json is safely regenerated, with no journals present.
    (tmp_path/'selection.json').unlink()
    write_selection(run,indices)
    save_batch(run,CONFIGS[0],0,predictions(rows))
    (tmp_path/'selection.json').unlink()
    with pytest.raises(ValueError,match='missing'): write_selection(run,indices)


def test_result_checks_record_schema_even_without_run(tmp_path):
    from smolvlm_attention.sample_store import atomic_json,digest
    run,rows,indices=fixture(tmp_path,2)
    save_batch(run,CONFIGS[0],0,predictions(rows))
    result=finalize_config(run,CONFIGS[0])
    path=result/'predictions.jsonl'
    records=[json.loads(line) for line in path.read_text().splitlines()]
    records[0]['generated_token_count']=200
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    info=json.loads((result/'metadata.json').read_text())
    info['sha256']['predictions.jsonl']=digest(path)
    atomic_json(result/'metadata.json',info)
    with pytest.raises(ValueError,match='continuation'): validate_result(result)
