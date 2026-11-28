import json
import numpy as np
import pytest

from smolvlm_combined_topk.config import BATCH_SIZE, LAYER_IDS, worker_configs
from smolvlm_combined_topk.store import (ShardStore, write_run, read_run, shard_bounds,
    finalize_export, open_attention, prepare_selections, load_selections)


def rows(n=5):
    return [dict(row_index=i, sample_id=f's-{i}', eval_id=i, question='What?', answer='Red.',
        camera_paths={'Front': str(i)}, feature_index=i, feature_key=f'k{i}',
        input_ids=[9]*384+[7], image_positions=list(range(384)), image_token_id=9,
        prompt_length=385, query_position=384, query_token_id=7) for i in range(n)]


def run(tmp_path):
    return write_run(tmp_path, dict(identity={'policy':'test'}, geometry=[10,384],
        layer_ids=list(LAYER_IDS), batch_size=64, shard_count=2), rows())


def test_config():
    assert BATCH_SIZE == 64
    assert worker_configs(0) == ('layer_10','layer_15','layer_16','layer_17','layer_21')
    assert worker_configs(1) == ('layer_22','layer_23','layer_25','layer_26','layer_27')
    assert shard_bounds(16817,2,0) == (0,8409)


def test_commit_recovery_and_corruption(tmp_path):
    store=ShardStore(tmp_path,'id',[2,3])
    value=np.full((1,10,384),.001,dtype=np.float32)
    store.write([2],value)
    assert ShardStore(tmp_path,'id',[2,3]).completed == set()
    store.commit()
    assert ShardStore(tmp_path,'id',[2,3]).completed == {2}
    with pytest.raises(ValueError): store.commit(batch_size=32)
    store.array[0,0,0]=.002; store.array.flush()
    with pytest.raises(ValueError,match='checksum'): ShardStore(tmp_path,'id',[2,3])


def test_export_selection_and_compatibility(tmp_path):
    path=run(tmp_path); config=read_run(path)
    before=path.read_bytes()
    with pytest.raises(ValueError,match='incompatible'):
        write_run(tmp_path,dict(config,identity={'policy':'other'}),rows())
    assert path.read_bytes()==before
    for shard in range(2):
        start,end=shard_bounds(5,2,shard)
        store=ShardStore(tmp_path/'shards'/str(shard),config['fingerprint'],range(start,end))
        values=np.full((end-start,10,384),.001,dtype=np.float32)
        values[:,0,100]=.002
        store.write(range(start,end),values); store.commit()
    export=finalize_export(path,progress=False)
    result=open_attention(export)
    assert result.attention.shape==(5,10,384) and not result.attention.flags.writeable
    assert result.metadata['layer_ids']==list(LAYER_IDS)
    assert result.sample_index['s-4']==4
    selection=prepare_selections(export)
    selected=load_selections(selection,10)
    assert selected['indices'][0].tolist()==list(range(31))+[100]
    assert selected['absolute_positions'][0].tolist()==selected['indices'][0].tolist()
    assert load_selections(selection,15)['indices'][0].tolist()==list(range(32))
    assert finalize_export(path,progress=False)==export
    (export/'samples.jsonl').write_text('bad')
    with pytest.raises(ValueError,match='hash'): open_attention(export)


def test_incomplete_invalid_and_layer_format(tmp_path):
    path=run(tmp_path)
    with pytest.raises(ValueError,match='incomplete'): finalize_export(path,progress=False)
    store=ShardStore(tmp_path/'tmp','id',[0])
    for value in (-.1,np.nan,.1):
        with pytest.raises(ValueError): store.write([0],np.full((1,10,384),value,dtype=np.float32))
    config=json.loads(path.read_text()); config['layer_ids']=list(range(10))
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError): read_run(path)


def test_failed_progress_publish_leaves_rows_uncommitted(tmp_path, monkeypatch):
    from smolvlm_combined_topk import store as module
    target=ShardStore(tmp_path,'id',[0])
    target.write([0],np.full((1,10,384),.001,dtype=np.float32))
    original=module.atomic_json
    def fail(*args, **kwargs):
        raise OSError('interrupted before publication')
    monkeypatch.setattr(module,'atomic_json',fail)
    with pytest.raises(OSError): target.commit()
    monkeypatch.setattr(module,'atomic_json',original)
    resumed=ShardStore(tmp_path,'id',[0])
    assert resumed.completed==set()
    resumed.write([0],np.full((1,10,384),.002,dtype=np.float32)); resumed.commit()
    assert ShardStore(tmp_path,'id',[0]).completed=={0}


def test_mapping_and_topology_cannot_silently_change(tmp_path):
    path=run(tmp_path)
    config=read_run(path)
    changed=rows(); changed[0]['image_positions'][0]=1
    with pytest.raises(ValueError,match='Image positions'): write_run(tmp_path,config,changed)
    other=dict(config,shard_count=1)
    with pytest.raises(ValueError,match='incompatible'): write_run(tmp_path,other,rows())
    with pytest.raises(ValueError,match='already'): write_run(tmp_path,config,rows(),mode='measure')
    assert read_run(path)['count']==5
