import json
import numpy as np
import pytest

from smolvlm_attention.combined_state import (write_run, read_run, input_rows,
    load_shard, save_shard, shard_path, merge_shards)
from smolvlm_attention.combined_worker import process_pending


def fixture(tmp_path, n=5):
    rows = [dict(row_index=i, sample_id=str(i), eval_id=i, input_ids=[7]*384+[9],
        prompt_length=385, query_position=384, query_token_id=9, image_token_id=7) for i in range(n)]
    path = write_run(tmp_path, dict(identity={'model':'base'}, geometry=[15,384],
        batch_size=64, shard_count=2), rows)
    return read_run(path), rows


def test_resume_and_weighted_merge(tmp_path):
    run, rows = fixture(tmp_path)
    for shard, ids, value in [(0,['0','1','2'],.001), (1,['3','4'],.002)]:
        acc = load_shard(run, shard)
        process_pending([r for r in rows if r['sample_id'] in ids], acc,
            lambda selected: np.full((len(selected),15,384),value,np.float32),
            checkpoint=shard_path(run,shard))
        assert load_shard(run,shard).processed_ids == ids
    total,count = merge_shards(run)
    assert count == 5
    np.testing.assert_allclose(total/count,.0014)
    assert input_rows(run) == rows


def test_fixed_batch_no_retry_and_commit_before_progress(tmp_path):
    run,rows = fixture(tmp_path,131)
    acc=load_shard(run,0)
    calls=[]
    def measure(selected):
        calls.append(len(selected))
        if len(calls)==2: raise RuntimeError('OOM')
        return np.zeros((len(selected),15,384),np.float32)
    def progress(event):
        assert load_shard(run,0).count == event['processed']
    with pytest.raises(RuntimeError,match='64'):
        process_pending(rows[:66],acc,measure,checkpoint=shard_path(run,0),progress=progress)
    assert calls==[64,2]
    assert load_shard(run,0).count==64


def test_identity_and_snapshot_corruption(tmp_path):
    run,rows=fixture(tmp_path)
    acc=load_shard(run,0)
    acc.add(['0'],np.zeros((1,15,384),np.float32))
    path=shard_path(run,0);save_shard(acc,path)
    with np.load(path,allow_pickle=False) as f: data={k:f[k] for k in f.files}
    data['sum']=np.ones((15,384),np.float64)
    data['mean']=data['sum'].copy()
    np.savez(path,**data)
    with pytest.raises(ValueError,match='checksum'):load_shard(run,0)
    original=(tmp_path/'combined_run.json').read_bytes()
    with pytest.raises(ValueError,match='incompatible'):
        write_run(tmp_path,dict(identity={'model':'other'},geometry=[15,384],batch_size=64,shard_count=2),rows)
    assert (tmp_path/'combined_run.json').read_bytes()==original


def test_relocated_and_incomplete(tmp_path):
    run,rows=fixture(tmp_path/'a')
    with pytest.raises(ValueError,match='Incomplete'):merge_shards(run)
    import shutil
    shutil.move(str(tmp_path/'a'),str(tmp_path/'b'))
    moved=read_run(tmp_path/'b'/'combined_run.json')
    assert input_rows(moved)==rows


def test_resume_runs_only_pending_and_preserves_order(tmp_path):
    run,rows=fixture(tmp_path,131)
    acc=load_shard(run,0)
    acc.add([str(i) for i in range(64)],np.zeros((64,15,384),np.float32))
    save_shard(acc,shard_path(run,0))
    resumed=load_shard(run,0)
    calls=[]
    def measure(batch):
        calls.append([r['sample_id'] for r in batch])
        return np.zeros((len(batch),15,384),np.float32)
    process_pending(rows[:66],resumed,measure,shard_path(run,0))
    assert calls==[['64','65']]
    assert load_shard(run,0).count==66


def test_failed_measurement_does_not_commit(tmp_path):
    run,rows=fixture(tmp_path)
    acc=load_shard(run,0)
    def fail(batch):raise MemoryError('GPU OOM')
    with pytest.raises(RuntimeError,match='no batch reduction'):
        process_pending(rows[:3],acc,fail,shard_path(run,0))
    assert acc.count==0 and not shard_path(run,0).exists()
    with pytest.raises(ValueError,match='Invalid batch_size'):
        write_run(tmp_path/'other',dict(identity={},batch_size=32),rows)
