import json
import numpy as np
import pytest
import torch
from smolvlm_attention.teacher_targets.loader import file_stat
from smolvlm_predictor_io.compact import CompactDataset, CompactCollator


def store(tmp_path):
    projected=tmp_path/'projected.npy'; np.save(projected,np.arange(2*6*2*3,dtype=np.float32).reshape(2,6,2,3))
    manifest=tmp_path/'manifest.json'; manifest.write_text(json.dumps({'shape':[2,6,2,3]}))
    splits={}; files={}
    for split in ('train','val','test'):
        paths={}
        for name,value in dict(token_offsets=[0,15,31],question_offsets=[0,1,3],images=[list(range(12)),list(range(12))],features=[1,0],target_rows=[0,1],metadata_offsets=[0,0,0]).items():
            path=tmp_path/f'{split}_{name}.npy'; np.save(path,np.asarray(value,dtype='<i8')); paths[name]=str(path)
        for name,value in dict(tokens=list(range(15))+list(range(16)),questions=[12,12,13]).items():
            path=tmp_path/f'{split}_{name}.bin'; path.write_bytes(np.asarray(value,dtype='<i8').tobytes()); paths[name]=str(path)
        metadata=tmp_path/f'{split}.jsonl'; lines=[json.dumps(dict(sample_id=f'{split}{i}',answer='answer',eval_id=i))+'\n' for i in range(2)]
        metadata.write_text(''.join(lines)); np.save(paths['metadata_offsets'],np.array([0,len(lines[0]),len(''.join(lines))],dtype='<i8')); paths['metadata']=str(metadata)
        if split!='test':
            target=tmp_path/f'{split}_target.npy'; np.save(target,np.full((2,12),-np.log(12),dtype='<f4')); paths['target']=str(target)
        splits[split]=paths
        files.update({p:file_stat(p) for p in paths.values()})
    path=tmp_path/'store.json'; path.write_text(json.dumps(dict(version=2,counts=dict(train=2,val=2,test=2),geometry={},max_length=100,pad_token_id=99,splits=splits,files=files,projected_manifest=str(manifest))))
    return path


def test_compact_projected_mapping_padding_and_targets(tmp_path):
    path=store(tmp_path); data=CompactDataset(path); collate=CompactCollator(path)
    batch=collate([data[('train',0)],data[('train',1)]])
    assert batch['projected_visual_features'].shape==(2,12,3)
    assert batch['projected_visual_features'].dtype==torch.float32
    assert batch['input_ids'][0,0]==99
    assert batch['image_positions'][0].tolist()==list(range(1,13))
    assert batch['question_columns'].tolist()==[[13,0],[12,13]]
    assert batch['query_columns'].tolist()==[15,15]
    assert batch['log_target'].shape==(2,12)
    assert batch['projected_visual_features'][0,0,0]==36
    test=collate([data[('test',0)]])
    assert 'log_target' not in test and test['answers']==['answer']


def test_compact_nonfinite_rejected(tmp_path):
    path=store(tmp_path); values=np.load(tmp_path/'projected.npy',mmap_mode='r+'); values[1,0,0,0]=np.inf; values.flush()
    with pytest.raises(ValueError,match='nonfinite'):
        CompactCollator(path)([CompactDataset(path)[('train',0)]])


def test_one_spawn_pool_cross_phase_and_cleanup(tmp_path):
    from smolvlm_predictor_io.loader import PersistentPool
    path=store(tmp_path); pool=PersistentPool(path,batch_size=1,num_workers=2,pin_memory=False)
    try:
        train=list(pool.train(0)); pids=[w.pid for w in pool.loader._iterator._workers]
        val=list(pool.validation())
        assert pids==[w.pid for w in pool.loader._iterator._workers]
        assert sorted(x['sample_ids'][0] for x in train)==['train0','train1']
        assert [x['sample_ids'][0] for x in val]==['val0','val1']
        second=pool.train(1); next(second); second.close()
        assert pool.loader._iterator is None
    finally: pool.close()


@pytest.mark.parametrize('n',[1,31,32,33,65,341381])
def test_native_sampler_parity(n):
    from accelerate.data_loader import prepare_data_loader, SeedableRandomSampler, BatchSamplerShard
    from torch.utils.data import DataLoader,BatchSampler
    for rank in range(2):
        for epoch in (0,1):
            reference=prepare_data_loader(DataLoader(range(n),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(42)),num_processes=2,process_index=rank,use_seedable_sampler=True,data_seed=42,even_batches=True)
            reference.set_epoch(epoch)
            sampler=SeedableRandomSampler(range(n),generator=torch.Generator().manual_seed(42),data_seed=42); sampler.set_epoch(epoch)
            actual=BatchSamplerShard(BatchSampler(sampler,32,False),num_processes=2,process_index=rank,even_batches=True,split_batches=False)
            assert [list(x) for x in actual]==[x.tolist() for x in reference]


def test_pool_refuses_switch_before_first_batch_is_consumed(tmp_path):
    from smolvlm_predictor_io.loader import PersistentPool
    path=store(tmp_path);pool=PersistentPool(path,batch_size=1,num_workers=0,pin_memory=False)
    first=pool.train(0)
    try:
        with pytest.raises(RuntimeError,match='Previous phase'):
            pool.validation()
    finally:
        first.close();pool.close()

def test_pool_resets_phase_if_iterator_start_fails(tmp_path):
    from smolvlm_predictor_io.loader import PersistentPool
    pool=PersistentPool(store(tmp_path),num_workers=0,pin_memory=False)
    class Failing:
        _iterator=None
        def __iter__(self):raise RuntimeError('loader startup failed')
    pool.loader=Failing()
    try:
        with pytest.raises(RuntimeError,match='startup failed'):next(pool.train(0))
        assert not pool._phase_active
    finally:pool.close()
