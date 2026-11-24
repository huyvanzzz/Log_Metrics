"""One persistent worker pool, explicit split-row tasks, native Accelerate order."""
import torch
import math
from torch.utils.data import DataLoader, BatchSampler
from accelerate.data_loader import SeedableRandomSampler, BatchSamplerShard
from .compact import CompactDataset, CompactCollator


class PhaseBatches:
    def __init__(self): self.batches=[]; self.active=False
    def __len__(self): return len(self.batches)
    def __iter__(self): yield from self.batches


class PersistentPool:
    def __init__(self,store,*,rank=0,world_size=1,batch_size=32,seed=42,num_workers=2,pin_memory=True,prefetch_factor=1):
        self.dataset=CompactDataset(store); self.rank=rank; self.world_size=world_size; self.batch_size=batch_size; self.seed=seed
        self.sampler=PhaseBatches(); self._iterator=None; self._phase_active=False
        opts=dict(num_workers=num_workers,pin_memory=pin_memory,persistent_workers=num_workers>0,generator=torch.Generator().manual_seed(seed+rank))
        if num_workers: opts.update(multiprocessing_context='spawn',prefetch_factor=prefetch_factor)
        self.loader=DataLoader(self.dataset,batch_sampler=self.sampler,collate_fn=CompactCollator(store),**opts)
    def _start(self,batches):
        if self._phase_active: raise RuntimeError('Previous phase must be exhausted or pool closed')
        self._phase_active=True
        self.sampler.batches=batches
        def iterate():
            complete=False
            try:
                self._iterator=iter(self.loader)
                yield from self._iterator; complete=True
            finally:
                self._iterator=None; self._phase_active=False
                if not complete: self.close()
        return iterate()
    def train(self,epoch):
        sampler=SeedableRandomSampler(range(self.dataset.info['counts']['train']),generator=torch.Generator().manual_seed(self.seed),data_seed=self.seed)
        sampler.set_epoch(epoch)
        batches=BatchSamplerShard(BatchSampler(sampler,self.batch_size,drop_last=False),num_processes=self.world_size,process_index=self.rank,split_batches=False,even_batches=True)
        return self._start([[('train',int(row)) for row in batch] for batch in batches])
    def rows(self,split,rows):
        rows=list(rows); return self._start([[(split,int(row)) for row in rows[i:i+self.batch_size]] for i in range(0,len(rows),self.batch_size)])
    def validation(self): return self.rows('val',range(self.rank,self.dataset.info['counts']['val'],self.world_size))
    def views(self): return PoolView(self,'train'), PoolView(self,'val')
    def close(self):
        iterator=getattr(self.loader,'_iterator',None)
        if iterator is not None:
            iterator._shutdown_workers(); self.loader._iterator=None
        self._iterator=None; self._phase_active=False


class PoolView:
    def __init__(self,pool,split): self.pool=pool; self.split=split; self.epoch=0
    def __len__(self):
        count=self.pool.dataset.info['counts'][self.split]
        if self.split=='train': return math.ceil(math.ceil(count/self.pool.batch_size)/self.pool.world_size)
        return math.ceil(len(range(self.pool.rank,count,self.pool.world_size))/self.pool.batch_size)
    @property
    def sample_count(self): return len(range(self.pool.rank,self.pool.dataset.info['counts'][self.split],self.pool.world_size))
    def set_epoch(self,epoch): self.epoch=epoch
    def __iter__(self):
        return self.pool.train(self.epoch) if self.split=='train' else self.pool.rows(self.split,range(self.pool.rank,self.pool.dataset.info['counts'][self.split],self.pool.world_size))
    def close(self): self.pool.close()

