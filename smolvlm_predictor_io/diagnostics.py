"""Isolated fixed-recipe diagnostics; no synchronization is added to training."""
import time
import numpy as np
import torch


def compact_stress_loader(receipt, settings, rank, world):
    from .loader import PersistentPool
    pool=PersistentPool(receipt,rank=rank,world_size=world,batch_size=settings['batch_size'],
        seed=settings['seed'],num_workers=settings['num_workers'],pin_memory=settings['pin_memory'],
        prefetch_factor=settings['prefetch_factor'])
    info=pool.dataset.info
    lengths=np.diff(np.load(info['splits']['train']['token_offsets'],mmap_mode='r'))
    rows=np.arange(rank,info['counts']['train'],world)
    rows=rows[np.lexsort((rows,-lengths[rows]))][:settings['batch_size']].tolist()
    if not rows: pool.close(); raise ValueError('No stress samples assigned to rank')
    # Repeat exactly the longest real batch: both timing and stress keep workers
    # prefetched, without creating a second persistent worker pool.
    class StressLoader:
        def __iter__(self):
            return pool._start([[('train',int(row)) for row in rows] for _ in range(23)])
        def __len__(self): return 23
        def close(self): pool.close()
    return StressLoader()


def measure_updates(iterator,predictor,optimizer,accelerator,extract,forward,kl,max_grad_norm,*,warmup,measured):
    """Run real Adam updates and return separate wall/event timings.

    CUDA is synchronized only at measurement boundaries. Event times report device
    intervals, not isolated kernel cost or proof of transfer/compute overlap.
    """
    from smolvlm_map_predictor.training import move_batch
    cuda=accelerator.device.type=='cuda'
    elapsed=dict(loader_wait_seconds=0.,h2d_wall_seconds=0.,frozen_wall_seconds=0.,predictor_update_wall_seconds=0.,cpu_collate_seconds=0.,cpu_feature_validation_copy_seconds=0.)
    events=[];samples=0; started=None;warmup_started=time.perf_counter()
    params=list(accelerator.unwrap_model(predictor).parameters())
    for update in range(warmup+measured):
        if update==warmup:
            if cuda: torch.cuda.synchronize(accelerator.device)
            started=time.perf_counter()
            warmup_seconds=started-warmup_started
        active=update>=warmup
        t=time.perf_counter(); batch=next(iterator); waited=time.perf_counter()-t
        key='projected_visual_features' if 'projected_visual_features' in batch else 'visual_features'
        pinned=batch[key].is_pinned(); feature_dtype=str(batch[key].dtype)
        record=[torch.cuda.Event(enable_timing=True) for _ in range(4)] if cuda and active else None
        if record: record[0].record()
        t=time.perf_counter(); batch=move_batch(batch,accelerator.device); transfer=time.perf_counter()-t
        if record: record[1].record()
        t=time.perf_counter(); features=extract(batch); frozen=time.perf_counter()-t
        if record: record[2].record()
        t=time.perf_counter(); optimizer.zero_grad(set_to_none=True)
        output_log=forward(predictor,features);loss=kl(output_log,batch['log_target']).mean()
        if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite diagnostic loss')
        accelerator.backward(loss)
        finite=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params)
        if not finite: raise FloatingPointError('Missing or nonfinite diagnostic gradients')
        norm=accelerator.clip_grad_norm_(predictor.parameters(),max_grad_norm)
        if not bool(torch.isfinite(norm)): raise FloatingPointError('Nonfinite diagnostic gradient norm')
        optimizer.step()
        if accelerator.optimizer_step_was_skipped: raise RuntimeError('Diagnostic optimizer update skipped')
        if record: record[3].record();events.append(record)
        compute=time.perf_counter()-t
        if active:
            samples+=len(batch['sample_ids'])
            elapsed['loader_wait_seconds']+=waited;elapsed['h2d_wall_seconds']+=transfer
            elapsed['frozen_wall_seconds']+=frozen;elapsed['predictor_update_wall_seconds']+=compute
            elapsed['cpu_collate_seconds']+=float(batch.get('cpu_collate_seconds',0.))
            elapsed['cpu_feature_validation_copy_seconds']+=float(batch.get('cpu_feature_validation_copy_seconds',0.))
    if cuda: torch.cuda.synchronize(accelerator.device)
    total=time.perf_counter()-started
    timings=dict(warmup_seconds=warmup_seconds,warmup_updates=warmup,measured_updates=measured,measured_samples=samples,pipeline_seconds=total,
        samples_per_second=samples/total,seconds_per_update=total/measured,**elapsed,
        h2d_cuda_seconds=sum(a.elapsed_time(b) for a,b,c,d in events)/1000 if cuda else None,
        frozen_cuda_seconds=sum(b.elapsed_time(c) for a,b,c,d in events)/1000 if cuda else None,
        predictor_update_cuda_seconds=sum(c.elapsed_time(d) for a,b,c,d in events)/1000 if cuda else None,
        cpu_collate_timing_available='cpu_collate_seconds' in batch,
        batch_policy='longest_actual_batch_repeated_stress',
        timing_scope='diagnose_only_boundary_synchronized_device_intervals')
    return dict(batch=batch,output_log=output_log,loss=loss,grad_norm=norm,finite=finite,
        pinned=pinned,feature_dtype=feature_dtype,params=params,timing=timings)
