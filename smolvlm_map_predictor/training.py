"""Selective SmolVLM training lifecycle, with no decoder call in training/validation.

Reference: efficient_vlm_ad/smolvlm/pipeline.py at
4d7dbd238c7960d305c882b75220fb95beb37644. Objective is predictor KL, not CE.
"""
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from accelerate import Accelerator, DataLoaderConfiguration

from smolvlm_attention.sample_store import atomic_json
from . import checkpoint
from .config import recipe


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def precision():
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def window_size(index, batches, accumulation):
    return min(accumulation, batches - (index // accumulation) * accumulation)


def window_boundary(index, batches, accumulation):
    return (index + 1) % accumulation == 0 or index + 1 == batches


def loss_for_backward(raw, accumulation, actual_window_size):
    return raw if actual_window_size == accumulation else raw * (accumulation / actual_window_size)


class ProgressWriter:
    """Same flat timestamp/event progress schema as the source SmolVLM lifecycle."""
    def __init__(self, output, enabled=True, filename='train_progress.jsonl'):
        self.path, self.enabled = Path(output) / 'debug' / filename, enabled
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event, **payload):
        if self.enabled:
            row = dict(timestamp=datetime.now(timezone.utc).isoformat(), event=event, **payload)
            with self.path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, default=str) + '\n')
                stream.flush()



def memory(device):
    if device.type != 'cuda':
        return dict(cuda_memory_allocated=None, cuda_memory_peak=None)
    return dict(cuda_memory_allocated=torch.cuda.memory_allocated(device),
                cuda_memory_peak=torch.cuda.max_memory_allocated(device),
                cuda_memory_reserved=torch.cuda.memory_reserved(device),
                cuda_peak_reserved=torch.cuda.max_memory_reserved(device))


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def make_predictor(backbone, cfg):
    from .model import MapPredictor
    architecture = cfg['architecture']
    return MapPredictor(backbone.native.config.text_config,
        hidden_dim=architecture['hidden_dim'], projection_dim=architecture['projection_dim'],
        visual_tokens=architecture['visual_tokens'])


def make_loader(receipt, backbone, cfg, split, *, rank=0, world_size=1, stress=False):
    from .data import PredictorDataset, PredictorCollator
    settings = cfg['training']
    dataset = PredictorDataset(receipt, split)
    if stress:
        lengths = dataset.prompt_lengths
        rows = np.arange(rank, len(dataset), world_size)
        # Keep a real prefetch queue alive while the first (longest) batch is exercised.
        queued_batches = settings['num_workers'] * settings['prefetch_factor'] + 1
        rows = rows[np.lexsort((rows, -np.asarray(lengths)[rows].astype(np.int64)))][:settings['batch_size'] * queued_batches]
        dataset = Subset(dataset, rows.tolist())
    elif split != 'train':
        dataset = Subset(dataset, range(rank, len(dataset), world_size))
    options = dict(num_workers=settings['num_workers'], pin_memory=settings['pin_memory'],
        generator=torch.Generator().manual_seed(settings['seed']),
        persistent_workers=settings['num_workers'] > 0)
    if settings['num_workers']:
        options.update(prefetch_factor=settings['prefetch_factor'], multiprocessing_context='spawn')
    return DataLoader(dataset, batch_size=settings['batch_size'],
        shuffle=split == 'train' and not stress, drop_last=False,
        collate_fn=PredictorCollator(receipt, backbone.processor, max_new_tokens=0), **options)


def close_loader(loader):
    iterator = getattr(loader, '_iterator', None)
    if iterator is not None:
        iterator._shutdown_workers()
        loader._iterator = None
    # Accelerate wraps the original loader and retains its iterator attributes itself.
    original = getattr(loader, 'base_dataloader', None)
    if original is not None and original is not loader:
        close_loader(original)


def forbid_backbone_forward(backbone, *, projected=False):
    """Fail immediately if the frozen prefix is accidentally replaced by full-model forward."""
    handles = []
    def forbidden(module, args):
        raise AssertionError('Decoder/vision/LM head must not run during predictor training or validation')
    modules = [backbone.native, backbone.native.model, backbone.native.model.vision_model,
                   backbone.native.model.text_model, backbone.native.lm_head,
                   *backbone.native.model.text_model.layers]
    if projected:
        modules.append(backbone.native.model.connector)
    for module in modules:
        handles.append(module.register_forward_pre_hook(forbidden))
    return handles


@torch.no_grad()
def validation(predictor, backbone, batches, accelerator, *, keep_tokens=32):
    from .model import frozen_features, kl_per_sample, topk_indices
    previous = predictor.training
    predictor.eval()
    totals = torch.zeros(3, dtype=torch.float64, device=accelerator.device)
    status = getattr(batches, 'status', None)
    processed = 0
    try:
        for batch in batches:
            batch = move_batch(batch, accelerator.device)
            log_pred = predictor(**frozen_features(backbone, batch))
            values = kl_per_sample(log_pred, batch['log_target'])
            processed += len(values)
            if status:
                status.update('validation', processed, batches.sample_count, epoch=batches.epoch + 1)
            if not bool(torch.isfinite(values).all()):
                raise FloatingPointError('Nonfinite validation predictor KL')
            selected = topk_indices(log_pred, min(keep_tokens, log_pred.shape[-1]))
            target = topk_indices(batch['log_target'], selected.shape[-1])
            overlap = (selected[:, :, None] == target[:, None, :]).any(-1).float().mean(-1)
            totals += torch.stack((values.double().sum(), values.new_tensor(len(values)).double(), overlap.double().sum()))
        if status:
            status.update('validation', processed, batches.sample_count, force=True, epoch=batches.epoch + 1)
        totals = accelerator.reduce(totals, reduction='sum')
        if totals[1] <= 0:
            raise ValueError('Empty validation split')
        return dict(val_loss=float(totals[0] / totals[1]), top32_overlap=float(totals[2] / totals[1]),
                    sample_count=int(totals[1]))
    finally:
        predictor.train(previous)


def accelerator_for(cfg, *, cpu=False):
    train = cfg['training']
    return Accelerator(cpu=cpu, mixed_precision='no',
        gradient_accumulation_steps=train['gradient_accumulation_steps'], step_scheduler_with_optimizer=False,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True, data_seed=train['seed'],
                                                  even_batches=True, non_blocking=True))


def train(cfg, receipt, backbone_checkpoint, output, *, resume=None, cpu=False,
          backbone=None, stop_after_epoch=None, max_updates=None):
    """Injected backbone is only for real tiny-model offline lifecycle verification."""
    from .data import read_receipt
    from .model import frozen_features, kl_per_sample
    from smolvlm_inference.model import load_model, CAMERA_ORDER
    train_cfg = cfg['training']
    precision(); seed(train_cfg['seed'])
    accelerator = accelerator_for(cfg, cpu=cpu)
    if accelerator.num_processes != train_cfg['world_size']:
        raise ValueError('Training world size differs from configured recipe')
    description = read_receipt(receipt)
    identity = description['identity']
    output = Path(output)
    if backbone is None:
        backbone = load_model(backbone_checkpoint, device=accelerator.device, expected_prompt_camera_order=CAMERA_ORDER)
    else:
        backbone = backbone.to(accelerator.device)
    backbone.eval().requires_grad_(False)
    predictor = make_predictor(backbone, cfg).to(accelerator.device)
    if any(p.requires_grad for p in backbone.parameters()):
        raise AssertionError('Backbone must be frozen')
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=train_cfg['learning_rate'], weight_decay=train_cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_cfg['scheduler_gamma'])
    pool = None
    if description.get('version') == 2:
        from smolvlm_predictor_io.loader import PersistentPool, PoolView
        pool = PersistentPool(receipt, rank=accelerator.process_index, world_size=accelerator.num_processes,
            batch_size=train_cfg['batch_size'], seed=train_cfg['seed'], num_workers=train_cfg['num_workers'],
            pin_memory=train_cfg['pin_memory'], prefetch_factor=train_cfg['prefetch_factor'])
        batches, val_batches = PoolView(pool, 'train'), PoolView(pool, 'val')
    else:
        batches = make_loader(receipt, backbone, cfg, 'train')
        val_batches = make_loader(receipt, backbone, cfg, 'val', rank=accelerator.process_index, world_size=accelerator.num_processes)
    start_epoch, cursor, updates, global_step, best, best_epoch = 0, 0, 0, 0, float('inf'), None
    restored_rng = None
    if resume:
        payload = checkpoint.read(resume, predictor, identity, training=True, recipe=recipe(cfg), world_size=accelerator.num_processes)
        predictor.load_state_dict(payload['model_state_dict'], strict=True)
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        scheduler.load_state_dict(payload['scheduler_state_dict'])
        state = payload['metadata']
        start_epoch, cursor, updates, global_step = (state[key] for key in ('epoch', 'batch_cursor', 'optimizer_step', 'global_step'))
        best, best_epoch = state['best_val_loss'], state['best_epoch']
        restored_rng = checkpoint.rank_rng(resume, state, accelerator.process_index)
        del payload
    before = (description['counts']['train'] + train_cfg['batch_size'] - 1) // train_cfg['batch_size'] if pool else len(batches)
    if pool:
        predictor, optimizer = accelerator.prepare(predictor, optimizer)
    else:
        predictor, optimizer, batches = accelerator.prepare(predictor, optimizer, batches)
    native = accelerator.unwrap_model(predictor)
    if restored_rng is None:
        seed(train_cfg['seed'] + accelerator.process_index)
    writer = ProgressWriter(output, accelerator.is_main_process)
    debug = ProgressWriter(output, accelerator.is_main_process, filename='debug_events.jsonl')
    debug.write('TRAINING', backend='map_predictor_v1', seed=train_cfg['seed'],
        sampler='accelerate_seedable_even_batches_v1', loader_before_prepare=before, loader_after_prepare=len(batches),
        batch_per_gpu=train_cfg['batch_size'], effective_batch_nominal=train_cfg['batch_size'] *
            train_cfg['gradient_accumulation_steps'] * accelerator.num_processes,
        world_size=accelerator.num_processes, parameter_count=sum(p.numel() for p in native.parameters()),
        frozen_modules=['vision_model', 'connector', 'text_model', 'lm_head'], trainable_modules=['predictor'],
        feature_input_dtype='float32' if pool else 'float16', compute_dtype='float32', loss_type='teacher_to_predictor_kl',
        rng_restored=resume is not None, decoder_calls=0, tf32=False, autocast=False,
        gradient_accumulation_steps=train_cfg['gradient_accumulation_steps'], num_workers=train_cfg['num_workers'],
        prefetch_factor=train_cfg['prefetch_factor'])
    guards = forbid_backbone_forward(backbone, projected=pool is not None)
    A = train_cfg['gradient_accumulation_steps']
    optimizer.zero_grad(set_to_none=True)
    final_state = None
    from smolvlm_predictor_io.progress import StatusWriter
    live = StatusWriter(output, accelerator.process_index)
    if pool:
        val_batches.status = live
    try:
        for epoch in range(start_epoch, min(train_cfg['epochs'], stop_after_epoch or train_cfg['epochs'])):
            batches.set_epoch(epoch)
            if pool:
                val_batches.set_epoch(epoch)
            iterator = iter(batches)
            for _ in range(cursor):
                next(iterator)
            if restored_rng is not None:
                checkpoint.restore_rng(restored_rng)
                restored_rng = None
            predictor.train()
            total = len(batches)
            if not 0 <= cursor < total:
                raise ValueError('Invalid resumed batch cursor')
            remaining = total - cursor
            epoch_started = time.perf_counter()
            sums = torch.zeros(2, dtype=torch.float64, device=accelerator.device)
            writer.write('epoch_start', stage='predictor', epoch=epoch + 1, total_epochs=train_cfg['epochs'],
                global_step=global_step, total_steps_epoch=total, step_in_epoch=cursor, learning_rate=optimizer.param_groups[0]['lr'])
            consumed = cursor
            live.update('train', consumed, total, force=True, epoch=epoch + 1)
            for index in range(remaining):
                batch = move_batch(next(iterator), accelerator.device)
                size, boundary = window_size(index, remaining, A), window_boundary(index, remaining, A)
                with nullcontext() if boundary else accelerator.no_sync(predictor):
                    losses = kl_per_sample(predictor(**frozen_features(backbone, batch)), batch['log_target'])
                    raw = losses.mean()
                    if not bool(torch.isfinite(raw)):
                        raise FloatingPointError('Nonfinite predictor KL')
                    accelerator.backward(loss_for_backward(raw, A, size))
                consumed += 1; global_step += 1
                live.update('train', consumed, total, epoch=epoch + 1)
                # Match the source train summary: mean native microbatch loss.
                # Validation separately weights every sample exactly once.
                sums[0] += raw.detach().double(); sums[1] += 1
                norm = None
                if boundary:
                    norm = accelerator.clip_grad_norm_(predictor.parameters(), train_cfg['max_grad_norm'])
                    if not bool(torch.isfinite(norm)):
                        raise FloatingPointError('Nonfinite predictor gradient norm')
                    optimizer.step()
                    if not accelerator.optimizer_step_was_skipped:
                        updates += 1
                    optimizer.zero_grad(set_to_none=True)
                if index == 0 or global_step % train_cfg['progress_log_every_steps'] == 0 or consumed == total:
                    elapsed = max(time.perf_counter() - epoch_started, 1e-9)
                    speed = (index + 1) / elapsed
                    writer.write('train_step', stage='predictor', epoch=epoch + 1, global_step=global_step,
                        step_in_epoch=consumed, total_steps_epoch=total, loss=float(raw.detach()),
                        lr=optimizer.param_groups[0]['lr'], finite_loss=True, optimizer_step=updates,
                        accumulation_window_size=size, elapsed_seconds=elapsed, steps_per_second=speed,
                        estimated_epoch_remaining_seconds=(remaining - index - 1) / speed, **memory(accelerator.device))
                    debug.write('TRAIN_STEP', epoch=epoch + 1, global_step=global_step, optimizer_step=updates,
                        loss=float(raw.detach()), gradient_norm_before_clipping=float(norm) if norm is not None else None,
                        gradients_finite=True, decoder_calls=0)
                if max_updates is not None and updates >= max_updates and boundary:
                    break
            live.update('train', consumed, total, force=True, epoch=epoch + 1)
            complete = consumed == total
            if pool:
                if complete:
                    sentinel = object()
                    if next(iterator, sentinel) is not sentinel:
                        raise AssertionError('Loader yielded more batches than its schedule')
                else:
                    iterator.close()
            if complete:
                scheduler.step()
            sums = accelerator.reduce(sums, reduction='sum')
            train_loss = float(sums[0] / sums[1])
            writer.write('validation_start', stage='predictor', epoch=epoch + 1, global_step=global_step, train_loss=train_loss)
            valid = validation(native, backbone, val_batches, accelerator, keep_tokens=cfg['evaluation']['keep_tokens'])
            writer.write('validation_end', stage='predictor', epoch=epoch + 1, global_step=global_step, train_loss=train_loss, **valid)
            improved = valid['val_loss'] < best
            if improved:
                best, best_epoch = valid['val_loss'], epoch + 1
            final_state = dict(epoch=epoch + 1 if complete else epoch, batch_cursor=0 if complete else consumed,
                completed_epochs=epoch + 1 if complete else epoch, optimizer_step=updates, global_step=global_step,
                best_val_loss=best, best_epoch=best_epoch, val_loss=valid['val_loss'], train_loss=train_loss)
            # Publish an improved best before latest can advertise its new best metadata.
            # A crash between the two saves leaves the previous latest resumable.
            for kind in (['best'] if improved else []) + ['latest']:
                path = output / 'checkpoints' / f'predictor_{kind}.pt'
                checkpoint.save(path, native, optimizer, scheduler, final_state, identity, recipe(cfg), accelerator)
                writer.write('checkpoint_saved', stage='predictor', checkpoint_kind=kind, path=str(path),
                    epoch=epoch + 1, global_step=global_step, train_loss=train_loss, val_loss=valid['val_loss'])
            writer.write('epoch_end', stage='predictor', epoch=epoch + 1, total_epochs=train_cfg['epochs'],
                global_step=global_step, completed=complete, train_loss=train_loss, **valid,
                best_val_loss=best, best_epoch=best_epoch, elapsed_seconds=time.perf_counter() - epoch_started,
                **memory(accelerator.device))
            if accelerator.is_main_process:
                atomic_json(output / 'training_status.json', dict(final_state, target_epochs=train_cfg['epochs'],
                    training_complete=final_state['completed_epochs'] >= train_cfg['epochs']))
            cursor = 0
            if not complete or (max_updates is not None and updates >= max_updates):
                break
        if final_state is None and resume:
            final_state = state
        if final_state is not None and accelerator.is_main_process:
            atomic_json(output / 'training_status.json', dict(final_state, target_epochs=train_cfg['epochs'],
                training_complete=final_state['completed_epochs'] >= train_cfg['epochs']))
        accelerator.wait_for_everyone()
        return final_state
    finally:
        for handle in guards:
            handle.remove()
        if pool:
            pool.close()
        else:
            close_loader(batches); close_loader(val_batches)
