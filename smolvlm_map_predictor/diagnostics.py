"""Two-rank stress diagnostic for the frozen-feature predictor pipeline."""
from pathlib import Path
import torch

from smolvlm_attention.sample_store import atomic_json
from . import checkpoint
from .training import (accelerator_for, close_loader, forbid_backbone_forward, make_loader,
                       make_predictor, memory, move_batch, precision, seed)


def diagnose(cfg, receipt, backbone_checkpoint, output, *, cpu=False, backbone=None):
    """Run one real optimizer update on each rank's longest training prompts.

    Call under the same accelerate launch and loader configuration as training.
    """
    from .data import read_receipt
    from .model import frozen_features, kl_per_sample
    from .runtime import resource_snapshot
    from smolvlm_inference.model import load_model, CAMERA_ORDER
    state = checkpoint.rng_state()
    loader = None
    guards = []
    try:
        precision()
        accelerator = accelerator_for(cfg, cpu=cpu)
        settings = cfg['training']
        if accelerator.num_processes != settings['world_size']:
            raise ValueError('Diagnostic world size differs from configured recipe')
        read_receipt(receipt)
        seed(settings['seed'])
        if backbone is None:
            backbone = load_model(backbone_checkpoint, device=accelerator.device,
                                  expected_prompt_camera_order=CAMERA_ORDER)
        else:
            backbone = backbone.to(accelerator.device)
        backbone.eval().requires_grad_(False)
        predictor = make_predictor(backbone, cfg).to(accelerator.device)
        optimizer = torch.optim.AdamW(predictor.parameters(), lr=settings['learning_rate'],
                                      weight_decay=settings['weight_decay'])
        parameter_ids = [id(p) for group in optimizer.param_groups for p in group['params']]
        unique = len(parameter_ids) == len(set(parameter_ids))
        if not unique or set(parameter_ids) != {id(p) for p in predictor.parameters()}:
            raise ValueError('Optimizer must contain each predictor parameter exactly once')
        loader = make_loader(receipt, backbone, cfg, 'train', rank=accelerator.process_index,
                             world_size=accelerator.num_processes, stress=True)
        if not len(loader):
            raise ValueError('No stress samples assigned to rank')
        predictor, optimizer = accelerator.prepare(predictor, optimizer)
        guards = forbid_backbone_forward(backbone)
        if accelerator.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(accelerator.device)
        before = dict(**resource_snapshot(), **memory(accelerator.device))
        batch = next(iter(loader))
        pinned = batch['visual_features'].is_pinned()
        feature_dtype = str(batch['visual_features'].dtype)
        batch = move_batch(batch, accelerator.device)
        output_log = predictor(**frozen_features(backbone, batch))
        losses = kl_per_sample(output_log, batch['log_target'])
        loss = losses.mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite diagnostic loss')
        accelerator.backward(loss)
        params = list(accelerator.unwrap_model(predictor).parameters())
        finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params)
        if not finite:
            raise FloatingPointError('Missing or nonfinite diagnostic gradients')
        grad_norm = accelerator.clip_grad_norm_(predictor.parameters(), settings['max_grad_norm'])
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError('Nonfinite diagnostic gradient norm')
        optimizer.step()
        if accelerator.optimizer_step_was_skipped:
            raise RuntimeError('Diagnostic optimizer update skipped')
        raw_optimizer = getattr(optimizer, 'optimizer', optimizer)
        if len(raw_optimizer.state) != len(params):
            raise AssertionError('Diagnostic optimizer state was not initialized')
        if any(p.requires_grad or p.grad is not None for p in backbone.parameters()):
            raise AssertionError('Frozen backbone received gradients')
        after = dict(**resource_snapshot(), **memory(accelerator.device))
        result = dict(ok=True, rank=accelerator.process_index, world_size=accelerator.num_processes,
            batch_size=len(batch['sample_ids']), sample_ids=batch['sample_ids'],
            prompt_lengths=batch['attention_mask'].sum(1).tolist(), loss=float(loss.detach()),
            gradient_norm=float(grad_norm), gradients_finite=finite, optimizer_parameter_ids_unique=unique,
            optimizer_parameter_count=len(parameter_ids), predictor_parameter_count=sum(p.numel() for p in params),
            input_feature_dtype=feature_dtype, compute_dtype=str(output_log.dtype), pinned_features=pinned, optimizer_state_count=len(raw_optimizer.state),
            backbone_frozen=True, decoder_calls=0, loader_num_workers=settings['num_workers'],
            loader_prefetch_factor=settings['prefetch_factor'] if settings['num_workers'] else None,
            memory_before=before, memory_after=after)
        atomic_json(Path(output)/f'diagnose_rank_{accelerator.process_index}.json', result)
        accelerator.wait_for_everyone()
        return result
    finally:
        for handle in guards:
            handle.remove()
        if loader is not None:
            close_loader(loader)
        checkpoint.restore_rng(state)
