"""Predictor-only checkpoints and rank RNG; frozen backbone is never serialized."""
import json
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch
from smolvlm_attention.sample_store import atomic_json, fingerprint


def rng_state():
    state = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def _atomic_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        with temporary.open('wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _counters(state):
    return {key: state[key] for key in ('epoch', 'batch_cursor', 'optimizer_step', 'global_step')}


def save(path, predictor, optimizer, scheduler, state, identity, recipe, accelerator):
    """Publish .pt last; immutable RNG generations keep the previous save usable on interruption."""
    from accelerate.utils import broadcast_object_list
    path = Path(path)
    identifier = [uuid.uuid4().hex if accelerator.is_main_process else None]
    identifier = broadcast_object_list(identifier)[0]
    rank, world = accelerator.process_index, accelerator.num_processes
    sidecar = Path(str(path) + '.rng') / identifier
    stamp = dict(checkpoint_id=identifier, identity_sha256=fingerprint(identity),
                 world_size=world, counters=_counters(state))
    accelerator.wait_for_everyone()
    _atomic_torch(sidecar / f'rank_{rank:05d}.pt', dict(rng=rng_state(), rank=rank,
        **stamp, sampler=dict(seed=recipe['training']['seed'], epoch=state['epoch'], batch_cursor=state['batch_cursor'])))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        atomic_json(sidecar / 'manifest.json', dict(stamp, ranks=list(range(world))))
        _atomic_torch(path, dict(format='map_predictor_checkpoint_v1',
            architecture_signature=predictor.architecture_signature, identity=identity,
            recipe=recipe, model_state_dict=predictor.state_dict(),
            optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
            metadata=dict(state, world_size=world, checkpoint_id=identifier)))
    accelerator.wait_for_everyone()


def read(path, predictor, identity, *, training=False, recipe=None, world_size=None):
    path = Path(path)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if (payload.get('format') != 'map_predictor_checkpoint_v1' or payload.get('identity') != identity or
            payload.get('architecture_signature') != predictor.architecture_signature):
        raise ValueError('Predictor checkpoint architecture/base/data/targets identity mismatch')
    if training:
        meta = payload['metadata']
        if payload.get('recipe') != recipe or meta.get('world_size') != world_size:
            raise ValueError('Predictor training recipe/world size mismatch')
        directory = Path(str(path) + '.rng') / meta['checkpoint_id']
        expected = dict(checkpoint_id=meta['checkpoint_id'], identity_sha256=fingerprint(identity),
                        world_size=world_size, counters=_counters(meta))
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        if manifest != dict(expected, ranks=list(range(world_size))):
            raise ValueError('Predictor RNG manifest/counters mismatch')
        for rank in range(world_size):
            side = torch.load(directory / f'rank_{rank:05d}.pt', map_location='cpu', weights_only=False)
            if side.get('rank') != rank or any(side.get(key) != value for key, value in expected.items()):
                raise ValueError('Predictor rank RNG identity/counters mismatch')
            sampler = dict(seed=recipe['training']['seed'], epoch=meta['epoch'], batch_cursor=meta['batch_cursor'])
            if side.get('sampler') != sampler or 'rng' not in side:
                raise ValueError('Predictor RNG sampler state mismatch')
        if 'optimizer_state_dict' not in payload or 'scheduler_state_dict' not in payload:
            raise ValueError('Missing predictor training state')
    return payload


def rank_rng(path, metadata, rank):
    return torch.load(Path(str(path) + '.rng') / metadata['checkpoint_id'] / f'rank_{rank:05d}.pt',
                      map_location='cpu', weights_only=False)['rng']
