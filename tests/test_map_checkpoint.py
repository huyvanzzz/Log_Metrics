import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch


class Predictor(torch.nn.Linear):
    @property
    def architecture_signature(self):
        return {'fixture': [self.in_features, self.out_features]}


def fixture_state():
    from accelerate import Accelerator
    from smolvlm_map_predictor.config import load_config, recipe
    predictor = Predictor(3, 2)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-4)
    predictor(torch.ones(2, 3)).sum().backward(); optimizer.step(); optimizer.zero_grad()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.9)
    state = dict(epoch=1, batch_cursor=0, optimizer_step=1, global_step=1,
                 best_val_loss=.4, val_loss=.4, best_epoch=1)
    return predictor, optimizer, scheduler, state, recipe(load_config()), Accelerator(cpu=True)


def test_predictor_checkpoint_rng_identity_and_inference_without_sidecars(tmp_path):
    from smolvlm_map_predictor.checkpoint import save, read, rank_rng, restore_rng
    predictor, opt, sched, state, recipe, accelerator = fixture_state()
    identity = {'base_sha': 'original', 'targets_sha': 'teacher'}
    path = tmp_path / 'predictor_latest.pt'
    save(path, predictor, opt, sched, state, identity, recipe, accelerator)
    expected = (random.random(), np.random.rand(), torch.rand(2))
    payload = read(path, predictor, identity, training=True, recipe=recipe, world_size=1)
    restore_rng(rank_rng(path, payload['metadata'], 0))
    actual = (random.random(), np.random.rand(), torch.rand(2))
    assert actual[:2] == expected[:2] and torch.equal(actual[2], expected[2])
    assert set(payload['model_state_dict']) == {'weight', 'bias'}
    with pytest.raises(ValueError, match='identity'):
        read(path, predictor, {'base_sha': 'other'})
    with pytest.raises(ValueError, match='recipe/world'):
        read(path, predictor, identity, training=True, recipe=recipe, world_size=2)
    import shutil
    shutil.rmtree(str(path) + '.rng')
    assert read(path, predictor, identity)['metadata']['epoch'] == 1
    with pytest.raises(FileNotFoundError):
        read(path, predictor, identity, training=True, recipe=recipe, world_size=1)


def test_unpublished_rng_generation_does_not_destroy_previous_checkpoint(tmp_path, monkeypatch):
    import smolvlm_map_predictor.checkpoint as checkpoint
    predictor, opt, sched, state, recipe, accelerator = fixture_state()
    path = tmp_path / 'predictor_latest.pt'
    checkpoint.save(path, predictor, opt, sched, state, {}, recipe, accelerator)
    old = path.read_bytes()
    write = checkpoint._atomic_torch
    def interrupted(target, value):
        if target == path:
            raise RuntimeError('crash before publication')
        return write(target, value)
    monkeypatch.setattr(checkpoint, '_atomic_torch', interrupted)
    with pytest.raises(RuntimeError, match='crash'):
        checkpoint.save(path, predictor, opt, sched, dict(state, epoch=2), {}, recipe, accelerator)
    assert path.read_bytes() == old
    assert checkpoint.read(path, predictor, {}, training=True, recipe=recipe, world_size=1)['metadata']['epoch'] == 1


def test_checkpoint_sidecar_counters_tampering_rejected(tmp_path):
    from smolvlm_map_predictor.checkpoint import save, read
    predictor, opt, sched, state, recipe, accelerator = fixture_state()
    path = tmp_path / 'best.pt'
    save(path, predictor, opt, sched, state, {}, recipe, accelerator)
    payload = read(path, predictor, {})
    side = tmp_path / 'best.pt.rng' / payload['metadata']['checkpoint_id'] / 'rank_00000.pt'
    value = torch.load(side, weights_only=False); value['counters']['epoch'] = 900
    torch.save(value, side)
    with pytest.raises(ValueError, match='counters'):
        read(path, predictor, {}, training=True, recipe=recipe, world_size=1)
