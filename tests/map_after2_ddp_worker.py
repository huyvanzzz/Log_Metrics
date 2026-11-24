"""Subprocess entry point for the two-rank CPU lifecycle integration test."""
import json
import os
from pathlib import Path
import sys
import torch


def main():
    from after2_support import tiny_config
    from smolvlm_map_predictor_after2.training import train
    root = Path(sys.argv[1]); cfg = tiny_config()
    if os.name == 'nt':
        torch.distributed.init_process_group('gloo', init_method=(root/'rendezvous').as_uri(),
            rank=int(os.environ['RANK']), world_size=2)
    cfg['training'].update(world_size=2, batch_size=2, num_workers=0, pin_memory=False, epochs=2)
    model = torch.load(root/'fixture_model.pt', map_location='cpu', weights_only=False)
    receipt = root/'after2_inputs'/'after2_receipt.json'
    full = root/'full'; resumed = root/'resumed'
    train(cfg, receipt, None, full, cpu=True, backbone=model)
    train(cfg, receipt, None, resumed, cpu=True, backbone=model, stop_after_epoch=1)
    state = train(cfg, receipt, None, resumed, cpu=True, backbone=model,
                  resume=resumed/'checkpoints'/'predictor_latest.pt')
    from smolvlm_map_predictor_after2.training import accelerator_for, make_loader, make_predictor, move_batch
    from smolvlm_map_predictor_after2.model import kl_per_sample
    from smolvlm_map_predictor_after2.prefix import frozen_features
    accelerator = accelerator_for(cfg, cpu=True)
    torch.manual_seed(17)
    predictor = make_predictor(model, cfg)
    optimizer = torch.optim.AdamW(predictor.parameters())
    loader = make_loader(receipt, model, cfg, 'train')
    predictor, optimizer, loader = accelerator.prepare(predictor, optimizer, loader)
    batch = move_batch(next(iter(loader)), accelerator.device)
    loss = kl_per_sample(predictor(**frozen_features(model, batch)), batch['log_target']).mean()
    accelerator.backward(loss)
    gradient = torch.cat([p.grad.flatten() for p in accelerator.unwrap_model(predictor).parameters()])
    gradients = [torch.empty_like(gradient) for _ in range(2)]
    torch.distributed.all_gather(gradients, gradient)
    assert torch.equal(gradients[0], gradients[1])
    optimizer.step()
    weights = torch.cat([p.detach().flatten() for p in accelerator.unwrap_model(predictor).parameters()])
    all_weights = [torch.empty_like(weights) for _ in range(2)]
    torch.distributed.all_gather(all_weights, weights)
    assert torch.equal(all_weights[0], all_weights[1])
    sample_ids = [None, None]
    torch.distributed.all_gather_object(sample_ids, batch['sample_ids'])
    assert len(sum(sample_ids, [])) == 4 and len(set(sum(sample_ids, []))) == 3
    val_loader = make_loader(receipt, model, cfg, 'val', rank=accelerator.process_index, world_size=2)
    val_ids = [sample for item in val_loader for sample in item['sample_ids']]
    validation_ids = [None,None]
    torch.distributed.all_gather_object(validation_ids,val_ids)
    assert len(sum(validation_ids,[])) == len(set(sum(validation_ids,[]))) == 3
    assert all(parameter.grad is None for parameter in model.parameters())
    state.update(validation_ids=validation_ids,gradient_ranks_equal=True, weights_ranks_equal=True, sample_ids=sample_ids)
    (root/f'worker_{os.environ["RANK"]}.json').write_text(json.dumps(state))


if __name__ == '__main__':
    main()

