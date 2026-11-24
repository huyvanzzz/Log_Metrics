import json
import zipfile

import torch


def test_after2_real_offline_lifecycle_to_bundle_and_target_free_test(tmp_path):
    from after2_support import make_inputs
    from after2_support import tiny_config as tiny_recipe
    from smolvlm_map_predictor_after2.diagnostics import diagnose
    from smolvlm_map_predictor_after2.training import train, make_predictor
    from smolvlm_map_predictor_after2.evaluation import evaluate_worker, finalize_evaluation
    from smolvlm_map_predictor_after2.runtime import read_run, sources, versions, bundle
    from smolvlm_attention.sample_store import atomic_json, digest
    from smolvlm_map_predictor_after2 import checkpoint
    inputs = tmp_path/'inputs'; inputs.mkdir()
    model, receipt = make_inputs(inputs)
    cfg = tiny_recipe()
    cfg['evaluation'].update(batch_size=2, max_new_tokens=2, keep_tokens=32)
    out = tmp_path/'output'
    identity = json.loads(receipt.read_text(encoding='utf-8'))['identity']
    run = dict(version=1, receipt=str(receipt), receipt_sha256=digest(receipt), identity=identity,
        config=cfg, output_dir=str(out), backbone_checkpoint=str(inputs/'base_fixture.pt'),
        execution=dict(sources=sources(),versions=versions()), _evaluation_world_size=1,
        _evaluation_session='offline-lifecycle')
    run_path = out/'run.json'; atomic_json(run_path,run)
    assert read_run(run_path)['identity'] == identity
    diagnose(cfg, receipt, None, out, cpu=True, backbone=model)
    train(cfg, receipt, None, out, cpu=True, backbone=model, stop_after_epoch=1)
    latest = out/'checkpoints/predictor_latest.pt'
    result = train(cfg, receipt, None, out, cpu=True, backbone=model, resume=latest)
    assert result['completed_epochs'] == 2
    best = out/'checkpoints/predictor_best.pt'
    predictor = make_predictor(model,cfg)
    payload = checkpoint.read(best,predictor,identity)
    assert payload['metadata']['best_epoch'] in (1,2)
    # Inference and bundling must remain independent of teacher Input files.
    for path in (inputs/'targets').rglob('*'):
        if path.is_file(): path.unlink()
    evaluate_worker(run,0,best,device='cpu',backbone=model)
    final = finalize_evaluation(run,best)
    assert final['diagnostics']['prediction_count'] == 3
    archive = bundle(run_path)
    with zipfile.ZipFile(archive) as stream:
        names = stream.namelist()
        assert 'checkpoints/predictor_best.pt' in names
        assert any(name.startswith('checkpoints/predictor_latest.pt.rng/') for name in names)
        assert 'predictions.jsonl' in names and 'metrics.json' in names
        assert not any(name.startswith('internal/') or 'memmap' in name or 'log_probs.npy' in name
                       or 'base_fixture.pt' in name for name in names)
    assert json.loads(archive.with_suffix('.sha256.json').read_text())['sha256'] == digest(archive)
