import json
from pathlib import Path

import pytest
import torch


def _independent_evaluation_worker(run, rank, checkpoint, model_path):
    from smolvlm_map_predictor.evaluation import evaluate_worker
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    evaluate_worker(run, rank, checkpoint, device='cpu', backbone=model)


def test_predictor_fusion_matches_training_inputs_and_generates_once():
    from test_smolvlm_pruned_decoder import fixture
    from smolvlm_map_predictor.model import MapPredictor, frozen_features, topk_indices
    from smolvlm_map_predictor.evaluation import predictor_inputs, predict_batch
    from smolvlm_topk_benchmark.generation import (
        fuse_batch, compact_prompt, SharedCausalLM, generate_compacted, greedy_config,
    )
    model, batch = fixture(layers=2)
    predictor = MapPredictor(model.native.config.text_config, hidden_dim=8, projection_dim=4, visual_tokens=24).eval()
    expected_inputs = frozen_features(model, batch)
    fused = fuse_batch(model, batch)
    actual_inputs = predictor_inputs(fused, batch['question_mask'])
    for key in expected_inputs:
        torch.testing.assert_close(actual_inputs[key], expected_inputs[key], rtol=0, atol=0)
    with torch.inference_mode():
        selection = topk_indices(predictor(**actual_inputs), 4).tolist()
        reference = generate_compacted(SharedCausalLM(model.native), compact_prompt(fused, selection),
                                       model.processor.tokenizer, greedy_config(model, 3))
    connector, calls = [], []
    handles = [model.native.model.connector.register_forward_hook(lambda *args: connector.append(1)),
               predictor.register_forward_hook(lambda *args: calls.append(1))]
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    try:
        result = predict_batch(model, predictor, batch, keep_tokens=4, max_new_tokens=3)
    finally:
        for handle in handles:
            handle.remove()
    assert connector == [1] and calls == [1]
    assert [row['visual_indices'] for row in result] == selection
    assert [{key: value for key, value in row.items() if key != 'visual_indices'} for row in result] == reference
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(value, weights[key]) for key, value in model.state_dict().items())
    all_keep = predict_batch(model, predictor, batch, keep_tokens=24, max_new_tokens=3)
    native = model.generate_from_features(**{key: batch[key] for key in
        ('input_ids', 'attention_mask', 'visual_features')}, generation_config=greedy_config(model, 3))
    from smolvlm_topk_benchmark.generation import decode_sequences
    expected = decode_sequences(model.processor.tokenizer, native,
        model.native.generation_config.eos_token_id, 3)
    assert [row['generated_ids'] for row in all_keep] == [row['generated_ids'] for row in expected]


def _evaluation_fixture(tmp_path):
    from map_support import make_inputs
    from smolvlm_map_predictor.training import make_predictor
    from smolvlm_map_predictor.config import load_config
    model, receipt = make_inputs(tmp_path)
    identity = json.loads(receipt.read_text(encoding='utf-8'))['identity']
    config = load_config()
    config['architecture'] = dict(hidden_dim=8, projection_dim=4, visual_tokens=384)
    config['training'].update(num_workers=0, pin_memory=False)
    config['evaluation'].update(batch_size=2, keep_tokens=4, max_new_tokens=2)
    predictor = make_predictor(model, config)
    path = tmp_path / 'predictor_best.pt'
    torch.save(dict(format='map_predictor_checkpoint_v1', identity=identity,
        architecture_signature=predictor.architecture_signature, model_state_dict=predictor.state_dict()), path)
    run = dict(receipt=str(receipt), identity=identity, config=config, output_dir=str(tmp_path/'output'),
               _evaluation_world_size=1, _evaluation_session='cpu-fixture')
    return model, path, run


def test_real_evaluation_resume_without_targets_and_tamper_rejection(tmp_path):
    from smolvlm_map_predictor.evaluation import evaluate_worker
    model, path, run = _evaluation_fixture(tmp_path)
    # Inference uses predictor weights only and must not touch targets or RNG sidecars.
    for target in (tmp_path/'targets').rglob('*'):
        if target.is_file():
            target.unlink()
    first = evaluate_worker(run, 0, path, device='cpu', backbone=model)
    assert first == dict(rank=0, count=3, resumed_batches=0)
    events = [json.loads(line) for line in (Path(run['output_dir'])/'debug'/'evaluate_rank_0.jsonl').read_text().splitlines()]
    updates = [row for row in events if row['event'] == 'evaluation_batch']
    assert [row['newly_processed_samples'] for row in updates] == [2, 3]
    assert updates[-1]['estimated_remaining_seconds'] == 0
    assert all(row['samples_per_second'] > 0 and row['resumed_samples'] == 0 for row in updates)
    shard = Path(run['output_dir'])/'evaluation'/'rank_0'
    assert len(json.loads((shard/'batch_000001.json').read_text())['records']) == 1
    hashes = {p.name: p.read_bytes() for p in shard.glob('batch_*.json')}
    second = evaluate_worker(run, 0, path, device='cpu', backbone=model)
    assert second['resumed_batches'] == 2
    assert all(p.read_bytes() == hashes[p.name] for p in shard.glob('batch_*.json'))
    damaged = shard/'batch_000000.json'
    data = json.loads(damaged.read_text()); data['records'][0]['prediction'] = 'changed'
    damaged.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='checksum'):
        evaluate_worker(run, 0, path, device='cpu', backbone=model)


def test_evaluation_requires_fresh_session_and_rejects_changed_predictor(tmp_path):
    from smolvlm_map_predictor.evaluation import evaluate_worker
    model, path, run = _evaluation_fixture(tmp_path)
    run.pop('_evaluation_session')
    with pytest.raises(ValueError, match='fresh launcher'):
        evaluate_worker(run, 0, path, device='cpu', backbone=model)
    run['_evaluation_session'] = 'valid'
    evaluate_worker(run, 0, path, device='cpu', backbone=model)
    data = torch.load(path, weights_only=False)
    key = next(iter(data['model_state_dict']))
    data['model_state_dict'][key].add_(1)
    torch.save(data, path)
    with pytest.raises(ValueError, match='another predictor'):
        evaluate_worker(run, 0, path, device='cpu', backbone=model)


def test_two_independent_processes_exact_coverage_and_real_metrics(tmp_path):
    import multiprocessing as mp
    from smolvlm_map_predictor.evaluation import finalize_evaluation
    model, checkpoint, run = _evaluation_fixture(tmp_path)
    run['_evaluation_world_size'] = 2
    model_path = tmp_path/'native_fixture.pt'
    torch.save(model, model_path)
    context = mp.get_context('spawn')
    processes = [context.Process(target=_independent_evaluation_worker,
        args=(run, rank, checkpoint, model_path)) for rank in (0, 1)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=60)
        assert [process.exitcode for process in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate(); process.join(timeout=10)
    root = Path(run['output_dir'])
    shards = [json.loads((root/'evaluation'/f'rank_{rank}'/'batch_000000.json').read_text()) for rank in (0, 1)]
    assert shards[0]['rows'] == [0, 2] and shards[1]['rows'] == [1]
    summary = finalize_evaluation(run, checkpoint)
    assert summary['diagnostics']['prediction_count'] == 3
    records = [json.loads(line) for line in (root/'predictions.jsonl').read_text().splitlines()]
    assert [row['prepared_row'] for row in records] == [0, 1, 2]
    assert set(json.loads((root/'metrics.json').read_text())) == {
        'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr'}
    marker = root/'evaluation'/'rank_1'/'complete.json'
    payload = json.loads(marker.read_text()); payload['count'] += 1
    marker.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='coverage'):
        finalize_evaluation(run, checkpoint)
