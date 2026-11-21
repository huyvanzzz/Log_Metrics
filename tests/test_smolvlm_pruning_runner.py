import json
import pytest
from smolvlm_inference import pruning_run as run
from test_smolvlm_cache import fixture_cache


def test_configs_are_fixed():
    for index in (2, 15):
        config = run.load_config(f'configs/mvpruner_layer_index_{index}.json')
        assert config['stage2_layer_index'] == index
        assert config['batch_size'] == 64
        assert config['stage1_keep'] == 196 and config['stage2_keep'] == 32
    with pytest.raises(ValueError):
        run.validate_config(dict(config, batch_size=32))


def test_receipt_rejects_changed_payload(tmp_path):
    cache, prepared, identity, _ = fixture_cache(tmp_path)
    receipt = tmp_path / 'receipt.json'
    run.validate_cache(cache, prepared, identity, receipt)
    run.open_validated_cache(cache, prepared, identity, receipt)
    payload = cache / 'features.float16.memmap'
    payload.write_bytes(payload.read_bytes())
    with pytest.raises(ValueError, match='receipt'):
        run.open_validated_cache(cache, prepared, identity, receipt)


def test_coverage_rejects_duplicate_and_missing():
    expected = [dict(eval_id=0, sample_id='a'), dict(eval_id=1, sample_id='b')]
    run.validate_coverage(expected, expected)
    for rows in (expected[:1], [expected[0], expected[0]]):
        with pytest.raises(ValueError):
            run.validate_coverage(rows, expected)


def test_metric_failure_preserves_predictions(tmp_path, monkeypatch):
    path = tmp_path / 'predictions.jsonl'
    content = json.dumps(dict(eval_id=1, reference=' A ', prediction=' B '))+'\n'
    path.write_text(content)
    def fail(rows):
        raise RuntimeError('metric unavailable')
    monkeypatch.setattr(run, 'caption_metrics', fail)
    with pytest.raises(RuntimeError):
        run.score_predictions(path)
    assert path.read_text() == content


def test_notebook_orders_validation_before_launch_and_pins_inputs():
    from pathlib import Path
    notebook = json.loads(Path('notebooks/kaggle_smolvlm_mvpruner_test.ipynb').read_text())
    source = '\n'.join(''.join(c['source']) for c in notebook['cells'])
    assert source.index('preflight_metrics()') < source.index("CHECKPOINT = hf_hub_download")
    assert source.index('cache = validate_cache') < source.index("'launch',")
    assert run.CHECKPOINT_REVISION in source and run.PREPARED_REVISION in source
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code': compile(''.join(cell['source']), '<notebook>', 'exec')


def test_launch_runs_stress_then_test_and_sequential_metrics(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(run, 'preflight_metrics', lambda: calls.append('preflight'))
    monkeypatch.setattr(run, 'launch_pair', lambda args, phase: calls.append(phase))
    def score(path):
        calls.append(str(path.parent.name))
        path.parent.mkdir(parents=True)
        path.write_text('{}\n')
    monkeypatch.setattr(run, 'score_predictions', score)
    run.main(['launch','--output',str(tmp_path),'--cache','cache','--prepared','prepared','--receipt','receipt','--checkpoint','checkpoint'])
    assert calls == ['preflight','stress','test','layer_index_2','layer_index_15']
    import zipfile
    with zipfile.ZipFile(tmp_path/'mvpruner_artifacts.zip') as archive:
        assert set(archive.namelist()) == {'layer_index_2/predictions.jsonl','layer_index_15/predictions.jsonl'}


def test_real_tiny_engine_predictions_keep_coverage_and_strip_only(tmp_path):
    from test_smolvlm_pruned_decoder import fixture
    from smolvlm_inference.decoder import PrunedGreedyDecoder
    model, batch = fixture()
    decoder = PrunedGreedyDecoder(model, 2, stage1_keep=18, stage2_keep=12)
    result = decoder.generate(batch, max_new_tokens=4, debug=True)
    rows = [dict(eval_id=i, sample_id=str(i), question=' Q ', answer=' Reference ') for i in range(2)]
    output = run.prediction_rows(model.processor.tokenizer, result, rows)
    run.validate_coverage(output, rows)
    assert [r['reference'] for r in output] == ['Reference','Reference']
    assert [r['question'] for r in output] == [' Q ',' Q ']
    assert all(r['prediction'] == r['prediction'].strip() for r in output)


def test_cpu_evaluation_lifecycle_real_engine(tmp_path):
    import torch
    from test_smolvlm_pruned_decoder import fixture
    from smolvlm_inference.decoder import PrunedGreedyDecoder
    from smolvlm_inference.pruning_text import PruningBatchCollator
    from smolvlm_inference.cache import record_key
    import numpy as np
    model, batch = fixture()
    rows = [dict(eval_id=i,sample_id=str(i),question=q,answer=' Reference ',
                 camera_paths={c:f'{i}/{c}.jpg' for c in run.CAMERA_ORDER})
            for i,q in enumerate(('What is visible?','What changed? A person.','Where is the car?'))]
    class Reader:
        index = {record_key(row): i for i,row in enumerate(rows)}
        array = np.random.default_rng(0).normal(size=(3,6,16,8)).astype(np.float16)
    collator = PruningBatchCollator(model.processor,image_seq_len=4,max_length=2048,max_new_tokens=4)
    decoder = PrunedGreedyDecoder(model,2,stage1_keep=18,stage2_keep=12)
    run.evaluate_batches(model,Reader(),rows,collator,decoder,tmp_path,'test',device='cpu',max_new_tokens=4,batch_size=2)
    output = list(run.records(tmp_path/'predictions.jsonl'))
    run.validate_coverage(output,rows)
    assert len(output) == 3 and (tmp_path/'debug.json').is_file()
    progress=list(run.records(tmp_path/'progress.jsonl'))
    assert progress[0]['samples_per_second'] > 0
    assert [p['completed'] for p in progress] == [2,3]
    assert progress[-1]['estimated_remaining_seconds'] == 0


def test_missing_required_arguments_are_parser_errors():
    with pytest.raises(SystemExit) as caught:
        run.main(['test'])
    assert caught.value.code == 2


def test_cpu_memory_summary_has_complete_fields():
    value = run.memory_summary('cpu')
    assert set(value) == {'allocated','reserved','peak_allocated','peak_reserved','free','total','gpu_name'}


def test_validation_progress_is_throttled(monkeypatch):
    seen=[]
    clock=iter([0,1,2,3,6,6])
    callback=run.validation_progress(lambda **e:seen.append(e),clock=lambda:next(clock))
    for completed in (0,1,2,3,4,10):
        callback(dict(stage='features',completed=completed,total=10))
    assert [e['completed'] for e in seen] == [0,4,10]


def test_metrics_write_source_display_scale(tmp_path,monkeypatch):
    path=tmp_path/'predictions.jsonl'
    path.write_text('{"eval_id":0,"reference":"car","prediction":"car"}\n')
    monkeypatch.setattr(run,'caption_metrics',lambda rows:{name:(0.123456789 if name=='BLEU-1' else 0.2) for name in run.METRIC_NAMES})
    run.score_predictions(path)
    assert json.loads((tmp_path/'metrics_display.json').read_text()) == {name:(12.345679 if name=='BLEU-1' else 20.0) for name in run.METRIC_NAMES}


def test_nonfinite_metric_rejected_without_artifacts(tmp_path,monkeypatch):
    path=tmp_path/'predictions.jsonl'
    content='{"eval_id":0,"reference":"car","prediction":"car"}\n'
    path.write_text(content)
    monkeypatch.setattr(run,'caption_metrics',lambda rows:{'BLEU-1':float('nan')})
    with pytest.raises(ValueError,match='metric'):
        run.score_predictions(path)
    assert path.read_text()==content
    assert not (tmp_path/'metrics.json').exists()


def test_notebook_allocator_before_torch_and_display_table():
    from pathlib import Path
    nb=json.loads(Path('notebooks/kaggle_smolvlm_mvpruner_test.ipynb').read_text())
    source='\n'.join(''.join(c['source']) for c in nb['cells'])
    for name in ('PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF','USE_TF','USE_FLAX'):
        assert source.index(name)<source.index('import torch')
    assert 'metrics_display.json' in source
    assert 'Markdown' in source and '| Metric |' in source
