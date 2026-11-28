from smolvlm_combined_topk.benchmark_worker import schedule_batches, worker_configs


def test_assignment_and_config_outer_schedule():
    assert worker_configs(0) == ('layer_10','layer_15','layer_16','layer_17','layer_21')
    assert worker_configs(1) == ('layer_22','layer_23','layer_25','layer_26','layer_27')
    rows=[{'sample_id':str(i)} for i in range(65)]
    events=[]; done={'layer_10':set(),'layer_15':set()}
    schedule_batches(rows,tuple(done),done,lambda x:x,
        lambda name,selected,shared: events.append((name,len(selected))) or selected,
        lambda *args:None)
    assert events == [('layer_10',64),('layer_10',1),('layer_15',64),('layer_15',1)]


def test_fixed_batch_failure_does_not_commit():
    import pytest
    done={'layer_10':set()}
    with pytest.raises(RuntimeError,match='no retry/reduction'):
        schedule_batches([{'sample_id':'a'}],tuple(done),done,lambda x:x,
            lambda *args: (_ for _ in ()).throw(RuntimeError('OOM')),
            lambda *args:None)
    assert done['layer_10'] == set()


def _complete_fixture(tmp_path):
    import numpy as np
    from smolvlm_combined_topk.store import write_run,read_run,ShardStore,finalize_export
    from smolvlm_combined_topk.benchmark import prepare_benchmark
    from smolvlm_topk_benchmark.state import read_run as read_benchmark, save_batch, input_rows
    from smolvlm_combined_topk.config import LAYER_IDS
    rows=[dict(row_index=0,sample_id='a',eval_id=0,question='What?',answer='Red',
        input_ids=[9]*384+[7],image_positions=list(range(384)),image_token_id=9,
        prompt_length=385,query_position=384,query_token_id=7)]
    compact=tmp_path/'compact.json'; compact.write_text(__import__('json').dumps(dict(
        source_identity='source',counts={'test':1},geometry={},max_length=8192,pad_token_id=0,splits={})))
    config=write_run(tmp_path,dict(identity={'checkpoint_sha256':'fake','projected_fingerprint':'projected'},geometry=[10,384],
        layer_ids=list(LAYER_IDS),batch_size=64,shard_count=1,projected_receipt='receipt',store=str(compact),
        artifacts={'model':{'checkpoint':'checkpoint'}}),rows)
    run=read_run(config); store=ShardStore(tmp_path/'shards'/'0',run['fingerprint'],[0])
    store.write([0],np.full((1,10,384),.001,dtype=np.float32));store.commit()
    export=finalize_export(config,progress=False)
    bench=prepare_benchmark(run,export); benchmark=read_benchmark(bench)
    record=dict(sample_id='a',eval_id=0,reference='Red',raw_prediction='Red',prediction='Red',
        generated_ids=[1],generated_token_count=1,stop_reason='eos')
    from smolvlm_combined_topk.benchmark import CONFIGS
    for name in CONFIGS: save_batch(benchmark,name,0,[record])
    return bench,export


def test_prepare_select_score_and_bundle(tmp_path):
    import json,zipfile
    import numpy as np
    from smolvlm_combined_topk.benchmark import score_results,bundle_results,CONFIGS
    from smolvlm_topk_benchmark.state import read_run,load_selection
    from smolvlm_topk_benchmark.scoring import METRICS
    bench,export=_complete_fixture(tmp_path)
    assert load_selection(read_run(bench),'layer_10')[0][0].tolist()==list(range(32))
    identity=dict(pycocoevalcap_version='1.2',java_version='test',meteor_jar_sha256='jar',
        scorer_sources={'x':{'sha256':'x'}},tokenization='direct',meteor_sanitization='minimal')
    calls=[]
    def scorer(output,**kwargs):
        from smolvlm_attention.sample_store import atomic_json
        calls.append(kwargs)
        result=dict(scoring_identity=identity,metrics={m:.5 for m in METRICS},diagnostics={'empty_prediction_count':0}) if kwargs else identity
        atomic_json(output,result); return result
    summary=score_results(bench,scorer_command=scorer)
    assert len(summary['results'])==10 and len(calls)==11
    assert 'delta_vs_base' not in summary['results'][0]
    score_results(bench,scorer_command=scorer)
    assert len(calls)==12 # only scorer identity preflight; cached full corpus scores
    archive=bundle_results(bench,export)
    with zipfile.ZipFile(archive) as zipped:
        names=zipped.namelist()
        assert 'attention/attention.npy' in names
        assert all(info.compress_type==zipfile.ZIP_STORED for info in zipped.infolist())
        assert len([n for n in names if n.endswith('/predictions.jsonl')])==10
        assert not any('shards' in n or 'projected.npy' in n for n in names)
        zipped.extractall(tmp_path/'extracted')
    extracted=tmp_path/'extracted'/'benchmark'/'benchmark_run.json'
    score_results(extracted,scorer_command=scorer)
    assert len(calls)==13 # completed ZIP can re-score without batch journals or cache
    changed=json.loads((bench.parent/'summary.json').read_text())
    changed['results'][0]['BLEU-1']=.9
    (bench.parent/'summary.json').write_text(json.dumps(changed))
    import pytest
    with pytest.raises(ValueError,match='Summary metrics'): bundle_results(bench,export)


def test_incomplete_scoring_and_modified_selection_fail(tmp_path):
    import pytest
    from smolvlm_combined_topk.benchmark import score_results
    bench,export=_complete_fixture(tmp_path)
    (bench.parent/'configs'/'layer_27'/'batches'/'000000.json').unlink()
    with pytest.raises(ValueError,match='incomplete'):
        score_results(bench,scorer_command=lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError('scorer must not run')))


def test_generation_fingerprint_excludes_scoring_and_detects_worker(monkeypatch):
    from smolvlm_combined_topk import benchmark as module
    import pytest
    run={'identity':dict(generation_sources=module.generation_sources(),generation_versions=module.generation_versions())}
    assert not any(name.endswith('benchmark.py') or name.endswith('scoring.py') for name in run['identity']['generation_sources'])
    module.assert_generation_compatible(run)
    monkeypatch.setattr(module,'generation_sources',lambda:{'worker':'changed'})
    with pytest.raises(ValueError,match='Generation code changed'): module.assert_generation_compatible(run)


def test_compact_store_fingerprint_ignores_relocation_and_detects_payload(tmp_path):
    import json
    from smolvlm_combined_topk.benchmark import input_store_fingerprint
    first=tmp_path/'one'; first.mkdir(); payload=first/'tokens.bin'; payload.write_bytes(b'abc')
    store=first/'store.json'; value=dict(source_identity='source',counts={'test':1},geometry={'text_dim':576},
        max_length=8192,pad_token_id=0,splits={'test':{'tokens':str(payload)}},files={'ignored':'stats'})
    store.write_text(json.dumps(value)); expected=input_store_fingerprint(store)
    second=tmp_path/'two'; second.mkdir(); other=second/'tokens.bin'; other.write_bytes(b'abc')
    value['splits']['test']['tokens']=str(other); value['files']={}
    relocated=second/'store.json'; relocated.write_text(json.dumps(value))
    assert input_store_fingerprint(relocated)==expected
    other.write_bytes(b'abd'); assert input_store_fingerprint(relocated)!=expected
