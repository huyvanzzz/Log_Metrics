"""CPU smoke with real native attention/generation and Java-backed corpus metrics.

The synthetic model has production layer/token geometry but width 16. Its two-token
generation budget is deliberately a test fixture; production remains 512 tokens.
"""
import importlib.util
import json
import shutil
import zipfile
import numpy as np
import pytest
import torch


def test_real_projected_attention_benchmark_metrics_zip(tmp_path,monkeypatch):
    if not shutil.which('java') or importlib.util.find_spec('pycocoevalcap') is None:
        pytest.skip('Real corpus scoring requires Java and pycocoevalcap')
    from test_smolvlm_model import tiny_components
    from transformers import Idefics3ForConditionalGeneration
    from smolvlm_inference.model import CachedSmolVLM,CAMERA_ORDER
    from smolvlm_inference.cache import record_key
    from smolvlm_predictor_io.projected import ProjectedCache,FORMAT
    from smolvlm_attention.sample_store import atomic_json,digest,fingerprint
    from smolvlm_combined_topk.inputs import prepare_inputs,make_batch_builder
    from smolvlm_combined_topk.projected import ProjectedSmolVLM
    from smolvlm_combined_topk.config import LAYER_IDS
    from smolvlm_attention.combined_collector import collect_combined_visual_attention
    from smolvlm_combined_topk.store import write_run,read_run,ShardStore,finalize_export,open_attention,prepare_selections
    from smolvlm_combined_topk.benchmark import prepare_benchmark,CONFIGS,score_results,bundle_results
    from smolvlm_topk_benchmark.state import read_run as read_benchmark,load_selection,save_batch
    from smolvlm_topk_benchmark.generation import SharedCausalLM,fuse_batch,compact_prompt,generate_compacted,greedy_config
    previous_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        tiny,processor=tiny_components();config=tiny.config
        config.vision_config.image_size=64
        config.text_config.num_hidden_layers=30
        config.text_config.max_position_embeddings=2048
        native=Idefics3ForConditionalGeneration._from_config(config,
            attn_implementation={'vision_config':'eager','text_config':'sdpa'})
        native.generation_config=tiny.generation_config
        model=CachedSmolVLM(native,processor)
        records=[dict(sample_id=f's{i}',eval_id=i,question=q,answer=a,
            camera_paths={camera:f'{camera}/{i}.jpg' for camera in CAMERA_ORDER})
            for i,(q,a) in enumerate((('What is visible?','A road.'),('What changed?','A person.')))]
        prepared=tmp_path/'prepared';prepared.mkdir()
        (prepared/'test.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        cache_dir=tmp_path/'projected';cache_dir.mkdir()
        with torch.inference_mode():
            values=model._project(torch.randn(2,6,256,8),2).reshape(2,6,64,16).numpy()
        np.save(cache_dir/'projected.npy',values)
        atomic_json(cache_dir/'index.json',{record_key(r):i for i,r in enumerate(records)})
        manifest=dict(format=FORMAT,complete=True,identity={'synthetic_width':16},shape=list(values.shape),dtype='<f4',
            file_bytes=(cache_dir/'projected.npy').stat().st_size,
            payload_sha256=digest(cache_dir/'projected.npy'),index_sha256=digest(cache_dir/'index.json'))
        manifest['fingerprint']=fingerprint(manifest);atomic_json(cache_dir/'manifest.json',manifest)
        cache=ProjectedCache(cache_dir)
        prepared_inputs=prepare_inputs(model,cache,prepared,tmp_path/'inputs',{'synthetic':True},expected_count=2)
        builder=make_batch_builder(prepared_inputs['store'])
        adapter=ProjectedSmolVLM.from_model(model)
        def forbidden(*args,**kwargs):raise AssertionError('Projected pipeline called connector')
        monkeypatch.setattr(native.model.connector,'forward',forbidden)
        scores=collect_combined_visual_attention(adapter,builder(prepared_inputs['rows']),LAYER_IDS,8).numpy()
        assert scores.shape==(2,10,384) and np.isfinite(scores).all()
        checkpoint=tmp_path/'fixture.pt';checkpoint.write_bytes(b'synthetic checkpoint is never reloaded')
        receipt=tmp_path/'fixture_receipt.json';receipt.write_text('{}')
        run_path=write_run(tmp_path/'output',dict(identity=dict(policy='integration',checkpoint_sha256=digest(checkpoint),
            projected_fingerprint=manifest['fingerprint'],camera_order=CAMERA_ORDER),
            camera_order=CAMERA_ORDER,geometry=[10,384],layer_ids=list(LAYER_IDS),batch_size=64,shard_count=2,
            store=prepared_inputs['store'],projected_receipt=str(receipt),artifacts={'model':{'checkpoint':str(checkpoint)}}),prepared_inputs['rows'])
        run=read_run(run_path)
        for shard in range(2):
            store=ShardStore(tmp_path/'output'/'shards'/str(shard),run['fingerprint'],[shard])
            store.write([shard],scores[shard:shard+1]);store.commit()
        exported=finalize_export(run_path,progress=False)
        np.testing.assert_array_equal(open_attention(exported).attention,scores)
        selections=prepare_selections(exported)
        benchmark_path=prepare_benchmark(run_path,exported,selections)
        benchmark=read_benchmark(benchmark_path)
        causal=SharedCausalLM(native);generation=greedy_config(adapter,2)
        seen_logits=[]
        hook=causal.lm_head.register_forward_pre_hook(lambda module,args:seen_logits.append(args[0].shape[1]))
        try:
            for name in CONFIGS:
                keep=load_selection(benchmark,name)[0].tolist()
                compact=compact_prompt(fuse_batch(adapter,builder(prepared_inputs['rows'])),keep)
                assert ((compact['input_ids']==native.config.image_token_id)&compact['attention_mask'].bool()).sum(-1).tolist()==[32,32]
                generated=generate_compacted(causal,compact,processor.tokenizer,generation)
                predictions=[dict(g,sample_id=r['sample_id'],eval_id=r['eval_id'],reference=r['answer'])
                    for g,r in zip(generated,prepared_inputs['rows'])]
                save_batch(benchmark,name,0,predictions)
        finally:hook.remove()
        # GenerationMixin must request only the final-token LM-head projection.
        assert seen_logits and set(seen_logits)=={1}
        summary=score_results(benchmark_path)
        assert len(summary['results'])==10
        assert all(r['count']==2 for r in summary['results'])
        archive=bundle_results(benchmark_path,exported)
        with zipfile.ZipFile(archive) as zipped:
            names=zipped.namelist()
            assert 'attention/attention.npy' in names and 'benchmark/summary.csv' in names
            assert all(i.compress_type==zipfile.ZIP_STORED for i in zipped.infolist())
            assert not any('projected.npy' in name or '/shards/' in name or name.endswith('.pt') for name in names)
    finally:torch.set_num_threads(previous_threads)
