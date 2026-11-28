import json
import numpy as np
import pytest
import torch
from test_attention_collector import fixture


def test_projected_adapter_matches_and_never_connects(monkeypatch):
    from smolvlm_combined_topk.projected import ProjectedSmolVLM
    from smolvlm_attention.combined_collector import collect_combined_visual_attention
    from smolvlm_topk_benchmark.generation import fuse_batch
    model,batch,_,_=fixture(questions=('What?', 'What is visible?'))
    expected=collect_combined_visual_attention(model,batch,(1,0))
    fused=fuse_batch(model,batch)
    projected=model._project(batch['visual_features'],2).reshape(2,model.visual_token_len,-1)
    adapter=ProjectedSmolVLM.from_model(model)
    def forbidden(*a,**k): raise AssertionError('connector called')
    monkeypatch.setattr(model.native.model.connector,'forward',forbidden)
    batch=dict(batch,visual_features=projected)
    torch.testing.assert_close(collect_combined_visual_attention(adapter,batch,(1,0)),expected,atol=1e-6,rtol=1e-4)
    torch.testing.assert_close(fuse_batch(adapter,batch)['inputs_embeds'],fused['inputs_embeds'])
    assert adapter.native is model.native and adapter.architecture_signature==model.architecture_signature
    with pytest.raises(ValueError): adapter._project(projected.half(),2)
    with pytest.raises(ValueError): adapter._project(projected[:,:-1],2)


def test_projected_generation_native_and_cached_preflight(monkeypatch):
    from smolvlm_combined_topk.projected import ProjectedSmolVLM
    from smolvlm_topk_benchmark.generation import fuse_batch,compact_prompt,generate_compacted,SharedCausalLM,greedy_config
    model,batch,_,_=fixture()
    expected=generate_compacted(SharedCausalLM(model.native),compact_prompt(fuse_batch(model,batch),[None]),model.processor.tokenizer,greedy_config(model,3))
    projected=model._project(batch['visual_features'],1).reshape(1,model.visual_token_len,-1)
    adapter=ProjectedSmolVLM.from_model(model)
    monkeypatch.setattr(model.native.model.connector,'forward',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('connector called')))
    optimized=dict(batch,visual_features=projected)
    actual=generate_compacted(SharedCausalLM(adapter.native),compact_prompt(fuse_batch(adapter,optimized),[None]),adapter.processor.tokenizer,greedy_config(adapter,3))
    assert actual==expected


def test_discovery_complete_unique_only(tmp_path):
    from smolvlm_combined_topk.projected import discover_projected_cache
    root=tmp_path/'input'; root.mkdir()
    with pytest.raises(FileNotFoundError): discover_projected_cache(search_roots=[root])
    for name in ('a','b'):
        directory=root/name; directory.mkdir()
        (directory/'manifest.json').write_text(json.dumps({'format':'smolvlm_projected_visual_fp32_v1','complete':True}))
        (directory/'projected.npy').write_bytes(b'x'); (directory/'index.json').write_text('{}')
        if name=='a': assert discover_projected_cache(search_roots=[root])==directory.resolve()
    with pytest.raises(ValueError,match='multiple'): discover_projected_cache(search_roots=[root])
    assert discover_projected_cache(root/'a')==(root/'a').resolve()


def test_inputs_preserve_source_and_tokenize_only_once(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from smolvlm_combined_topk.inputs import prepare_inputs,make_batch_builder
    from smolvlm_inference.cache import record_key
    from smolvlm_inference.model import CAMERA_ORDER
    model,_,_,_=fixture()
    prepared=tmp_path/'prepared'; prepared.mkdir(); projected=tmp_path/'cache'; projected.mkdir()
    rows=[dict(sample_id=f's{i}',eval_id=i,question=q,answer='A road.',camera_paths={c:f'{c}/{i}.jpg' for c in CAMERA_ORDER}) for i,q in enumerate(('What?','What is visible?'))]
    (prepared/'test.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    np.save(projected/'projected.npy',np.zeros((2,6,4,16),dtype='<f4'))
    (projected/'manifest.json').write_text(json.dumps({'shape':[2,6,4,16]}))
    cache=SimpleNamespace(directory=projected,index={record_key(r):i for i,r in enumerate(rows)})
    result=prepare_inputs(model,cache,prepared,tmp_path/'inputs',{'fixture':True},expected_count=2)
    assert [r['eval_id'] for r in result['rows']]==[0,1]
    for original,mapping in zip(rows,result['rows']):
        assert all(mapping[k]==v for k,v in original.items())
        assert mapping['query_position']==len(mapping['input_ids'])-1 and len(mapping['image_positions'])==24
    def forbidden(*a,**kw): raise AssertionError('tokenized again')
    from smolvlm_inference.pruning_text import PruningBatchCollator
    monkeypatch.setattr(PruningBatchCollator,'_encode_question',forbidden)
    batch=make_batch_builder(result['store'])(result['rows'])
    assert batch['visual_features'].shape==(2,24,16)
    assert batch['eval_ids']==[0,1]
    second=prepare_inputs(model,cache,prepared,tmp_path/'inputs',{'fixture':True},expected_count=2)
    assert second['rows']==result['rows']
    identity=json.loads((tmp_path/'inputs'/'input_identity.json').read_text())
    assert 'compact/test.tokens.bin' in identity['compact_content_sha256']
    with (tmp_path/'inputs'/'compact'/'test.tokens.bin').open('ab') as stream: stream.write(b'bad')
    with pytest.raises(ValueError,match='content changed'):
        prepare_inputs(model,cache,prepared,tmp_path/'inputs',{'fixture':True},expected_count=2)


def test_projected_receipt_checks_origin_and_changed_payload(tmp_path):
    from types import SimpleNamespace
    from smolvlm_combined_topk.projected import validate_projected_cache,open_projected_receipt
    from smolvlm_predictor_io.projected import FORMAT,GROUPED_PROJECTION_POLICY
    from smolvlm_attention.sample_store import atomic_json,digest,fingerprint
    from smolvlm_attention.config import CACHE_FINGERPRINT
    from smolvlm_inference.model import CAMERA_ORDER
    geometry={'visual_tokens_per_view':64,'text_dim':576}
    model=SimpleNamespace(architecture_signature={'fixture':True},geometry=geometry,prompt_camera_order=CAMERA_ORDER,
        visual_token_len=384,native=SimpleNamespace(model=SimpleNamespace(text_model=SimpleNamespace(layers=[None]*30))))
    checkpoint=tmp_path/'checkpoint.pt';checkpoint.write_bytes(b'checkpoint')
    prepared=tmp_path/'prepared';prepared.mkdir();(prepared/'test.jsonl').write_text('{}\n')
    cache=tmp_path/'cache';cache.mkdir();np.save(cache/'projected.npy',np.zeros((1,6,64,576),dtype='<f4'))
    atomic_json(cache/'index.json',{'key':0})
    identity=dict(projection_policy=GROUPED_PROJECTION_POLICY,checkpoint_sha256=digest(checkpoint),model=model.architecture_signature,
        raw_cache_manifest=dict(fingerprint=CACHE_FINGERPRINT,geometry=geometry,camera_order=CAMERA_ORDER),
        raw_cache_descriptor=dict(prepared_hashes={'test.jsonl':digest(prepared/'test.jsonl')}))
    manifest=dict(format=FORMAT,complete=True,identity=identity,dtype='<f4',shape=[1,6,64,576],
        file_bytes=(cache/'projected.npy').stat().st_size,payload_sha256=digest(cache/'projected.npy'),index_sha256=digest(cache/'index.json'))
    manifest['fingerprint']=fingerprint(manifest);atomic_json(cache/'manifest.json',manifest)
    receipt=validate_projected_cache(cache,model,checkpoint,prepared,tmp_path/'receipt.json')
    assert open_projected_receipt(receipt).array.flags.writeable is False
    with (cache/'projected.npy').open('ab') as stream:stream.write(b'bad')
    with pytest.raises(ValueError,match='changed'):open_projected_receipt(receipt)
    checkpoint.write_bytes(b'wrong')
    with pytest.raises(ValueError):validate_projected_cache(cache,model,checkpoint,prepared,tmp_path/'another.json')
