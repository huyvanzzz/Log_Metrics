import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_smolvlm_pruned_decoder import fixture
from smolvlm_inference.decoder import PrunedGreedyDecoder
from smolvlm_inference.pruning import stage2_indices
from smolvlm_inference.saved_attention_run import batch_scores, validate_config, validate_export


def test_saved_scores_mapping_native_blocks_and_cache(monkeypatch):
    model,batch=fixture()
    engine=PrunedGreedyDecoder(model,2,stage1_keep=18,stage2_keep=12)
    scores=torch.arange(24,dtype=torch.float32).repeat(2,1)/10000
    weights={k:v.clone() for k,v in model.state_dict().items()}
    rng=torch.get_rng_state().clone()
    import smolvlm_inference.decoder as module
    def forbidden(*args,**kwargs):
        raise AssertionError('online scorer/hook must not run')
    monkeypatch.setattr(module,'question_attention_scores',forbidden)
    monkeypatch.setattr(module,'_capture_qk',forbidden)
    result=engine.generate(batch,max_new_tokens=4,debug=True,stage2_visual_scores=scores)
    reference=engine.generate_reference(batch,max_new_tokens=4,stage2_visual_scores=scores)
    assert torch.equal(result['generated_ids'],reference['generated_ids'])
    debug=result['debug']
    first=torch.tensor(debug['stage1_source_indices'])
    local,_,_=stage2_indices(scores.gather(1,first),first//4,keep=12)
    assert debug['stage2_source_indices']==first.gather(1,local).tolist()
    assert debug['visual_counts_by_layer']==[18,18,18,12]
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,weights[k]) for k,v in model.state_dict().items())
    assert not any(layer.self_attn.q_proj._forward_hooks for layer in model.native.model.text_model.layers)


@pytest.mark.parametrize('kind',['negative','nan','mass','dtype','shape'])
def test_bad_scores_rejected(kind):
    model,batch=fixture()
    engine=PrunedGreedyDecoder(model,2,18,12)
    s=torch.zeros(2,24)
    if kind=='negative':s[0,0]=-1
    if kind=='nan':s[0,0]=float('nan')
    if kind=='mass':s.fill_(1)
    if kind=='dtype':s=s.double()
    if kind=='shape':s=s[:,:23]
    with pytest.raises(ValueError,match='saved visual'):
        engine.generate(batch,max_new_tokens=1,stage2_visual_scores=s)


def test_layer22_and_shuffled_rows():
    a=np.zeros((3,30,384),np.float32)
    a[:,21,:]=.0001
    a[0,22,8]=.4
    a[2,22,9]=.6
    result=batch_scores(SimpleNamespace(attention=a),[{'attention_row':2},{'attention_row':0}])
    assert result[0,9]==.6 and result[1,8]==.4
    assert result.sum().item()==pytest.approx(1)


def test_axes_rejected_before_identity():
    export=SimpleNamespace(metadata={'axes':['sample','visual_token','layer'],'layer_ids':list(range(30))})
    with pytest.raises(ValueError,match='axes/layer'):
        validate_export(export,[],{},'x')


def test_hidden_states_unchanged_until_prune_and_hard_gather():
    model,batch=fixture()
    engine=PrunedGreedyDecoder(model,2,18,12)
    captures=[]
    def hook(module,args,output):
        captures.append(output[0].detach().clone())
    handle=model.native.model.text_model.layers[2].register_forward_hook(hook)
    try:
        engine.generate(batch,max_new_tokens=1)
        engine.generate(batch,max_new_tokens=1,stage2_visual_scores=torch.zeros(2,24))
    finally:
        handle.remove()
    torch.testing.assert_close(captures[0],captures[1],rtol=0,atol=0)
    from smolvlm_inference.decoder import _retain_images
    source,first,_=engine._inputs(batch)
    state=_retain_images(source,first,24)
    state['hidden']=captures[1]
    # Marker-based gather verifies retained values are current block outputs.
    kept=first[:,:12]
    actual=_retain_images(state,kept,24)
    for b in range(2):
        mask=(state['image'][b,:,None]==kept[b,None,:]).any(1) | (state['image'][b]<0)
        expected=state['hidden'][b,mask & state['valid'][b]]
        torch.testing.assert_close(actual['hidden'][b,actual['valid'][b]],expected)


def test_config_and_layer_order():
    from smolvlm_inference.pruning_run import FIXED
    config=dict(FIXED,stage2_layer_index=2,stage2_score_source='saved_base_attention',score_layer_index=22)
    assert validate_config(config)==config
    with pytest.raises(ValueError):validate_config(dict(config,score_layer_index=23))
    export=SimpleNamespace(metadata={'axes':['sample','layer','visual_token'],'layer_ids':list(reversed(range(30)))})
    with pytest.raises(ValueError,match='axes/layer'):
        validate_export(export,[],{},'x')


def test_real_384_to_196_to_32_lifecycle(tmp_path):
    from smolvlm_inference.pruning_text import PruningBatchCollator
    from smolvlm_inference.saved_attention_run import SavedScoreCollator,SavedScoreDecoder
    from smolvlm_inference.pruning_run import evaluate_batches
    from smolvlm_inference.cache import record_key,CAMERA_ORDER
    import json
    model,_=fixture()
    model.geometry.update(vision_tokens=256,visual_tokens_per_view=64)
    model.visual_token_len=384
    rows=[dict(sample_id=str(i),eval_id=i,question=q,answer='car',attention_row=2-i,
               camera_paths={c:f'{i}/{c}' for c in CAMERA_ORDER})
          for i,q in enumerate(['What is visible?','What changed? A person.','What?'])]
    features=np.random.default_rng(12).normal(size=(3,6,256,8)).astype(np.float16)
    cache=SimpleNamespace(array=features,index={record_key(r):i for i,r in enumerate(rows)})
    scores=np.zeros((3,30,384),np.float32)
    for i in range(3):scores[i,22,i::3]=.001
    collator=SavedScoreCollator(PruningBatchCollator(model.processor,image_seq_len=64,max_length=2048,max_new_tokens=3),SimpleNamespace(attention=scores))
    decoder=SavedScoreDecoder(model,2)
    evaluate_batches(model,cache,rows,collator,decoder,tmp_path/'stress','stress',device='cpu',max_new_tokens=3,batch_size=2)
    evaluate_batches(model,cache,rows,collator,decoder,tmp_path,'test',device='cpu',max_new_tokens=3,batch_size=2)
    predictions=[json.loads(line) for line in (tmp_path/'predictions.jsonl').read_text().splitlines()]
    assert [r['sample_id'] for r in predictions]==['0','1','2']
    debug=json.loads((tmp_path/'debug.json').read_text())
    assert debug['visual_counts_by_layer']==[196,196,196,32]
    assert all(len(set(indices))==32 for indices in debug['stage2_source_indices'])


@pytest.mark.parametrize('damage',[None,'checkpoint','data','layer','axes','missing','duplicate','prompt'])
def test_export_identity_and_sample_alignment(damage):
    from smolvlm_attention.config import MODELS,QUERY_POLICY
    from smolvlm_inference.cache import CAMERA_ORDER
    identity={k:'fixture' for k in ('cache_fingerprint','prepared_hashes','cache_manifest_sha256',
        'cache_index_sha256','model_id','native_revision','processor_policy','geometry','image_token_id',
        'tokenizer_class','tokenizer_model_max_length')}
    rows=[dict(row_index=i,sample_id=f's{i}',eval_id=i,question='Q',answer='A',camera_paths={},
               input_ids=[9,3],image_token_id=9,feature_key='x',feature_index=0,
               prompt_length=2,query_position=1,query_token_id=3) for i in range(2)]
    saved=[dict(rows[1],row_index=0),dict(rows[0],row_index=1)]
    metadata=dict(axes=['sample','layer','visual_token'],layer_ids=list(range(30)),camera_order=CAMERA_ORDER,
        identity=dict(model=MODELS['base']['repo'],revision=MODELS['base']['revision'],checkpoint_sha256='sha',
                      policy=QUERY_POLICY,reduction='mean_heads_only_per_sample_v1',camera_order=CAMERA_ORDER,
                      data=copy.deepcopy(identity)))
    if damage=='checkpoint':metadata['identity']['checkpoint_sha256']='other'
    if damage=='data':metadata['identity']['data']['geometry']='other'
    if damage=='layer':metadata['layer_ids'][22]=21
    if damage=='axes':metadata['axes'].reverse()
    if damage=='missing':saved.pop()
    if damage=='duplicate':saved[1]=saved[0]
    if damage=='prompt':saved[0]['input_ids']=[9,4]
    export=SimpleNamespace(metadata=metadata,samples=saved,attention=np.zeros((2,30,384),np.float32))
    if damage:
        with pytest.raises(ValueError):validate_export(export,rows,identity,'sha',expected_count=2)
    else:
        aligned=validate_export(export,rows,identity,'sha',expected_count=2)
        assert [r['attention_row'] for r in aligned]==[1,0]


@pytest.mark.parametrize('damage',[None,'missing','duplicate','wrong_shard','identity'])
def test_two_gpu_shards_merge_exact_coverage(tmp_path,damage):
    import json
    from smolvlm_inference.saved_attention_run import shard_rows,merge_shards
    expected=[dict(eval_id=i,sample_id=str(i)) for i in range(7)]
    assert [len(shard_rows(expected,r)) for r in range(2)]==[4,3]
    for rank in range(2):
        folder=tmp_path/f'rank_{rank}'
        folder.mkdir()
        rows=[dict(r,prediction=f'p{r["eval_id"]}') for r in reversed(shard_rows(expected,rank))]
        if rank==0 and damage=='missing':rows.pop()
        if rank==0 and damage=='duplicate':rows.append(rows[0])
        if rank==0 and damage=='wrong_shard':rows[0]=dict(expected[1],prediction='bad')
        (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        metadata=dict(config={},checkpoint_sha256='sha',export_fingerprint='fp',camera_order=[],dtype='float32',query_policy='last')
        if rank==1 and damage=='identity':metadata['checkpoint_sha256']='other'
        (folder/'metadata.json').write_text(json.dumps(metadata))
    if damage:
        with pytest.raises(ValueError):merge_shards(tmp_path,expected)
        assert not (tmp_path/'predictions.jsonl').exists()
    else:
        values=merge_shards(tmp_path,expected)
        assert [v['eval_id'] for v in values]==list(range(7))
        assert [v['prediction'] for v in values]==[f'p{i}' for i in range(7)]
        metadata=json.loads((tmp_path/'metadata.json').read_text())
        assert metadata['world_size']==2 and metadata['test_count']==7
        assert metadata['shard_counts']==[4,3]


def test_two_gpu_notebook_launches_one_config_on_both_gpus():
    import json,ast
    from pathlib import Path
    n=json.loads(Path('notebooks/kaggle_smolvlm_mvpruner_saved_attention_test.ipynb').read_text())
    source='\n'.join(''.join(c['source']) for c in n['cells'] if c['cell_type']=='code')
    for c in n['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    assert "'--gpus','2'" in source
    assert "CUDA_VISIBLE_DEVICES='0'" not in source
    assert 'mvpruner_prune_index_2_score_index_22.json' in source


@pytest.mark.parametrize('fail',[False,True])
def test_launcher_barrier_and_failure_cleanup(tmp_path,monkeypatch,fail):
    import json,subprocess,sys
    from pathlib import Path
    from smolvlm_inference import saved_attention_run as module
    from smolvlm_inference.pruning_run import FIXED
    prepared=tmp_path/'prepared';prepared.mkdir()
    expected=[dict(eval_id=i,sample_id=str(i)) for i in range(3)]
    (prepared/'test.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in expected))
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(FIXED,stage2_layer_index=2,stage2_score_source='saved_base_attention',score_layer_index=22)))
    args=SimpleNamespace(config=str(config),output=str(tmp_path/'out'),prepared=str(prepared),
                         cache='cache',receipt='receipt',checkpoint='checkpoint',attention_input='export')
    real_popen=subprocess.Popen
    children=[]
    devices=[]
    code=r"""import json,sys,time
from pathlib import Path
folder=Path(sys.argv[1]);rank=int(sys.argv[2]);fail=sys.argv[3]=='True'
if fail and rank==0:sys.exit(7)
metadata=dict(config={},checkpoint_sha256='sha',export_fingerprint='fp',camera_order=[],dtype='float32',query_policy='last')
(folder/'metadata.json').write_text(json.dumps(metadata))
(folder/'stress.ready').touch()
while not (folder.parent/'test.go').exists():time.sleep(.02)
rows=[dict(eval_id=i,sample_id=str(i),prediction='car') for i in range(3) if i%2==rank]
(folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
"""
    def popen(argv,**kwargs):
        folder=argv[argv.index('--output')+1]
        rank=argv[argv.index('--worker-rank')+1]
        devices.append(kwargs['env']['CUDA_VISIBLE_DEVICES'])
        process=real_popen([sys.executable,'-c',code,folder,rank,str(fail)],**kwargs)
        children.append(process)
        return process
    monkeypatch.setattr(module.subprocess,'Popen',popen)
    monkeypatch.setattr(module.torch.cuda,'device_count',lambda:2)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','3,5')
    monkeypatch.setattr(module.common,'preflight_metrics',lambda:None)
    metrics=[]
    monkeypatch.setattr(module.common,'score_predictions',lambda path:metrics.append(path))
    if fail:
        with pytest.raises(RuntimeError,match='failed'):
            module.launch(args)
        assert not metrics
        assert not (Path(args.output)/'test.go').exists()
    else:
        module.launch(args)
        assert len(metrics)==1
        assert (Path(args.output)/'mvpruner_saved_attention_artifacts.zip').exists()
        assert (Path(args.output)/'test.go').exists()
    assert devices==['3','5']
    assert all(p.poll() is not None for p in children)
