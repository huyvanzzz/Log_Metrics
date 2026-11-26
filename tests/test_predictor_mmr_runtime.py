import copy
import json
import pytest
import torch


def test_test_only_config_has_no_training_or_targets():
    from smolvlm_predictor_mmr.config import load_config
    for variant in ('before', 'after2'):
        cfg = load_config(variant=variant)
        assert cfg['variant'] == variant
        assert 'training' not in cfg and 'targets' not in cfg
        assert cfg['evaluation'] == dict(batch_size=64, keep_tokens=32, max_new_tokens=512)


def test_inference_checkpoint_five_epoch_recipe_needs_no_rng(tmp_path):
    from smolvlm_predictor_mmr.runtime import read_predictor
    model = torch.nn.Linear(2, 2)
    model.architecture_signature = {'architecture': 'test'}
    identity = {'checkpoint_sha256': 'base', 'model': {'id': 'native'}, 'prepared_hashes': {'test.jsonl': 'data'},
                'predictor_query_policy': 'question_tokens_visual_softmax_then_masked_query_mean_v1'}
    path = tmp_path / 'predictor_best.pt'
    payload = dict(format='map_predictor_checkpoint_v1', architecture_signature=model.architecture_signature,
                   identity=identity, recipe={'training': {'epochs': 5}}, model_state_dict=model.state_dict())
    torch.save(payload, path)
    assert read_predictor(path, model, identity, 'before')['recipe']['training']['epochs'] == 5
    for key in ('checkpoint_sha256', 'model', 'prepared_hashes', 'predictor_query_policy'):
        bad = copy.deepcopy(identity); bad[key] = 'wrong'
        with pytest.raises(ValueError): read_predictor(path, model, bad, 'before')
    with pytest.raises(ValueError): read_predictor(path, model, identity, 'after2')


def test_checkpoint_missing_parameter_rejected(tmp_path):
    from smolvlm_predictor_mmr.runtime import read_predictor
    model = torch.nn.Linear(2, 2); model.architecture_signature = {}
    identity = {'predictor_query_policy': 'question_tokens_visual_softmax_then_masked_query_mean_v1'}
    torch.save(dict(format='map_predictor_checkpoint_v1',architecture_signature={},identity=identity,
                    model_state_dict={}), tmp_path/'bad.pt')
    with pytest.raises((ValueError, RuntimeError)): read_predictor(tmp_path/'bad.pt', model, identity, 'before')


@pytest.mark.parametrize('variant',['before','after2'])
def test_real_fixture_test_only_worker_resume_finalize_bundle(tmp_path,monkeypatch,variant):
    import numpy as np
    import shutil
    import zipfile
    from map_support import make_inputs
    from smolvlm_predictor_mmr import runtime
    from smolvlm_predictor_io.projected import build_cache,finalize_cache,ProjectedCache,project_native
    from smolvlm_attention.sample_store import atomic_json,digest
    from smolvlm_predictor_io.compact import CompactDataset
    from torch.utils.data import DataLoader
    model,receipt=make_inputs(tmp_path)
    if variant=='after2':
        from smolvlm_map_predictor_after2.data import adapt_receipt
        receipt=adapt_receipt(receipt)
    info=json.loads(receipt.read_text(encoding='utf-8')); raw=tmp_path/'cache'
    raw=__import__('pathlib').Path(info['cache_dir'])
    array=np.memmap(raw/'features.float16.memmap',mode='r',dtype='<f2',shape=tuple(info['shape']))
    index=json.loads((raw/'index.json').read_text(encoding='utf-8'))
    destination=tmp_path/'projected'
    build_cache(array,index,destination,lambda x:project_native(model,x),{},batch_size=2,
                projected_shape=(6,64,model.geometry['text_dim']))
    finalize_cache(destination)
    shutil.rmtree(tmp_path/'targets') # no teacher targets available anywhere to test-only code
    cfg=dict(variant=variant,architecture=dict(hidden_dim=4,projection_dim=4,visual_tokens=384),
             evaluation=dict(batch_size=2,keep_tokens=32,max_new_tokens=1))
    predictor=runtime.make_predictor(model,cfg).eval(); checkpoint=tmp_path/'best.pt'
    torch.save(dict(format='map_predictor_checkpoint_v1',architecture_signature=predictor.architecture_signature,
        identity=info['identity'],recipe={'training':{'epochs':5}},model_state_dict=predictor.state_dict()),checkpoint)
    output=tmp_path/'result'; output.mkdir(); (output/'logs').mkdir()
    store=runtime.prepare_test_store(model,ProjectedCache(destination),tmp_path/'prepared',output/'internal',
                                     info['identity'],expected_count=3)
    before_stat=store.stat().st_mtime_ns
    assert runtime.prepare_test_store(model,ProjectedCache(destination),tmp_path/'prepared',output/'internal',
                                     info['identity'],expected_count=3)==store
    assert store.stat().st_mtime_ns==before_stat
    assert len(CompactDataset(store,'test'))==3
    run=dict(version=1,config=cfg,identity=info['identity'],receipt=str(store),receipt_sha256=digest(store),
        output_dir=str(output),backbone_checkpoint=str(tmp_path/'base_fixture.pt'),predictor_checkpoint=str(checkpoint),
        stable=dict(predictor_sha256=digest(checkpoint),projected_fingerprint='fixture',execution=runtime.execution_identity()))
    atomic_json(output/'run.json',run); atomic_json(output/'runtime_config.json',cfg)
    # Synchronous test loader avoids spawning workers from pytest; production remains 2 spawn workers.
    def loader(*a,**kw):
        for key in ('num_workers','persistent_workers','prefetch_factor','multiprocessing_context'): kw.pop(key,None)
        return DataLoader(*a,num_workers=0,**kw)
    monkeypatch.setattr(runtime,'DataLoader',loader)
    gates=[]; monkeypatch.setattr(runtime,'_stress_gate',lambda *a: gates.append(a[2]))
    for rank in (0,1): runtime.worker(run,rank,device='cpu',backbone=model)
    commits={str(p):p.read_bytes() for p in (output/'evaluation').glob('rank_*/batch_*.json')}
    for rank in (0,1): runtime.worker(run,rank,device='cpu',backbone=model)
    assert gates==[0,1,0,1]
    # Simulate interruption before the final rank-0 commit: only missing work is recomputed.
    last=sorted((output/'evaluation/rank_0').glob('batch_*.json'))[-1]
    last.unlink(); (output/'evaluation/rank_0/complete.json').unlink()
    runtime.worker(run,0,device='cpu',backbone=model)
    assert gates==[0,1,0,1,0]
    assert all(__import__('pathlib').Path(p).read_bytes()==data for p,data in commits.items())
    import smolvlm_topk_benchmark.scoring as scoring
    monkeypatch.setattr(scoring,'score_rows',lambda records,expected_count:dict(metrics={},metrics_display={},scoring_identity={},diagnostics={'count':len(records)}))
    runtime.finalize(run)
    assert len((output/'predictions.jsonl').read_text(encoding='utf-8').splitlines())==3
    archive=runtime.bundle(output/'run.json')
    with zipfile.ZipFile(archive) as file:
        assert 'checkpoints/predictor_best.pt' in file.namelist()
        assert not any('projected.npy' in name or 'targets/' in name or '.rng' in name for name in file.namelist())


def test_test_only_notebooks_syntax_sync_and_no_training_targets():
    import ast
    from pathlib import Path
    for variant in ('before','after2'):
        notebook=json.loads((Path('notebooks')/f'kaggle_smolvlm_predictor_{variant}_mmr_test.ipynb').read_text(encoding='utf-8'))
        codes=[''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type']=='code']
        for code in codes: ast.parse(code)
        text='\n'.join(codes)
        assert text.index("os.environ['PYTORCH_CUDA_ALLOC_CONF']") < text.index('from smolvlm_predictor_io.progress')
        assert text.index("'pull','--ff-only'") < text.index('from smolvlm_predictor_io.progress')
        assert "'--untracked-files=no'" in text and 'TARGETS_INPUT' not in text
        assert "run('train'" not in text and "run('evaluate'" in text and "run('bundle'" in text
        assert 'smolvlm_map_predictor/requirements.txt' in text


def test_mmr_config_rejects_protocol_changes(tmp_path):
    from smolvlm_predictor_mmr.config import load_config
    cfg=load_config()
    for key,value in [('batch_size',32),('keep_tokens',31),('max_new_tokens',100)]:
        bad=copy.deepcopy(cfg); bad['evaluation'][key]=value
        path=tmp_path/'config.json'; path.write_text(json.dumps(bad))
        with pytest.raises(ValueError): load_config(path)


def test_mmr_identity_is_separate_from_mass_and_topk():
    from smolvlm_predictor_mmr.runtime import evaluation_identity
    for variant,boundary in [('before','before_block_0'),('after2','after_full_block_2')]:
        run=dict(config=dict(variant=variant,architecture={},evaluation={}),identity={},
                 stable=dict(predictor_sha256='checkpoint',projected_fingerprint='projected'))
        identity=evaluation_identity(run)
        assert identity['version']=='predictor_mmr_hard_prune_v1'
        assert identity['mmr']['lambda_weight']==0.9
        assert identity['prune_boundary']==boundary
        assert not any('mass' in key or 'intervention' in key for key in identity)
