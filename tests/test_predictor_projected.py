import json
import numpy as np
import pytest

from smolvlm_predictor_io.projected import build_cache, finalize_cache, ProjectedCache


def test_two_rank_resume_roundtrip_without_raw(tmp_path):
    raw = np.arange(5 * 6 * 4 * 2, dtype=np.float16).reshape(5, 6, 4, 2)
    index = {f'camera-key-{i}': i for i in range(5)}
    identity = {'checkpoint_sha256': 'abc', 'raw_cache_manifest': {'fingerprint': 'raw'}}
    calls = []
    def project(batch):
        calls.append(len(batch))
        assert batch.dtype == np.float32
        return batch[:, :, ::2, :]
    options = dict(batch_size=2, world_size=2, projected_shape=(6, 2, 2))
    build_cache(raw, index, tmp_path, project, identity, rank=0, **options)
    with pytest.raises(ValueError, match='coverage incomplete'):
        finalize_cache(tmp_path)
    build_cache(raw, index, tmp_path, project, identity, rank=0, **options)
    assert calls == [2, 1]
    build_cache(raw, index, tmp_path, project, identity, rank=1, **options)
    result = finalize_cache(tmp_path)
    assert isinstance(result.array, np.memmap) and not result.array.flags.writeable
    assert result.index == index
    np.testing.assert_array_equal(result.array, raw[:, :, ::2, :].astype(np.float32))
    assert ProjectedCache(tmp_path, identity).manifest['complete']
    with pytest.raises(ValueError, match='identity mismatch'):
        ProjectedCache(tmp_path, {'checkpoint_sha256': 'other'})
    # Completed caches are reused, callback never invoked.
    build_cache(raw, index, tmp_path, lambda _: pytest.fail(), identity, rank=1, **options)


def test_corrupt_commit_and_changed_schedule_refused(tmp_path):
    raw = np.ones((3, 6, 2, 2), np.float16)
    index = {str(i): i for i in range(3)}
    opts = dict(projected_shape=(6, 2, 2), batch_size=2)
    build_cache(raw, index, tmp_path, lambda x: x, {}, **opts)
    with pytest.raises(ValueError, match='schedule mismatch'):
        build_cache(raw, index, tmp_path, lambda x: x, {}, **{**opts, 'batch_size': 1})
    array = np.load(tmp_path / 'projected.npy', mmap_mode='r+')
    array[0, 0, 0, 0] = 99
    array.flush()
    with pytest.raises(ValueError, match='checksum'):
        build_cache(raw, index, tmp_path, lambda x: x, {}, **opts)


def test_crash_before_receipt_recomputed_and_nonfinite_refused(tmp_path):
    raw = np.ones((3, 6, 2, 2), np.float16)
    index = {str(i): i for i in range(3)}
    calls = 0
    def fail_second(x):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('interrupted')
        return x
    opts = dict(projected_shape=(6, 2, 2), batch_size=2)
    with pytest.raises(RuntimeError):
        build_cache(raw, index, tmp_path, fail_second, {}, **opts)
    with pytest.raises(ValueError, match='connector output'):
        build_cache(raw, index, tmp_path, lambda x: x * np.nan, {}, **opts)
    build_cache(raw, index, tmp_path, lambda x: x, {}, **opts)
    finalize_cache(tmp_path)
    with (tmp_path / 'projected.npy').open('ab') as stream:
        stream.write(b'extra')
    with pytest.raises(ValueError, match='file size'):
        ProjectedCache(tmp_path)


def test_native_connector_only_exact_parity(tmp_path):
    import torch
    from test_smolvlm_pruned_decoder import fixture
    from smolvlm_predictor_io.projected import project_native
    backbone, batch = fixture()
    backbone.requires_grad_(False)
    raw = batch['visual_features'].numpy() if 'visual_features' in batch else batch['features'].numpy()
    with torch.inference_mode():
        expected = backbone._project(torch.from_numpy(raw), len(raw)).reshape(len(raw), 6, -1, backbone.geometry['text_dim'])
    def forbid(*args, **kwargs):
        pytest.fail('decoder/vision/head must not run')
    hooks = [m.register_forward_pre_hook(forbid) for m in
             [backbone.native.model.vision_model, backbone.native.lm_head, *backbone.native.model.text_model.layers]]
    try:
        np.testing.assert_array_equal(project_native(backbone, raw), expected.numpy())
    finally:
        for hook in hooks:
            hook.remove()


def test_projected_input_reuse_does_not_access_raw(tmp_path):
    from types import SimpleNamespace
    from smolvlm_attention.sample_store import digest
    from smolvlm_predictor_io.orchestration import resolve_projected
    from smolvlm_predictor_io.projected import GROUPED_PROJECTION_POLICY
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    (prepared / 'train.jsonl').write_text('sample', encoding='utf-8')
    checkpoint = tmp_path / 'best.pt'
    checkpoint.write_bytes(b'fixture')
    model = SimpleNamespace(architecture_signature={'fixture': True})
    identity = dict(checkpoint_sha256=digest(checkpoint), model=model.architecture_signature,
                    projection_policy=GROUPED_PROJECTION_POLICY.copy(),
                    raw_cache_descriptor={'prepared_hashes': {'train.jsonl': digest(prepared / 'train.jsonl')}})
    projected = tmp_path / 'projected'
    build_cache(np.ones((1, 6, 2, 2), np.float16), {'key': 0}, projected, lambda x: x,
                identity, projected_shape=(6, 2, 2))
    finalize_cache(projected)
    artifacts = dict(model={'checkpoint': str(checkpoint)}, prepared_dir=str(prepared),
                     cache_dir=str(tmp_path / 'RAW_DOES_NOT_EXIST'))
    assert resolve_projected({}, artifacts, model, tmp_path / 'output', projected_input=str(projected)) == str(projected)

def test_parity_failure_saves_scale_and_fp64_diagnostics_without_changing_guard(tmp_path,monkeypatch):
    import torch
    import smolvlm_predictor_io.projected as projected
    from test_smolvlm_pruned_decoder import fixture
    backbone,batch=fixture()
    raw=batch['visual_features'].numpy()
    original=projected.project_native
    def shifted(model,values):
        result=original(model,values).copy()
        if len(values)>1: result[0,0,0,0]+=0.01
        return result
    monkeypatch.setattr(projected,'project_native',shifted)
    weights={k:v.clone() for k,v in backbone.state_dict().items()};rng=torch.get_rng_state().clone()
    path=tmp_path/'parity.json'
    with pytest.raises(ValueError,match='batching parity failed'):
        projected.verify_native_parity(backbone,raw,report_path=path)
    report=json.loads(path.read_text())
    assert report['ok'] is False and report['rtol']==1e-5 and report['atol']==1e-6
    assert report['mismatched_elements']>=1 and report['reference_abs_max']>0
    assert report['fp64_reference']['scope']=='selected_projection_elements_cpu_float64'
    first=next(x for x in report['fp64_reference']['elements'] if x['index']==[0,0,0,0])
    connector=backbone.native.model.connector
    source=connector.pixel_shuffle(torch.from_numpy(raw[0,0])[None],connector.scale_factor).double()
    expected=torch.nn.functional.linear(source,connector.modality_projection.proj.weight.double())[0,0,0].item()
    assert first['fp64_value']==pytest.approx(expected,rel=1e-12,abs=1e-12)
    assert first['batch_abs_error']>first['singleton_abs_error']
    assert all(torch.equal(v,weights[k]) for k,v in backbone.state_dict().items())
    assert torch.equal(rng,torch.get_rng_state())


def test_parity_pass_remains_strict_and_writes_small_report(tmp_path):
    from test_smolvlm_pruned_decoder import fixture
    from smolvlm_predictor_io.projected import verify_native_parity
    model,batch=fixture();path=tmp_path/'parity.json'
    report=verify_native_parity(model,batch['visual_features'].numpy(),report_path=path)
    assert report['ok'] and report['mismatched_elements']==0
    assert 'fp64_reference' not in report
    assert json.loads(path.read_text())==report

def test_nonfinite_parity_stops_and_persists_finite_false(tmp_path,monkeypatch):
    import smolvlm_predictor_io.projected as projected
    from test_smolvlm_pruned_decoder import fixture
    model,batch=fixture();original=projected.project_native
    def corrupt(model,values):
        result=original(model,values).copy();result[0,0,0,0]=np.nan;return result
    monkeypatch.setattr(projected,'project_native',corrupt)
    path=tmp_path/'parity.json'
    with pytest.raises(ValueError,match='batching parity failed'):
        projected.verify_native_parity(model,batch['visual_features'].numpy(),report_path=path)
    report=json.loads(path.read_text());assert not report['ok'] and not report['finite']
    assert 'fp64_reference' not in report


def test_zero_reference_cancellation_diagnostics_are_json_safe():
    from smolvlm_predictor_io.parity_diagnostics import comparison_report,fp64_selected_reference
    from test_smolvlm_pruned_decoder import fixture
    model,batch=fixture();raw=batch['visual_features'].numpy()
    import torch
    with torch.no_grad():model.native.model.connector.modality_projection.proj.weight.zero_()
    reference=np.zeros((len(raw),6,model.geometry['visual_tokens_per_view'],model.geometry['text_dim']),np.float32)
    changed=reference.copy();changed[0,0,0,0]=.01
    report,details=comparison_report(changed,reference,1e-5,1e-6)
    report['fp64_reference']=fp64_selected_reference(model,raw,changed,reference,details)
    json.dumps(report,allow_nan=False)
    assert all(x['cancellation_ratio'] is None for x in report['fp64_reference']['elements'])

def test_grouped_projection_calls_connector_once_per_six_camera_group():
    import torch
    from test_smolvlm_pruned_decoder import fixture
    from smolvlm_predictor_io.projected import project_native,project_native_grouped
    model,batch=fixture();raw=batch['visual_features'].numpy()
    expected=np.concatenate([project_native(model,raw[i:i+1]) for i in range(len(raw))])
    calls=[]
    hook=model.native.model.connector.register_forward_pre_hook(lambda module,args:calls.append(args[0].shape[0]))
    try:actual=project_native_grouped(model,raw)
    finally:hook.remove()
    assert calls==[6]*len(raw)
    np.testing.assert_array_equal(actual,expected)


def test_grouped_parity_uses_fp64_and_accepts_only_actual_grouped_path(tmp_path,monkeypatch):
    import smolvlm_predictor_io.projected as projected
    from test_smolvlm_pruned_decoder import fixture
    model,batch=fixture();raw=batch['visual_features'].numpy();original=projected.project_native
    def bulk_only_error(model,values):
        result=original(model,values).copy()
        if len(values)>1:result[0,0,0,0]+=.01
        return result
    monkeypatch.setattr(projected,'project_native',bulk_only_error)
    report=projected.verify_grouped_parity(model,raw,report_path=tmp_path/'grouped.json')
    assert report['ok'] and report['rtol']==1e-5 and report['atol']==1e-5
    assert not report['bulk_diagnostic']['ok']
    assert report['acceptance_scope']=='grouped_path_finite_and_selected_cpu_fp64'
    def corrupt_all(model,values):
        result=original(model,values).copy();result[0,0,0,0]+=.01;return result
    monkeypatch.setattr(projected,'project_native',corrupt_all)
    with pytest.raises(ValueError,match='grouped parity failed'):
        projected.verify_grouped_parity(model,raw,report_path=tmp_path/'grouped.json')
    assert not json.loads((tmp_path/'grouped.json').read_text())['ok']

def test_grouped_parity_applies_new_absolute_tolerance_and_rejects_nonfinite(tmp_path,monkeypatch):
    import smolvlm_predictor_io.projected as projected
    from test_smolvlm_pruned_decoder import fixture
    model,batch=fixture();raw=batch['visual_features'].numpy();original=projected.project_native
    def small_error(model,values):
        result=original(model,values).copy();result[0,0,0,0]+=5e-6;return result
    monkeypatch.setattr(projected,'project_native',small_error)
    result=projected.verify_grouped_parity(model,raw,report_path=tmp_path/'grouped.json')
    assert result['ok']
    def nan(model,values):
        result=original(model,values).copy();result[0,0,0,0]=np.nan;return result
    monkeypatch.setattr(projected,'project_native',nan)
    with pytest.raises(ValueError,match='grouped parity failed'):
        projected.verify_grouped_parity(model,raw,report_path=tmp_path/'grouped.json')
    assert not json.loads((tmp_path/'grouped.json').read_text())['finite']


def test_input_rejects_cache_without_new_grouped_policy(tmp_path):
    from types import SimpleNamespace
    from smolvlm_predictor_io.orchestration import _check_origin
    cache=SimpleNamespace(manifest={'identity':{}})
    with pytest.raises(ValueError,match='projection policy'):
        _check_origin(cache,None,None,None)

def test_worker_refuses_old_projection_policy_before_cuda_startup(tmp_path):
    from smolvlm_predictor_io.orchestration import build_worker
    path=tmp_path/'spec.json';path.write_text(json.dumps({'identity':{}}))
    with pytest.raises(ValueError,match='projection policy'):
        build_worker(path,0)
