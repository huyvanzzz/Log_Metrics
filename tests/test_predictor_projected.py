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
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    (prepared / 'train.jsonl').write_text('sample', encoding='utf-8')
    checkpoint = tmp_path / 'best.pt'
    checkpoint.write_bytes(b'fixture')
    model = SimpleNamespace(architecture_signature={'fixture': True})
    identity = dict(checkpoint_sha256=digest(checkpoint), model=model.architecture_signature,
                    raw_cache_descriptor={'prepared_hashes': {'train.jsonl': digest(prepared / 'train.jsonl')}})
    projected = tmp_path / 'projected'
    build_cache(np.ones((1, 6, 2, 2), np.float16), {'key': 0}, projected, lambda x: x,
                identity, projected_shape=(6, 2, 2))
    finalize_cache(projected)
    artifacts = dict(model={'checkpoint': str(checkpoint)}, prepared_dir=str(prepared),
                     cache_dir=str(tmp_path / 'RAW_DOES_NOT_EXIST'))
    assert resolve_projected({}, artifacts, model, tmp_path / 'output', projected_input=str(projected)) == str(projected)
