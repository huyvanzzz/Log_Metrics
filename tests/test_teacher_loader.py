import json
import pickle
import numpy as np
import pytest
import torch


def test_pinned_loader_settings_match_reduced_prefetch_protocol():
    from smolvlm_attention.teacher_targets.loader import SETTINGS
    from smolvlm_attention.teacher_targets.runtime import load_config
    assert SETTINGS == dict(num_workers=2, persistent_workers=True, pin_memory=True,
                            prefetch_factor=1, multiprocessing_context='spawn')
    config = load_config()
    assert config['loader'] == SETTINGS
    assert config['batch_size'] == 64
    assert config['precision'] == 'float32_no_autocast_no_tf32'


def loader_inputs(tmp_path, model=None):
    from test_smolvlm_cache import fixture_cache
    from test_smolvlm_model import tiny_components
    from smolvlm_attention.data import open_validated_cache, validate_cache
    from smolvlm_attention.sample_store import digest, fingerprint, canonical
    from smolvlm_inference.cache import record_key
    from smolvlm_attention.teacher_targets.prompts import prepare_prompt_store
    from smolvlm_attention.teacher_targets.loader import prepare_loader_receipt
    cache, prepared, _, _ = fixture_cache(tmp_path)
    processor = tiny_components()[1] if model is None else model.processor
    first = json.loads((prepared / 'train.jsonl').read_text())
    second = dict(first, sample_id='second', question='What is visible?\nA person?',
                  camera_paths={key: 'second/' + value for key, value in first['camera_paths'].items()})
    third = dict(first, sample_id='third', question='Where is that car located now?')
    for split in ('train', 'val', 'test'):
        (prepared / f'{split}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in (first, second, third)))
    (prepared / 'manifest.json').write_text(json.dumps(dict(counts=dict(train=3, val=3, test=3))))
    index = {record_key(first): 0, record_key(second): 1}
    (cache / 'index.json').write_bytes(canonical(index))
    manifest = json.loads((cache / 'manifest.json').read_text())
    if model is not None:
        manifest['geometry'] = model.geometry
    geometry = manifest['geometry']
    shape = (2, 6, geometry['vision_tokens'], geometry['vision_dim'])
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 1000
    values.astype('<f2').tofile(cache / 'features.float16.memmap')
    manifest.update(shape=list(shape), num_samples=2, record_counts=dict(train=3, val=3, test=3),
                    prepared_hashes={p.name: digest(p) for p in prepared.iterdir()},
                    index_sha256=digest(cache / 'index.json'))
    manifest.pop('fingerprint'); manifest['fingerprint'] = fingerprint(manifest)
    (cache / 'manifest.json').write_text(json.dumps(manifest))
    reader = open_validated_cache(validate_cache(cache, prepared, tmp_path / 'validated'))
    stores = {s: prepare_prompt_store(reader, processor, s, tmp_path / s, max_length=512)
              for s in ('train', 'val')}
    descriptor = tmp_path / 'descriptor.json'
    descriptor.write_text(json.dumps(dict(cache_dir=str(reader.directory), shape=list(reader.array.shape),
                                           payload_sha256='validated-parent')))
    receipt = prepare_loader_receipt(tmp_path / 'loader.json', descriptor,
                                      {s: str(store.directory) for s, store in stores.items()})
    return reader, processor, stores, receipt


def test_loader_collation_exact_reference_and_receipt_checks(tmp_path):
    from smolvlm_attention.teacher_targets.loader import TeacherDataset, TeacherCollator
    from smolvlm_attention.data import make_batch
    from smolvlm_inference.cache import CAMERA_ORDER
    reader, processor, stores, receipt = loader_inputs(tmp_path)
    dataset = TeacherDataset(receipt)
    collator = TeacherCollator(receipt, processor.tokenizer.pad_token_id, CAMERA_ORDER)
    batch = collator([dataset[('extract', 'train', 1)], dataset[('extract', 'train', 0)]])
    reference = make_batch(reader, [stores['train'].row(i) for i in (1, 0)], processor, CAMERA_ORDER)
    for key in ('input_ids', 'attention_mask', 'visual_features'):
        assert torch.equal(batch[key], reference[key])
    assert batch['visual_features'].dtype == torch.float32
    assert batch['visual_features'].device.type == 'cpu'
    assert batch['attention_mask'].sum(1).unique().numel() == 2
    assert not torch.equal(batch['visual_features'][0], batch['visual_features'][1])
    assert pickle.loads(pickle.dumps(dataset))._inputs is None
    assert pickle.loads(pickle.dumps(collator))._inputs is None
    path = stores['train'].directory / 'input_ids.bin'
    with path.open('ab') as stream:
        stream.write(b'x')
    with pytest.raises(ValueError, match='changed'):
        TeacherDataset(receipt)[('extract', 'train', 0)]


def test_loader_native_attention_resume_finalize(tmp_path):
    from test_smolvlm_pruned_decoder import fixture
    from smolvlm_attention.teacher_targets.loader import teacher_loader
    from smolvlm_attention.teacher_targets.schedule import prepare_schedule, load_rows, pending_batches
    from smolvlm_attention.teacher_targets.storage import TargetShard, finalize_split, validate_export
    from smolvlm_attention.teacher_targets.attention import collect_teacher_log_scores
    from smolvlm_attention.data import make_batch
    from smolvlm_inference.cache import CAMERA_ORDER
    model, _ = fixture(24)
    reader, processor, prompts, receipt = loader_inputs(tmp_path, model)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    description = prepare_schedule(tmp_path / 'schedule', {'train': 3, 'val': 3}, batch_size=1)
    for split in ('train', 'val'):
        expected = collect_teacher_log_scores(model, make_batch(reader,
            [prompts[split].row(i) for i in range(3)], processor, CAMERA_ORDER))
        shards = []
        for rank in (0, 1):
            rows = load_rows(tmp_path / 'schedule', description, split, rank)
            shard = TargetShard(tmp_path / f'{split}-{rank}', {'split': split}, rows, visual_tokens=24)
            batches = pending_batches(shard, rows, batch_size=1)
            # A flushed-but-uncommitted first row must be recomputed on resume.
            shard.array[0] = expected[int(rows[0])].numpy(); shard.array.flush()
            with teacher_loader(receipt, [[('extract', split, row) for row in batch] for batch in batches],
                                processor.tokenizer.pad_token_id, CAMERA_ORDER, rank) as iterator:
                for batch in iterator:
                    scores = collect_teacher_log_scores(model, batch)
                    torch.testing.assert_close(scores, expected[batch['rows']], atol=2e-6, rtol=1e-5)
                    shard.commit(batch['rows'].tolist(), scores.numpy())
            assert pending_batches(shard, rows, batch_size=1) == []
            shards.append(shard)
        export = tmp_path / 'export' / split
        finalize_split(export, {'split': split}, shards, (prompts[split].portable_row(i) for i in range(3)), 3, 24)
        validate_export(export, {'split': split})
        torch.testing.assert_close(torch.from_numpy(np.load(export / 'log_probs.npy')), expected, atol=2e-6, rtol=1e-5)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())


def test_spawn_combined_pool_tail_rng_and_cleanup(tmp_path):
    from smolvlm_attention.teacher_targets.loader import teacher_loader
    from smolvlm_inference.cache import CAMERA_ORDER
    _, processor, _, receipt = loader_inputs(tmp_path)
    batches = [[('stress', 'train', 0)], [('stress', 'val', 0)],
               [('extract', 'train', 0)] * 2, [('extract', 'val', 0)]]
    before = torch.get_rng_state().clone()
    workers = []
    with pytest.raises(RuntimeError, match='intentional'):
        with teacher_loader(receipt, batches, processor.tokenizer.pad_token_id, CAMERA_ORDER) as iterator:
            workers = iterator._workers
            got = [next(iterator) for _ in batches]
            assert [b['phase'] for b in got] == ['stress', 'stress', 'extract', 'extract']
            assert [len(b['rows']) for b in got] == [1, 1, 2, 1]
            assert len({worker.pid for worker in workers}) == 2
            assert all(worker.is_alive() for worker in workers)
            raise RuntimeError('intentional')
    assert all(not worker.is_alive() for worker in workers)
    assert torch.equal(before, torch.get_rng_state())


def test_schedule_seed_sharding_integrity_and_whole_batch_resume(tmp_path):
    from smolvlm_attention.teacher_targets.schedule import prepare_schedule, load_rows, pending_batches
    from smolvlm_attention.teacher_targets.storage import TargetShard
    before = torch.get_rng_state().clone()
    description = prepare_schedule(tmp_path / 'schedule', {'train': 135, 'val': 7})
    assert torch.equal(before, torch.get_rng_state())
    a, b = [load_rows(tmp_path / 'schedule', description, 'train', r) for r in (0, 1)]
    assert not set(a) & set(b)
    assert set(a) | set(b) == set(range(135))
    assert not np.array_equal(a, np.arange(0, 135, 2))
    np.testing.assert_array_equal(load_rows(tmp_path / 'schedule', description, 'val', 1), [1, 3, 5])
    store = TargetShard(tmp_path / 'shard', {'schedule': description}, a, visual_tokens=2)
    batches = pending_batches(store, a)
    assert list(map(len, batches)) == [64, 4]
    store.commit(batches[0], np.full((64, 2), -np.log(2), dtype='<f4'))
    assert pending_batches(store, a) == [batches[1]]
    store.commit([batches[1][0]], np.full((1, 2), -np.log(2), dtype='<f4'))
    with pytest.raises(ValueError, match='Partial'):
        pending_batches(store, a)
    path = tmp_path / 'schedule' / 'train.npy'
    values = np.load(path); values[:2] = values[:2][::-1]
    a._mmap.close(); b._mmap.close()
    np.save(path, values)
    with pytest.raises(ValueError, match='checksum'):
        load_rows(tmp_path / 'schedule', description, 'train', 0)


def test_spawn_worker_exception_joins_pool(tmp_path):
    from smolvlm_attention.teacher_targets.loader import teacher_loader
    from smolvlm_inference.cache import CAMERA_ORDER
    _, processor, _, receipt = loader_inputs(tmp_path)
    workers = []
    with pytest.raises(IndexError):
        with teacher_loader(receipt, [[('extract', 'train', 999)]],
                            processor.tokenizer.pad_token_id, CAMERA_ORDER) as iterator:
            workers = iterator._workers
            next(iterator)
    assert workers and all(not worker.is_alive() for worker in workers)


def test_old_run_and_mutated_loader_receipt_rejected(tmp_path):
    from smolvlm_attention.teacher_targets.runtime import _read_run, _sources, load_config
    from smolvlm_attention.sample_store import digest
    path = tmp_path / 'run.json'
    path.write_text(json.dumps(dict(version=1, identity={})))
    with pytest.raises(ValueError, match='incompatible resume'):
        _read_run(path)
    receipt = tmp_path / 'receipt.json'; receipt.write_text('{}')
    run = dict(version=2, identity=dict(sources=_sources(), protocol=load_config()),
               loader_receipt=str(receipt), loader_receipt_sha256=digest(receipt))
    path.write_text(json.dumps(run))
    assert _read_run(path) == run
    receipt.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='receipt checksum'):
        _read_run(path)


def test_resume_rejects_receipts_not_matching_complete_schedule_batch(tmp_path):
    from smolvlm_attention.teacher_targets.schedule import pending_batches
    from smolvlm_attention.teacher_targets.storage import TargetShard
    store = TargetShard(tmp_path / 'shard', {}, [0, 1], visual_tokens=2)
    for row in (0, 1):
        store.commit([row], np.full((1, 2), -np.log(2), dtype='<f4'))
    with pytest.raises(ValueError, match='Receipt'):
        pending_batches(store, np.array([0, 1]), batch_size=2)
