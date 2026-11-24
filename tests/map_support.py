"""Small real native teacher/cache fixture for MAP integration tests."""
def make_inputs(tmp_path, *, reorder=False):
    import copy, json, hashlib
    import numpy as np
    import torch
    from transformers import Idefics3ForConditionalGeneration
    from test_smolvlm_model import tiny_components
    from test_teacher_loader import loader_inputs
    from smolvlm_inference.model import CachedSmolVLM
    from smolvlm_attention.sample_store import digest, fingerprint
    from smolvlm_attention.teacher_targets.runtime import validate_teacher_cache
    from smolvlm_attention.teacher_targets.prompts import prepare_prompt_store
    from smolvlm_attention.teacher_targets.storage import TargetShard, finalize_split
    from smolvlm_attention.teacher_targets.attention import collect_teacher_log_scores
    from smolvlm_attention.data import open_validated_cache, make_batch
    from smolvlm_inference.cache import CAMERA_ORDER, canonical
    from smolvlm_map_predictor.data import prepare_inputs
    native, processor = tiny_components()
    cfg = copy.deepcopy(native.config)
    cfg.vision_config.image_size = 64
    cfg.text_config.num_hidden_layers = 24
    cfg.text_config.max_position_embeddings = 1024
    native = Idefics3ForConditionalGeneration._from_config(cfg, attn_implementation={'vision_config':'eager','text_config':'sdpa'})
    model = CachedSmolVLM(native, processor, model_id='fixture', model_revision='fixture').eval()
    reader, _, _, _ = loader_inputs(tmp_path, model)
    cache, prepared = reader.directory, reader.prepared
    for split in ('train', 'val', 'test'):
        path = prepared/f'{split}.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        for i, row in enumerate(rows): row['eval_id'] = i
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
    manifest_path = cache/'manifest.json'; manifest = json.loads(manifest_path.read_text())
    manifest['prepared_hashes'] = {name: digest(prepared/name) for name in ('manifest.json','train.jsonl','val.jsonl','test.jsonl')}
    manifest.update(model_id=model.model_id, revision=model.model_revision, processor_policy=model.architecture_signature['processor'])
    manifest.pop('fingerprint'); manifest['fingerprint'] = hashlib.sha256(canonical(manifest)).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    descriptor_path = validate_teacher_cache(cache, prepared, tmp_path/'fixture_validation')
    descriptor = json.loads(descriptor_path.read_text(encoding='utf-8')); reader = open_validated_cache(descriptor_path)
    counts = {'train':3,'val':3,'test':3}
    torch.save(model, tmp_path/'base_fixture.pt')
    checkpoint_sha256 = digest(tmp_path/'base_fixture.pt')
    protocol = dict(score_layer_index=22, score_layer_number=23, visual_tokens=384,
                    splits={'train':3,'val':3}, query_policy='last_valid_native_generation_prompt_token',
                    normalization='mean_heads_all_valid_keys_then_visual_normalization_log_v1')
    expected_model = dict(model_id=model.model_id, revision=model.model_revision, geometry=model.geometry,
                          processor_policy=model.architecture_signature['processor'], camera_order=CAMERA_ORDER)
    for split in ('train','val'):
        store = prepare_prompt_store(reader, processor, split, tmp_path/'fixture_prompts'/split, max_length=1024)
        batch = make_batch(reader, [store.row(i) for i in range(3)], processor, CAMERA_ORDER)
        scores = collect_teacher_log_scores(model, batch)
        identity = dict(split=split, score_layer_index=22, score_layer_number=23, protocol=protocol,
            axes=['sample','visual_token'], dtype='<f4', normalization=protocol['normalization'],
            checkpoint_sha256=checkpoint_sha256, native_model=model.model_id, native_revision=model.model_revision,
            model=expected_model, cache_fingerprint=manifest['fingerprint'], cache_manifest_sha256=descriptor['manifest_sha256'],
            cache_index_sha256=descriptor['index_sha256'], cache_payload_sha256=descriptor['payload_sha256'],
            prepared_hashes=descriptor['prepared_hashes'])
        shard = TargetShard(tmp_path/'fixture_shards'/split, identity, np.arange(3), visual_tokens=384)
        shard.commit([0,1,2], scores.numpy())
        finalize_split(tmp_path/'targets'/split, identity, [shard], (store.portable_row(i) for i in range(3)), 3, 384)
    if reorder:
        for split in ('train', 'val'):
            directory = tmp_path/'targets'/split
            rows = [json.loads(line) for line in (directory/'samples.jsonl').read_text(encoding='utf-8').splitlines()]
            order = [2, 0, 1]; rows = [rows[i] for i in order]
            for i, row in enumerate(rows): row['row'] = i
            (directory/'samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
            array = np.load(directory/'log_probs.npy'); np.save(directory/'log_probs.npy', array[order])
            path = directory/'manifest.json'; export = json.loads(path.read_text(encoding='utf-8'))
            export['checksums'] = {name: digest(directory/name) for name in ('samples.jsonl','log_probs.npy')}
            path.write_text(json.dumps(export), encoding='utf-8')
    receipt = prepare_inputs(tmp_path/'targets', cache, prepared, model, tmp_path/'map_inputs',
                             expected_counts=counts, checkpoint_sha256=checkpoint_sha256)
    return model, receipt
