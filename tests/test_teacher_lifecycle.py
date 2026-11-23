"""Real native CPU extraction through compact prompts and durable resumed shards."""
import json
import shutil

import numpy as np
import pytest
import torch

from smolvlm_attention.data import make_batch, open_validated_cache, validate_cache
from smolvlm_attention.sample_store import canonical, digest, fingerprint
from smolvlm_attention.teacher_targets.attention import collect_teacher_log_scores
from smolvlm_attention.teacher_targets.prompts import prepare_prompt_store
from smolvlm_attention.teacher_targets.storage import TargetShard, finalize_split, validate_export
from smolvlm_inference.cache import record_key
from test_smolvlm_cache import fixture_cache
from test_smolvlm_pruned_decoder import fixture


def test_real_native_train_val_interrupt_resume_finalize_relocate(tmp_path):
    model, _ = fixture(30)
    cache, prepared, _, _ = fixture_cache(tmp_path)
    first = json.loads((prepared/'train.jsonl').read_text())
    source_rows = [dict(first, sample_id=f'sample-{i}', question=question)
                   for i, question in enumerate(('What is visible?', 'Where?', 'What changed?\nA person.'))]
    # Cross-split IDs/eval IDs intentionally repeat; identity is (split, sample_id).
    for split in ('train','val','test'):
        (prepared/f'{split}.jsonl').write_bytes(b''.join(canonical(row)+b'\n' for row in source_rows))
    (prepared/'manifest.json').write_bytes(canonical({'counts':dict(train=3,val=3,test=3)}))
    values = np.arange(6*16*8, dtype=np.float32).reshape(1,6,16,8)/1000
    values.astype(np.float16).tofile(cache/'features.float16.memmap')
    manifest = json.loads((cache/'manifest.json').read_text())
    manifest['geometry'] = dict(model.geometry)
    manifest['shape'] = [1,6,16,8]
    manifest['record_counts'] = dict(train=3,val=3,test=3)
    manifest['prepared_hashes'] = {path.name:digest(path) for path in prepared.iterdir()}
    manifest.pop('fingerprint')
    manifest['fingerprint'] = fingerprint(manifest)
    (cache/'manifest.json').write_bytes(canonical(manifest))
    reader = open_validated_cache(validate_cache(cache,prepared,tmp_path/'receipt'))
    weights = {key:value.clone() for key,value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    for split in ('train','val'):
        prompts = prepare_prompt_store(reader,model.processor,split,tmp_path/f'prompts-{split}',max_length=2048)
        batch = make_batch(reader,[prompts.row(i) for i in range(3)],model.processor,manifest['camera_order'])
        assert batch['attention_mask'].sum(1).unique().numel() > 1
        expected = collect_teacher_log_scores(model,batch).numpy()
        identity = dict(split=split,score_layer_index=22,score_layer_number=23,prompt_identity=prompts.identity)
        shard0 = TargetShard(tmp_path/f'{split}-shard0',identity,[2,0],24)
        shard1 = TargetShard(tmp_path/f'{split}-shard1',identity,[1],24)
        with shard0.lock():
            shard0.commit([2],expected[[2]])
        # Crash after payload flush but before receipt: bytes do not imply completion.
        shard0.array[1] = expected[0]
        shard0.array.flush()
        reopened = TargetShard(shard0.directory,identity,[2,0],24)
        assert reopened.pending_rows.tolist() == [0]
        assert TargetShard(shard1.directory,identity,[1],24).pending_rows.tolist() == [1]
        with pytest.raises(ValueError,match='Incomplete'):
            finalize_split(tmp_path/f'export-{split}',identity,[reopened,shard1],
                           (prompts.portable_row(i) for i in range(3)),3,24)
        with reopened.lock():
            reopened.commit([0],expected[[0]])
        with shard1.lock():
            shard1.commit([1],expected[[1]])
        result = finalize_split(tmp_path/f'export-{split}',identity,[shard1,reopened],
                                (prompts.portable_row(i) for i in range(3)),3,24)
        assert result['complete'] and result['shape'] == [3,24]
        np.testing.assert_array_equal(np.load(tmp_path/f'export-{split}'/'log_probs.npy'),expected)
        portable_rows = [json.loads(line) for line in (tmp_path/f'export-{split}'/'samples.jsonl').read_text().splitlines()]
        assert [row['sample_id'] for row in portable_rows] == [row['sample_id'] for row in source_rows]
        assert all(row['cache_key'] == record_key(first) for row in portable_rows)
        assert all(not ({'answer','input_ids','ground_truth'} & row.keys()) for row in portable_rows)
        moved = tmp_path/f'relocated-{split}'
        shutil.copytree(tmp_path/f'export-{split}',moved)
        assert validate_export(moved,identity) == result
        assert finalize_split(tmp_path/f'export-{split}',identity,[shard1,reopened],[],3,24) == result
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(value,weights[key]) for key,value in model.state_dict().items())
