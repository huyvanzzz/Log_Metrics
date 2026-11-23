import json
import shutil
import numpy as np
import pytest
from test_smolvlm_cache import fixture_cache
from test_smolvlm_model import tiny_components
from smolvlm_attention.data import validate_cache, open_validated_cache
from smolvlm_attention.teacher_targets.prompts import prepare_prompt_store, PromptStore


def inputs(tmp_path):
    cache, prepared, _, _ = fixture_cache(tmp_path)
    reader = open_validated_cache(validate_cache(cache, prepared, tmp_path / 'receipt'))
    _, processor = tiny_components()
    return reader, processor


def test_compact_answer_free_native_prompt_and_relocation(tmp_path):
    from smolvlm_inference.text import CachedBatchCollator, chat_messages
    reader, processor = inputs(tmp_path)
    store = prepare_prompt_store(reader, processor, 'train', tmp_path / 'prompts', max_length=512)
    row = store.row(0)
    collator = CachedBatchCollator(processor, image_seq_len=1, max_length=512, max_new_tokens=0)
    prompt = processor.apply_chat_template(chat_messages('Where?'), tokenize=False, add_generation_prompt=True)
    expected = processor.tokenizer(collator._expand(prompt), add_special_tokens=True, truncation=False)['input_ids']
    assert row['input_ids'] == expected
    assert row['query_token_id'] == expected[-1] and row['query_position'] == len(expected)-1
    assert row['split'] == 'train' and row['row'] == 0 and row['feature_index'] == 0
    assert isinstance(store.lengths, np.memmap)
    assert 'input_ids' not in store.portable_row(0)
    assert 'answer' not in (tmp_path / 'prompts' / 'rows.jsonl').read_text()
    moved = tmp_path / 'moved'
    shutil.copytree(tmp_path / 'prompts', moved)
    assert PromptStore(moved).row(0) == row
    assert prepare_prompt_store(reader, processor, 'train', tmp_path / 'prompts', max_length=512).identity == store.identity
    source = reader.prepared / 'train.jsonl'
    record = json.loads(source.read_text()); record['answer'] = 'Different answer'
    source.write_text(json.dumps(record)+'\n')
    other = prepare_prompt_store(reader, processor, 'train', tmp_path / 'other', max_length=512)
    assert other.row(0) == row


def test_split_identity_and_reject_truncation_duplicates_corruption(tmp_path):
    reader, processor = inputs(tmp_path)
    train = prepare_prompt_store(reader, processor, 'train', tmp_path / 'train', max_length=512)
    val = prepare_prompt_store(reader, processor, 'val', tmp_path / 'val', max_length=512)
    assert train.row(0)['sample_id'] == val.row(0)['sample_id']
    assert train.identity != val.identity
    with pytest.raises(ValueError, match='context'):
        prepare_prompt_store(reader, processor, 'train', tmp_path / 'short', max_length=2)
    with pytest.raises(ValueError, match='split'):
        prepare_prompt_store(reader, processor, 'test', tmp_path / 'test')
    with (tmp_path / 'train' / 'input_ids.bin').open('ab') as out:
        out.write(b'xxxx')
    with pytest.raises(ValueError, match='checksum'):
        PromptStore(tmp_path / 'train')
    source = reader.prepared / 'train.jsonl'
    source.write_text(source.read_text()*2)
    reader.counts['train'] = 2
    with pytest.raises(ValueError, match='duplicate'):
        prepare_prompt_store(reader, processor, 'train', tmp_path / 'duplicates', max_length=512)

def test_multiple_rows_stream_mapping_and_changed_identity(tmp_path):
    reader, processor = inputs(tmp_path)
    source = reader.prepared / 'train.jsonl'
    first = json.loads(source.read_text())
    second = dict(first, sample_id='second', question='What changed?\n<c1,CAM_FRONT,1>')
    source.write_text(json.dumps(second)+'\n'+json.dumps(first)+'\n')
    reader.counts['train'] = 2
    store = prepare_prompt_store(reader, processor, 'train', tmp_path / 'many', max_length=512)
    assert [store.row(i)['sample_id'] for i in (1,0)] == ['test-0','second']
    assert store.row(0)['prompt_ids_sha256'] != store.row(1)['prompt_ids_sha256']
    assert all(store.row(i)['row'] == i for i in range(2))
    assert store.ids.flags.writeable is False
    with pytest.raises(IndexError):
        store.row(-1)
    with pytest.raises(ValueError, match='identity'):
        prepare_prompt_store(reader, processor, 'val', tmp_path / 'many', max_length=512)

def test_prompt_progress_is_bounded_and_reports_completion(tmp_path):
    reader, processor = inputs(tmp_path)
    events = []
    prepare_prompt_store(reader, processor, 'train', tmp_path/'progress', max_length=512, progress=events.append)
    assert len(events) == 1
    assert events[-1]['processed'] == events[-1]['total'] == 1
    assert events[-1]['eta_seconds'] == 0
    assert events[-1]['split'] == 'train' and events[-1]['phase'] == 'prepare_prompts'
    assert events[-1]['elapsed_seconds'] >= 0 and events[-1]['samples_per_second'] >= 0
