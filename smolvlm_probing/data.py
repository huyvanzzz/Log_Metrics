"""Read the immutable metadata freeze and join only its selected test prompts."""
import copy
import json
from pathlib import Path

from .config import CAMERA_ORDER, DATASET_FINGERPRINT
from .state import digest, fingerprint


def load_dataset(directory=None):
    directory = Path(directory) if directory else Path(__file__).resolve().parents[1] / 'data' / 'drivelm_attention_v1'
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['fingerprint'] != DATASET_FINGERPRINT or fingerprint(manifest['identity']) != DATASET_FINGERPRINT:
        raise ValueError('Unexpected frozen dataset identity')
    for name, expected in manifest['identity']['file_sha256'].items():
        if digest(directory / name) != expected:
            raise ValueError(f'Frozen dataset hash mismatch: {name}')
    def read(name):
        return [json.loads(line) for line in (directory / (name + '.jsonl')).read_text(encoding='utf-8').splitlines()]
    rows = read('causal')
    by_id = {row['sample_id']: row for row in rows}
    if len(rows) != 2581 or len(by_id) != len(rows):
        raise ValueError('Causal must contain exactly 2581 unique questions')
    for row in rows:
        if list(row['camera_paths']) != CAMERA_ORDER or row.get('split') != 'test':
            raise ValueError('Dataset camera order/split changed')
        row['localization_targets'] = {'with_coordinates': [], 'without_coordinates': []}
    for partition, count in (('with_coordinates', 749), ('without_coordinates', 652)):
        selected = read('localization_' + partition)
        target_keys = set()
        for row in selected:
            source = by_id.get(row['sample_id'])
            if source is None or any(source[k] != row[k] for k in (
                    'question', 'answer', 'eval_id', 'camera_paths', 'frame_id', 'scene_id')):
                raise ValueError('Localization sample differs from causal source')
            for target in row['targets']:
                span = target['reference_span']
                if not (0 <= span['start'] < span['end'] <= len(row['answer'])) or row['answer'][span['start']:span['end']] != span['text']:
                    raise ValueError('Invalid frozen reference span')
                key = (row['frame_id'], target['object_tag'])
                if key in target_keys:
                    raise ValueError('Repeated frozen localization instance')
                target_keys.add(key)
            source['localization_targets'][partition] = copy.deepcopy(row['targets'])
        if len(target_keys) != count or len(selected) != (749 if partition == 'with_coordinates' else 210):
            raise ValueError('Unexpected localization coverage')
    return {'rows': sorted(rows, key=lambda row: row['eval_id']), 'manifest': manifest,
            'directory': str(directory.resolve())}


def prepare_inputs(dataset, reader, processor, config, *, progress=True):
    from tqdm.auto import tqdm
    from smolvlm_inference.cache import record_key, records
    from smolvlm_inference.text import CachedBatchCollator, chat_messages
    prepared = {row['sample_id']: row for row in records(reader.prepared / 'test.jsonl')}
    if len(prepared) != 16817:
        raise ValueError('Prepared test count differs')
    collator = CachedBatchCollator(processor, image_seq_len=64,
        max_length=config.text_config.max_position_embeddings, max_new_tokens=64)
    image_id = processor.tokenizer.convert_tokens_to_ids(str(processor.image_token))
    output = []
    for source in tqdm(dataset['rows'], desc='Prepare probing prompts', unit='question', disable=not progress):
        original = prepared.get(source['sample_id'])
        if original is None or any(original[k] != source[k] for k in ('question', 'answer', 'eval_id', 'camera_paths')):
            raise ValueError('Frozen source differs from pinned prepared data')
        if str(processor.image_token) in source['question']:
            raise ValueError('Question contains a reserved image token')
        prompt = processor.apply_chat_template(chat_messages(source['question']), tokenize=False, add_generation_prompt=True)
        ids = processor.tokenizer(collator._expand(prompt), add_special_tokens=True, truncation=False)['input_ids']
        if ids.count(image_id) != 384 or len(ids) + 64 > config.text_config.max_position_embeddings:
            raise ValueError('Prompt geometry/context budget differs; truncation disabled')
        key = record_key(original)
        if key not in reader.index:
            raise ValueError('Selected test sample has no cache features')
        output.append(dict(source, input_ids=ids, feature_index=reader.index[key]))
    return output
