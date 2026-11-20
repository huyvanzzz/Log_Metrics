"""Pinned full-test prompt preparation and measurement identity (no aggregation)."""
import importlib.metadata
import hashlib
import inspect
import json
from pathlib import Path
import tempfile

from .sample_store import digest, fingerprint


def library_versions():
    return {name: importlib.metadata.version(name) for name in (
        'torch', 'transformers', 'tokenizers', 'numpy', 'huggingface-hub')}


def measurement_sources():
    from . import sample_store
    root = Path(__file__).resolve().parents[1]
    result = {name: digest(root / name) for name in (
        'smolvlm_attention/sample_inputs.py', 'smolvlm_attention/collector.py',
        'smolvlm_attention/sample_worker.py',
        'smolvlm_attention/config.py', 'smolvlm_attention/data.py',
        'smolvlm_inference/model.py', 'smolvlm_inference/text.py', 'smolvlm_inference/cache.py')}
    # Hash measurement/storage semantics separately from reader/ZIP/export presentation.
    for name, function in (('ShardStore.write', sample_store.ShardStore.write),
                           ('shard_bounds', sample_store.shard_bounds),
                           ('validate_samples', sample_store.validate_samples),
                           ('validate_attention', sample_store.validate_attention)):
        result['smolvlm_attention/sample_store.py:' + name] = hashlib.sha256(
            inspect.getsource(function).encode('utf-8')).hexdigest()
    return result


def prepare_inputs(artifacts, output):
    """Candidate inputs/receipt are isolated; existing run files are never overwritten."""
    from transformers import AutoConfig, AutoProcessor
    from tqdm.auto import tqdm
    from smolvlm_inference.cache import CAMERA_ORDER, FEATURE_FILE, record_key, records
    from smolvlm_inference.model import MODEL_ID, MODEL_REVISION, _configure_processor, _geometry, processor_policy
    from .config import CACHE_FINGERPRINT, TEST_COUNT
    from .data import validate_cache, open_validated_cache, prepare_test
    directory = Path(artifacts['cache_dir'])
    stat = (directory / FEATURE_FILE).stat()
    key = fingerprint(dict(cache=str(directory.resolve()), prepared=str(Path(artifacts['prepared_dir']).resolve()),
        manifest=digest(directory / 'manifest.json'), index=digest(directory / 'index.json'),
        payload_size=stat.st_size, payload_mtime_ns=stat.st_mtime_ns,
        prepared_hashes={n: digest(Path(artifacts['prepared_dir']) / n)
                         for n in ('manifest.json', 'train.jsonl', 'val.jsonl', 'test.jsonl')}))
    receipt = Path(output) / 'validation' / key / 'validated_cache.json'
    if receipt.exists():
        reader = open_validated_cache(receipt)
        print('Reusing validated feature-cache receipt.', flush=True)
    else:
        with tqdm(desc='Validate cache', unit='item') as bar:
            def progress(event):
                if bar.desc != event['stage']:
                    bar.set_description(event['stage'])
                    bar.unit = 'byte' if event['stage'] == 'features' else 'record'
                    bar.reset(total=event['total'])
                bar.update(event['completed'] - bar.n)
            receipt = validate_cache(directory, artifacts['prepared_dir'], receipt.parent,
                                     CACHE_FINGERPRINT, progress=progress)
        reader = open_validated_cache(receipt)
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=False)
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=False)
    _configure_processor(processor, config)
    geometry = _geometry(config)
    expected = dict(model_id=MODEL_ID, revision=MODEL_REVISION, processor_policy=processor_policy(processor),
                    geometry=geometry, camera_order=CAMERA_ORDER, compute_dtype='float32', storage_dtype='float16')
    if (reader.counts['test'] != TEST_COUNT or config.text_config.num_hidden_layers != 30
            or geometry['visual_tokens_per_view'] != 64
            or any(reader.manifest.get(k) != v for k, v in expected.items())):
        raise ValueError('Pinned model/processor/test data differs from cache')
    # prepare_test writes only inside a disposable candidate directory.
    with tempfile.TemporaryDirectory(prefix='prompt-candidate-', dir=Path(output)) as temporary:
        path = prepare_test(reader, processor, CAMERA_ORDER, temporary,
                            config.text_config.max_position_embeddings, progress=True)
        prepared = json.loads(path.read_text(encoding='utf-8'))
    original = list(records(reader.prepared / 'test.jsonl'))
    rows = []
    for index, (source, prompt) in enumerate(zip(original, prepared['records'])):
        if (source['sample_id'] != prompt['sample_id'] or source['eval_id'] != index
                or prompt['eval_id'] != index or source.get('split') != 'test'):
            raise ValueError('Prepared test order/split differs from pinned full test')
        ids = prompt['input_ids']
        row = dict(source, **prompt)
        row.update(row_index=index, feature_key=record_key(source),
            prompt_length=len(ids), query_position=len(ids)-1, query_token_id=ids[-1],
            image_token_id=prepared['image_token_id'])
        rows.append(row)
    if len(rows) != TEST_COUNT or len(original) != TEST_COUNT:
        raise ValueError('Incomplete full test')
    descriptor = json.loads(receipt.read_text(encoding='utf-8'))
    return rows, receipt, dict(cache_fingerprint=reader.manifest['fingerprint'],
        prepared_hashes=reader.manifest['prepared_hashes'], cache_payload_stat=descriptor['payload_stat'],
        cache_manifest_sha256=descriptor['manifest_sha256'], cache_index_sha256=descriptor['index_sha256'],
        model_id=MODEL_ID, native_revision=MODEL_REVISION, processor_policy=processor_policy(processor),
        geometry=geometry, image_token_id=prepared['image_token_id'],
        tokenizer_class=prepared['tokenizer_class'], tokenizer_model_max_length=prepared['tokenizer_model_max_length'])
