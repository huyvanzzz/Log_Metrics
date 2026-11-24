"""Method-specific receipts over the original validated read-only input pipeline."""
from copy import deepcopy
import json
from pathlib import Path
from smolvlm_attention.sample_store import atomic_json
from smolvlm_map_predictor import data as original
from .config import METHOD


def read_receipt(path, *, include_targets=True):
    receipt = json.loads(Path(path).read_text(encoding='utf-8'))
    identity = receipt.get('identity', {})
    if (identity.get('method') != METHOD or
            identity.get('predictor_query_policy') != METHOD['query_policy'] or
            identity.get('predictor_input_boundary') != 'decoder_block_output' or
            identity.get('predictor_input_layer_index') != METHOD['input_layer_index'] or
            identity.get('production_prune_after_layer_index') != METHOD['prune_after_layer_index']):
        raise ValueError('Unsupported after2 receipt method')
    return original.read_receipt(path, include_targets=include_targets)


def adapt_receipt(common_receipt, output_path=None):
    receipt = deepcopy(original.read_receipt(common_receipt))
    receipt['identity'].update(method=deepcopy(METHOD), predictor_query_policy=METHOD['query_policy'],
        predictor_input_boundary='decoder_block_output', predictor_input_layer_index=2,
        production_prune_after_layer_index=2)
    path = Path(output_path) if output_path is not None else Path(common_receipt).with_name('after2_receipt.json')
    if path.resolve() == Path(common_receipt).resolve():
        raise ValueError('after2 receipt must have a distinct path')
    atomic_json(path, receipt)
    return path


def prepare_inputs(targets_root, cache_dir, prepared_dir, model, output_dir, *, expected_counts=None,
                   expected_cache_fingerprint=None, checkpoint_sha256=None):
    common = original.prepare_inputs(targets_root, cache_dir, prepared_dir, model, output_dir,
        expected_counts=expected_counts, expected_cache_fingerprint=expected_cache_fingerprint,
        checkpoint_sha256=checkpoint_sha256)
    return adapt_receipt(common)


class PredictorDataset(original.PredictorDataset):
    def __init__(self, receipt, split):
        read_receipt(receipt, include_targets=split != 'test')
        super().__init__(receipt, split)


class PredictorCollator(original.PredictorCollator):
    def __init__(self, receipt, processor, *, max_new_tokens=0):
        read_receipt(receipt, include_targets=False)
        super().__init__(receipt, processor, max_new_tokens=max_new_tokens)
