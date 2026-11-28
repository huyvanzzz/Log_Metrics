"""Method-specific receipts over the original validated read-only input pipeline."""
from copy import deepcopy
import json
from pathlib import Path
from smolvlm_attention.sample_store import atomic_json
from smolvlm_map_predictor import data as original
from .config import METHOD
import spacy
import torch
from smolvlm_predictor_io.compact import CompactCollator

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
                   expected_cache_fingerprint=None, checkpoint_sha256=None, projected_cache=None):
    common = original.prepare_inputs(targets_root, cache_dir, prepared_dir, model, output_dir,
        expected_counts=expected_counts, expected_cache_fingerprint=expected_cache_fingerprint,
        checkpoint_sha256=checkpoint_sha256, projected_cache=projected_cache)
    return adapt_receipt(common)


class PredictorDataset(original.PredictorDataset):
    def __init__(self, receipt, split):
        read_receipt(receipt, include_targets=split != 'test')
        super().__init__(receipt, split)


GIFT_POS_TAGS = {'NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV', 'NUM'}


class _PosFilterMixin:
    def _init_pos(self):
        self._nlp = None
        self._pos_cache = {}

    def _get_nlp(self):
        if self._nlp is None:
            self._nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'lemmatizer'])
        return self._nlp

    def _pos_mask_for_row(self, ids_row, attn_row, q_mask_row):
        q_indices = q_mask_row.nonzero(as_tuple=True)[0]
        if len(q_indices) == 0:
            last_col = attn_row.nonzero(as_tuple=True)[0][-1]
            out = torch.zeros_like(q_mask_row, dtype=torch.bool)
            out[last_col] = True
            return out

        tokenizer = self.processor.tokenizer
        q_token_ids = ids_row[q_indices].tolist()
        cache_key = tuple(q_token_ids)

        if cache_key not in self._pos_cache:
            q_text = tokenizer.decode(q_token_ids, skip_special_tokens=False)
            doc = self._get_nlp()(q_text)
            kept_spans = [(tok.idx, tok.idx + len(tok.text))
                          for tok in doc if tok.pos_ in GIFT_POS_TAGS]

            enc = tokenizer(q_text, add_special_tokens=False, return_offsets_mapping=True)
            rel_indices = []
            if kept_spans and 'offset_mapping' in enc and len(enc['input_ids']) == len(q_indices):
                for tok_pos, (s, e) in enumerate(enc['offset_mapping']):
                    if s < e and any(not (e <= ks or s >= ke) for ks, ke in kept_spans):
                        rel_indices.append(tok_pos)
            else:
                kept_words = {tok.text.lower() for tok in doc if tok.pos_ in GIFT_POS_TAGS}
                for tok_pos, tid in enumerate(q_token_ids):
                    if tokenizer.decode([tid]).strip().lower() in kept_words:
                        rel_indices.append(tok_pos)
            self._pos_cache[cache_key] = rel_indices

        pos_mask = torch.zeros_like(q_mask_row, dtype=torch.bool)
        rel_indices = self._pos_cache[cache_key]
        if rel_indices:
            pos_mask[q_indices[rel_indices]] = True
        else:
            last_col = attn_row.nonzero(as_tuple=True)[0][-1]
            pos_mask[last_col] = True
        return pos_mask

    def _apply_pos_mask(self, result):
        ids = result['input_ids']
        attn = result['attention_mask'].bool()
        q_mask = result['question_mask'].bool()
        result['question_mask'] = torch.stack([
            self._pos_mask_for_row(ids[i], attn[i], q_mask[i])
            for i in range(ids.shape[0])
        ])
        result.pop('question_columns', None)
        result.pop('question_valid', None)
        return result


class PredictorCollator(_PosFilterMixin, original.PredictorCollator):
    def __init__(self, receipt, processor, *, max_new_tokens=0):
        read_receipt(receipt, include_targets=False)
        super().__init__(receipt, processor, max_new_tokens=max_new_tokens)
        self._init_pos()

    def __getstate__(self):
        return {**super().__getstate__(), '_nlp': None}

    def __call__(self, items):
        return self._apply_pos_mask(super().__call__(items))


class POSCompactCollator(_PosFilterMixin, CompactCollator):
    def __init__(self, store, processor, *, max_new_tokens=0):
        super().__init__(store, max_new_tokens=max_new_tokens)
        self.processor = processor
        self._init_pos()

    def __getstate__(self):
        return {**super().__getstate__(), '_nlp': None}

    def __call__(self, items):
        return self._apply_pos_mask(super().__call__(items))