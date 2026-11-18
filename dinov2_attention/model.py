"""Strict inference-only port of the cached DINOv2/native T5 baseline.

Parameter names and feature enrichment order match the unpruned training model.
The vision encoder and all pruning/training machinery are deliberately absent.
"""
from pathlib import Path
import hashlib
import math

import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer, T5ForConditionalGeneration

from .config import (CHECKPOINT_SHA256, TEXT_MODEL_ID, TEXT_REVISION,
                     VISION_MODEL_ID, VISION_REVISION)


class CameraEmbeddings(nn.Module):
    def __init__(self, num_views, tokens_per_view, vision_dim):
        super().__init__()
        self.num_views, self.tokens_per_view, self.vision_dim = num_views, tokens_per_view, vision_dim
        self.camera_embeddings = nn.Embedding(num_views, vision_dim)

    def forward(self, visual):
        if visual.ndim != 4 or tuple(visual.shape[1:]) != (
                self.num_views, self.tokens_per_view, self.vision_dim):
            raise ValueError('Incompatible visual feature geometry')
        ids = torch.arange(self.num_views, device=visual.device)
        return visual + self.camera_embeddings(ids).view(1, self.num_views, 1, self.vision_dim)


class LinearProjector(nn.Module):
    def __init__(self, input_dim, d_model, seq_len):
        super().__init__()
        self.input_dim, self.seq_len = input_dim, seq_len
        self.proj = nn.Linear(input_dim, d_model)

    def forward(self, features):
        if features.ndim != 3 or tuple(features.shape[1:]) != (self.seq_len, self.input_dim):
            raise ValueError('Incompatible projector feature geometry')
        return self.proj(features)


class CachedDinoT5(nn.Module):
    def __init__(self, text_model, *, num_views=6, vision_dim=384, seq_len=64):
        super().__init__()
        self.text_model = text_model
        self.seq_len = seq_len
        self.visual_fusion = CameraEmbeddings(num_views, seq_len, vision_dim)
        self.projector = LinearProjector(vision_dim, text_model.config.d_model, num_views * seq_len)
        self.modal_embeddings = nn.Embedding(2, text_model.config.d_model)
        side = math.isqrt(seq_len)
        self.row_embeddings = nn.Embedding(side, vision_dim) if side * side == seq_len else None
        self.col_embeddings = nn.Embedding(side, vision_dim) if side * side == seq_len else None

    @property
    def visual_token_len(self):
        return self.visual_fusion.num_views * self.seq_len

    def add_spatial_embeddings(self, visual_features):
        if self.row_embeddings is None or self.col_embeddings is None:
            return visual_features
        side = self.row_embeddings.num_embeddings
        rows = torch.arange(side, device=visual_features.device).repeat_interleave(side)
        cols = torch.arange(side, device=visual_features.device).repeat(side)
        spatial = self.row_embeddings(rows) + self.col_embeddings(cols)
        return visual_features + spatial.view(1, 1, self.seq_len, -1)

    def _native_inputs(self, input_ids, attention_mask, visual_features):
        # Validate before broadcasting, to avoid unhelpful tensor-size exceptions.
        expected = (input_ids.shape[0], self.visual_fusion.num_views, self.seq_len,
                    self.visual_fusion.vision_dim)
        if tuple(visual_features.shape) != expected:
            raise ValueError(f'Incompatible visual feature geometry: expected {expected}')
        spatial = self.add_spatial_embeddings(visual_features)
        enriched = self.visual_fusion(spatial)
        projected = self.projector(enriched.reshape(enriched.shape[0], -1, enriched.shape[-1]))
        text = self.text_model.get_input_embeddings()(input_ids)
        visual = projected + self.modal_embeddings(torch.ones(
            projected.shape[:2], dtype=torch.long, device=projected.device))
        text_modal = text + self.modal_embeddings(torch.zeros_like(input_ids))
        embeds = torch.cat([visual, text_modal], dim=1)
        mask = torch.cat([attention_mask.new_ones(attention_mask.shape[0], self.visual_token_len),
                          attention_mask], dim=1)
        return embeds, mask, spatial, projected, text

    def generate_from_features(self, *, input_ids, attention_mask, visual_features,
                               max_new_tokens=1, num_beams=1, **kwargs):
        embeds, mask, *_ = self._native_inputs(input_ids, attention_mask, visual_features)
        return self.text_model.generate(inputs_embeds=embeds, attention_mask=mask,
            max_new_tokens=max_new_tokens, num_beams=num_beams, **kwargs)


def expected_signature():
    return {
        'vision.name': 'dinov2_small_avgpool', 'vision.model_id': VISION_MODEL_ID,
        'vision.revision': VISION_REVISION, 'vision.output_dim': 384,
        'vision.seq_len': 64, 'vision.image_size': 224,
        'vision.feature_extraction': {
            'output_layer': 'last_hidden_state', 'prefix_tokens_to_remove': 1,
            'source_grid': [16, 16], 'pooling': 'avg_pool2d', 'kernel_size': 2,
            'stride': 2, 'output_grid': [8, 8]},
        'projector.type': 'linear', 'text.model_id': TEXT_MODEL_ID,
        'text.revision': TEXT_REVISION, 'text.d_model': 384, 'fusion.name': 'unpruned',
    }


def validate_checkpoint_metadata(metadata):
    signature = metadata.get('architecture_signature')
    if not isinstance(signature, dict):
        raise ValueError('Missing metadata.architecture_signature')
    expected = expected_signature()
    # Check fusion first so a pruned checkpoint has an actionable error.
    for key in ['fusion.name', *[k for k in expected if k != 'fusion.name']]:
        if signature.get(key) != expected[key]:
            raise ValueError(f'Checkpoint architecture_signature mismatch: {key}')
    if set(signature) != set(expected):
        raise ValueError('Unexpected architecture_signature fields')
    saved = (metadata.get('tokenizer') or {}).get('less_than_token')
    if not isinstance(saved, dict) or saved.get('token') != '<' or saved.get('normal_token') is not True:
        raise ValueError('Missing or incompatible tokenizer less_than_token metadata')
    return signature


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(checkpoint_path, device='cuda', *, verify_checkpoint_hash=True):
    """Load only fine-tuned weights. Optimizer, RNG and training state are unused."""
    if verify_checkpoint_hash and _sha256(checkpoint_path) != CHECKPOINT_SHA256:
        raise ValueError('Checkpoint SHA-256 mismatch')
    # Historical checkpoint includes NumPy/Python RNG objects, so weights_only is
    # incompatible; the pinned file is verified above before deserializing it.
    payload = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get('model'), dict):
        raise ValueError('Checkpoint must contain model state and metadata')
    metadata = payload.get('metadata', {})
    validate_checkpoint_metadata(metadata)
    config = AutoConfig.from_pretrained(TEXT_MODEL_ID, revision=TEXT_REVISION)
    actual = (config.model_type, config.d_model, config.num_layers, config.num_decoder_layers,
              config.num_heads, config.d_kv, config.decoder_start_token_id,
              config.pad_token_id, config.eos_token_id, config.vocab_size)
    if actual != ('t5', 384, 4, 4, 8, 64, 0, 0, 1, 32128):
        raise ValueError(f'Unexpected pinned T5 architecture geometry: {actual}')
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID, revision=TEXT_REVISION)
    if '<' not in tokenizer.get_vocab():
        tokenizer.add_tokens('<')
    token_id = tokenizer.convert_tokens_to_ids('<')
    saved = metadata['tokenizer']['less_than_token']
    if (saved.get('token_id') != token_id or token_id != 32100 or
            len(tokenizer) > config.vocab_size or tokenizer.pad_token_id != 0 or
            tokenizer.eos_token_id != 1):
        raise ValueError('Tokenizer identity/less_than_token mismatch; refusing embedding repair')
    native = T5ForConditionalGeneration(config)
    model = CachedDinoT5(native)
    state = payload['model']
    aliases = ['text_model.shared.weight', 'text_model.encoder.embed_tokens.weight',
               'text_model.decoder.embed_tokens.weight', 'text_model.lm_head.weight']
    for name in aliases:
        if name not in state:
            raise ValueError(f'Missing tied weight: {name}')
        if not torch.equal(state[name], state[aliases[0]]):
            raise ValueError(f'Inconsistent tied weight: {name}')
    model.load_state_dict(state, strict=True)
    native.tie_weights()
    for module in (native.encoder.embed_tokens, native.decoder.embed_tokens, native.lm_head):
        if module.weight is not native.shared.weight:
            raise ValueError('T5 tied weight identity mismatch')
    model.checkpoint_metadata = metadata
    model.architecture_signature = metadata['architecture_signature']
    model.requires_grad_(False)
    return model.to(device=device, dtype=torch.float32).eval(), tokenizer
