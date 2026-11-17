"""Load once, then run individual prepared samples from the validated cache."""
from pathlib import Path

import torch

from .cache import CAMERA_ORDER, FeatureCache
from .model import load_model, processor_policy
from .text import CachedBatchCollator


class InferenceSession:
    def __init__(self, checkpoint, cache_directory, prepared_directory, device='cuda'):
        self.model = load_model(checkpoint, device=device)
        self.device = torch.device(device)
        self.processor = self.model.processor
        identity = dict(backend='smolvlm_no_pruner', version=1,
                        model_id=self.model.model_id, revision=self.model.model_revision,
                        processor_policy=processor_policy(self.processor), geometry=self.model.geometry,
                        storage_dtype='float16', compute_dtype='float32', camera_order=CAMERA_ORDER)
        self.cache = FeatureCache(cache_directory, prepared_directory, identity)
        self.collator = CachedBatchCollator(
            self.processor, image_seq_len=self.model.geometry['visual_tokens_per_view'],
            max_length=self.model.native.config.text_config.max_position_embeddings,
            generation=True, max_new_tokens=512, prompt_camera_order=self.model.prompt_camera_order)
        self.checkpoint = str(Path(checkpoint))

    def describe(self):
        return dict(checkpoint=self.checkpoint, metadata=self.model.checkpoint_metadata,
                    model_id=self.model.model_id, revision=self.model.model_revision,
                    camera_order=list(self.model.prompt_camera_order), dtype='float32', cache_dtype='float16',
                    visual_tokens=self.model.visual_token_len, device=str(self.device),
                    cache_fingerprint=self.cache.manifest['fingerprint'],
                    samples=self.cache.counts)

    @torch.inference_mode()
    def predict(self, split='test', index=0):
        sample = self.cache.sample(split, index)
        item = {**sample, 'visual_features': torch.from_numpy(sample['visual_features'])}
        batch = self.collator([item])
        inputs = {key: batch[key].to(self.device) for key in
                  ('input_ids', 'attention_mask', 'visual_features')}
        inputs['visual_features'] = inputs['visual_features'].float()
        continuation = self.model.generate_from_features(
            **inputs, max_new_tokens=512, num_beams=3, length_penalty=1.0, early_stopping=True)
        prediction = self.processor.tokenizer.batch_decode(continuation, skip_special_tokens=True)[0]
        return dict(sample_id=sample['sample_id'], split=split, index=index,
                    question=sample['question'], ground_truth=sample['answer'],
                    prediction=prediction, camera_order=list(self.model.prompt_camera_order))
