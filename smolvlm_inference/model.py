"""Strict FP32 inference from six cached native SmolVLM camera features."""
from pathlib import Path
import json
import torch
from torch import nn
from transformers import AutoConfig, AutoProcessor, GenerationConfig, Idefics3ForConditionalGeneration

MODEL_ID = 'HuggingFaceTB/SmolVLM-256M-Instruct'
MODEL_REVISION = '7e3e67edbbed1bf9888184d9df282b700a323964'
CAMERA_ORDER = ['Front', 'Front-Left', 'Front-Right', 'Back', 'Back-Left', 'Back-Right']


def validate_prompt_camera_order(order):
    order = list(order)
    if order not in (CAMERA_ORDER, list(reversed(CAMERA_ORDER))):
        raise ValueError('prompt camera order must be canonical or its exact reverse')
    return order

def _canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _geometry(config):
    vision=config.vision_config
    side=vision.image_size // vision.patch_size
    if vision.image_size % vision.patch_size or side % config.scale_factor:
        raise ValueError('vision geometry must divide evenly into connector patches')
    return dict(num_views=6,vision_tokens=side**2,vision_dim=vision.hidden_size,
                visual_tokens_per_view=side**2 // config.scale_factor**2,
                text_dim=config.text_config.hidden_size)


def _configure_processor(processor,config):
    image=processor.image_processor
    image.do_image_splitting=False
    image.size={'longest_edge':config.vision_config.image_size}
    image.max_image_size={'longest_edge':config.vision_config.image_size}
    processor.image_seq_len=_geometry(config)['visual_tokens_per_view']


def processor_policy(processor):
    """Content identity of preprocessing, vocabulary, and native chat policy."""
    tok = processor.tokenizer
    image = processor.image_processor.to_dict()
    image.pop('_processor_class', None)
    backend=json.loads(tok.backend_tokenizer.to_str())
    backend.pop('padding',None)
    backend.pop('truncation',None)
    return _canonical(dict(image_processor=image, chat_template=processor.chat_template,
        tokenizer=dict(vocab=tok.get_vocab(), backend=backend,
                       special_tokens=tok.special_tokens_map, model_max_length=tok.model_max_length),
        image_seq_len=processor.image_seq_len, camera_order=CAMERA_ORDER,
        fake_image_token=str(processor.fake_image_token), image_token=str(processor.image_token),
        global_image_tag=processor.global_image_tag, image_splitting=False,
        policy='native_six_camera_chat_v1'))


class CachedSmolVLM(nn.Module):
    def __init__(self, native, processor, *, model_id=MODEL_ID, model_revision=None,
                 prompt_camera_order=CAMERA_ORDER):
        super().__init__()
        self.prompt_camera_order = validate_prompt_camera_order(prompt_camera_order)
        self.native, self.processor = native, processor
        self.geometry = _geometry(native.config)
        self.visual_token_len = 6 * self.geometry['visual_tokens_per_view']
        self.model_id, self.model_revision = str(model_id), model_revision
        _configure_processor(processor, native.config)
        self.requires_grad_(False)
        self.eval()

    @property
    def architecture_signature(self):
        # Configs contain a runtime attention backend but no checkpoint identity paths.
        config = self.native.config.to_dict()
        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k,v in obj.items() if k not in
                        {'_name_or_path', 'transformers_version', 'torch_dtype', '_attn_implementation_autoset', 'gradient_checkpointing'}}
            if isinstance(obj, list):
                return [clean(v) for v in obj]
            return obj
        signature = dict(model_id=self.model_id, model_revision=self.model_revision,
            native_config=clean(config), geometry=self.geometry, fusion='no_pruner',
            processor=processor_policy(self.processor), generation_special_tokens={
                key:getattr(self.native.generation_config,key) for key in
                ('bos_token_id','eos_token_id','pad_token_id')})
        if self.prompt_camera_order != CAMERA_ORDER:
            signature['prompt_camera_order'] = self.prompt_camera_order
        return _canonical(signature)


    def _project(self, visual_features, batch):
        shape=(batch,6,self.geometry['vision_tokens'],self.geometry['vision_dim'])
        if tuple(visual_features.shape) != shape:
            raise ValueError(f'visual_features must have shape {shape}')
        connector=self.native.model.connector
        weight=next(connector.parameters())
        dtype = visual_features.dtype if weight.device.type == 'cuda' and torch.is_autocast_enabled() else weight.dtype
        return connector(visual_features.flatten(0,1).to(device=weight.device,dtype=dtype))


    def _context(self, input_ids, extra=0):
        if input_ids.shape[1]+extra > self.native.config.text_config.max_position_embeddings:
            raise ValueError('expanded prompt exceeds native context budget')


    @torch.no_grad()
    def generate_from_features(self, *, input_ids, attention_mask, visual_features, **kwargs):
        kwargs = dict(kwargs)
        generation_config=kwargs.get('generation_config',self.native.generation_config)
        budget=kwargs.get('max_new_tokens',generation_config.max_new_tokens)
        if budget is None:
            budget=max(0,kwargs.get('max_length',generation_config.max_length)-input_ids.shape[1])
        self._context(input_ids,budget)
        projected=self._project(visual_features,input_ids.shape[0])
        embeddings=self.native.get_input_embeddings()(input_ids)
        fused=self.native.model.inputs_merger(input_ids=input_ids,inputs_embeds=embeddings,
                                              image_hidden_states=projected)
        kwargs['use_cache']=True
        output=self.native.generate(input_ids=input_ids,inputs_embeds=fused,
                                    attention_mask=attention_mask,**kwargs)
        if hasattr(output,'sequences'):
            return output.sequences[:,input_ids.shape[1]:]
        return output[:,input_ids.shape[1]:]



def _check_attention(native):
    if (native.model.vision_model.config._attn_implementation != 'eager'
            or native.model.text_model.config._attn_implementation != 'sdpa'):
        raise ValueError('native attention backends must be vision eager and text sdpa')
    vision = [type(layer.self_attn).__name__ for layer in native.model.vision_model.encoder.layers]
    text = [type(layer.self_attn).__name__ for layer in native.model.text_model.layers]
    if (not vision or not text or any(name != 'Idefics3VisionAttention' for name in vision)
            or any(not name.endswith('SdpaAttention') for name in text)):
        raise ValueError('actual native attention modules must be vision eager and text sdpa')


def load_model(checkpoint_path, device='cuda', *, expected_prompt_camera_order=None):
    """Restore a complete canonical checkpoint without downloading pretrained weights."""
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError('checkpoint must be a dictionary')
    signature = checkpoint.get('architecture_signature')
    if not isinstance(signature, dict) or not isinstance(checkpoint.get('model_state_dict'), dict):
        raise ValueError('checkpoint requires full architecture_signature and model_state_dict')
    if signature.get('model_id') != MODEL_ID or signature.get('model_revision') != MODEL_REVISION:
        raise ValueError('checkpoint must use the pinned SmolVLM model and revision')
    order = validate_prompt_camera_order(signature.get('prompt_camera_order', CAMERA_ORDER))
    if expected_prompt_camera_order is not None and order != validate_prompt_camera_order(expected_prompt_camera_order):
        raise ValueError('checkpoint prompt camera order differs from requested model order')
    if signature.get('processor', {}).get('camera_order') != CAMERA_ORDER:
        raise ValueError('processor/cache camera order must remain canonical')
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=False)
    if config.model_type != 'idefics3':
        raise ValueError('native SmolVLM requires Idefics3 configuration')
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=False)
    native = Idefics3ForConditionalGeneration._from_config(config, torch_dtype=torch.float32,
        attn_implementation={'vision_config': 'eager', 'text_config': 'sdpa'})
    native.generation_config = GenerationConfig.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, token=False)
    _check_attention(native)
    model = CachedSmolVLM(native, processor, model_revision=MODEL_REVISION,
                         prompt_camera_order=order)
    if model.architecture_signature != signature:
        raise ValueError('checkpoint architecture_signature does not match native model and processor')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.checkpoint_metadata = checkpoint.get('metadata', {})
    if not isinstance(model.checkpoint_metadata, dict):
        raise ValueError('checkpoint metadata must be a dictionary')
    model.checkpoint_path = str(checkpoint_path)
    return model.to(device=device, dtype=torch.float32).eval()
