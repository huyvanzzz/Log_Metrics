import copy
import importlib
import sys
from pathlib import Path
import pytest
import torch

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import (Idefics3Config, Idefics3ForConditionalGeneration,
    Idefics3ImageProcessor, Idefics3Processor, PreTrainedTokenizerFast)
CAMERA_ORDER = ['Front','Front-Left','Front-Right','Back','Back-Left','Back-Right']

CHAT_TEMPLATE = "<|im_start|>{% for message in messages %}{{message['role'] | capitalize}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"

def tiny_components(*, dropout=0.0):
    special = ['<|im_start|>', '<|im_end|>', '<end_of_utterance>', '<image>',
               '<fake_token_around_image>', '<global-img>', '<unk>']
    backend = Tokenizer(models.BPE(unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        [' '.join(CAMERA_ORDER), 'User: Assistant: What is visible? A road. A car.',
         'What changed? A person. <c1,CAM_FRONT,1>'] * 3,
        trainers.BpeTrainer(vocab_size=360, special_tokens=special,
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token=special[0], pad_token=special[1],
        eos_token=special[2], unk_token=special[-1], chat_template=CHAT_TEMPLATE,
        model_max_length=512,
    )
    processor = Idefics3Processor(
        Idefics3ImageProcessor(do_image_splitting=False, size={'longest_edge': 16},
                              max_image_size={'longest_edge': 16}),
        tokenizer=tokenizer, image_seq_len=4, chat_template=CHAT_TEMPLATE,
    )
    config = Idefics3Config(
        vision_config=dict(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                           num_attention_heads=2, image_size=16, patch_size=4),
        text_config=dict(model_type='llama', hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id,
                         bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                         attention_dropout=dropout, max_position_embeddings=512),
        scale_factor=2, image_token_id=tokenizer.convert_tokens_to_ids('<image>'),
        pad_token_id=tokenizer.pad_token_id,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = Idefics3ForConditionalGeneration._from_config(
            config, attn_implementation={'vision_config': 'eager', 'text_config': 'sdpa'})
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    return model, processor

def modules():
    return importlib.import_module('smolvlm_inference.model'), importlib.import_module('smolvlm_inference.text')


def test_standalone_model_is_available():
    assert importlib.util.find_spec('smolvlm_inference.model') is not None


def fixture():
    model_module, text = modules()
    native, processor = tiny_components()
    model = model_module.CachedSmolVLM(native, processor).eval()
    item = dict(question='What is visible?', answer='A road.', visual_features=torch.randn(6,16,8), sample_id='0', eval_id=0)
    batch = text.CachedBatchCollator(processor, image_seq_len=4, max_length=512, max_new_tokens=3)([item])
    return model, batch, item


def test_signature_and_prompt_match_source():
    model, batch, item = fixture()
    # Exercise native processor expansion independently of the cache collator.
    from PIL import Image
    _, text = modules()
    messages = text.chat_messages(item['question'])
    prompt = model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    expected = model.processor(text=[prompt], images=[[Image.new('RGB',(16,16),(30,40,50)) for _ in range(6)]], return_tensors='pt')
    assert torch.equal(batch['input_ids'], expected['input_ids'])
    assert torch.equal(batch['attention_mask'], expected['attention_mask'])
    assert model.architecture_signature['processor']['camera_order'] == CAMERA_ORDER
    assert 'labels' not in batch
    assert not hasattr(model, 'extract_features')


@pytest.mark.parametrize('beams', [1,3])
def test_generation_parity_and_single_fusion_without_vision(beams):
    model, batch, _ = fixture()
    args = {key:batch[key] for key in ('input_ids','attention_mask','visual_features')}
    native = model.native
    fused = native.model.inputs_merger(input_ids=batch['input_ids'],
        inputs_embeds=native.get_input_embeddings()(batch['input_ids']),
        image_hidden_states=native.model.connector(batch['visual_features'].flatten(0,1)))
    expected = native.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
        inputs_embeds=fused, max_new_tokens=3, min_new_tokens=3, num_beams=beams, use_cache=True)[:,batch['input_ids'].shape[1]:]
    calls = []
    merger_calls = []
    merger = native.model.inputs_merger
    def counted_merger(*args, **kwargs):
        merger_calls.append(1)
        return merger(*args, **kwargs)
    native.model.inputs_merger = counted_merger
    handle = model.native.model.connector.register_forward_hook(lambda *args:calls.append(1))
    def no_vision(*args, **kwargs):
        raise AssertionError('vision must not run')
    model.native.model.vision_model.forward = no_vision
    try:
        actual = model.generate_from_features(**args, max_new_tokens=3, min_new_tokens=3, num_beams=beams)
    finally:
        handle.remove()
    assert torch.equal(actual, expected)
    assert calls == [1]
    assert merger_calls == [1]
    assert not actual.requires_grad


def checkpoint(tmp_path, monkeypatch):
    module, _ = modules()
    model, _, _ = fixture()
    model.model_revision = module.MODEL_REVISION
    # Native generation metadata can differ from text_config defaults.
    model.native.generation_config.bos_token_id = 8
    model.native.generation_config.eos_token_id = 9
    model.native.generation_config.pad_token_id = 10
    generation_config = copy.deepcopy(model.native.generation_config)
    monkeypatch.setattr(module.GenerationConfig, 'from_pretrained',
        lambda *args, **kwargs: copy.deepcopy(generation_config))
    config = copy.deepcopy(model.native.config)
    processor = copy.deepcopy(model.processor)
    monkeypatch.setattr(module.AutoConfig, 'from_pretrained', lambda *args, **kwargs: copy.deepcopy(config))
    monkeypatch.setattr(module.AutoProcessor, 'from_pretrained', lambda *args, **kwargs: copy.deepcopy(processor))
    path = tmp_path/'model.pt'
    data = dict(architecture_signature=model.architecture_signature, model_state_dict=model.state_dict(), metadata=dict(stage='finetune', epoch=2))
    torch.save(data, path)
    return module, model, data, path


def test_load_model_full_checkpoint_only(tmp_path, monkeypatch):
    module, original, data, path = checkpoint(tmp_path, monkeypatch)
    model = module.load_model(path, device='cpu')
    assert model.architecture_signature == original.architecture_signature
    assert model.checkpoint_metadata['epoch'] == 2
    assert not model.training
    assert all(p.dtype == torch.float32 and not p.requires_grad for p in model.parameters())
    for key, value in original.state_dict().items():
        assert torch.equal(value, model.state_dict()[key])


@pytest.mark.parametrize('mutation', ['signature', 'camera', 'missing', 'unexpected'])
def test_reject_incompatible_checkpoint(tmp_path, monkeypatch, mutation):
    module, model, data, path = checkpoint(tmp_path, monkeypatch)
    if mutation == 'signature':
        data['architecture_signature']['geometry']['vision_dim'] += 1
    elif mutation == 'camera':
        data['architecture_signature']['prompt_camera_order'] = list(reversed(module.CAMERA_ORDER))
    elif mutation == 'missing':
        data['model_state_dict'].pop(next(iter(data['model_state_dict'])))
    else:
        data['model_state_dict']['unexpected'] = torch.ones(1)
    torch.save(data, path)
    with pytest.raises((ValueError, RuntimeError)):
        module.load_model(path, device='cpu')


def test_session_reads_cache_and_generates_real_continuation(tmp_path, monkeypatch):
    import hashlib
    import json
    import numpy as np
    from smolvlm_inference import session
    from smolvlm_inference.cache import canonical, digest, record_key
    module, model, _, path = checkpoint(tmp_path, monkeypatch)
    # Keep the real checkpoint loader and native model; enlarge only fixture context.
    config = copy.deepcopy(model.native.config)
    config.text_config.max_position_embeddings = 2048
    model.native.config.text_config.max_position_embeddings = 2048
    monkeypatch.setattr(module.AutoConfig, 'from_pretrained', lambda *a, **kw: copy.deepcopy(config))
    torch.save(dict(architecture_signature=model.architecture_signature,
                    model_state_dict=model.state_dict(), metadata={'epoch': 2}), path)
    prepared, cache = tmp_path / 'prepared', tmp_path / 'cache'
    prepared.mkdir()
    cache.mkdir()
    row = dict(question='What is visible?', answer='A road.', sample_id='test-0', eval_id=0,
               camera_paths={camera: f'images/{camera}.jpg' for camera in CAMERA_ORDER})
    (prepared / 'manifest.json').write_text(json.dumps({'counts': dict(train=1, val=1, test=1)}))
    for split in ('train', 'val', 'test'):
        (prepared / f'{split}.jsonl').write_text(json.dumps(row) + '\n')
    (cache / 'index.json').write_bytes(canonical({record_key(row): 0}))
    values = np.random.default_rng(1).normal(size=(1, 6, 16, 8)).astype(np.float16)
    values.tofile(cache / 'features.float16.memmap')
    manifest = dict(backend='smolvlm_no_pruner', version=1, model_id=model.model_id,
                    revision=model.model_revision, processor_policy=module.processor_policy(model.processor),
                    geometry=model.geometry, storage_dtype='float16', compute_dtype='float32',
                    camera_order=CAMERA_ORDER, shape=[1, 6, 16, 8], num_samples=1,
                    record_counts=dict(train=1, val=1, test=1),
                    prepared_hashes={p.name: digest(p) for p in prepared.iterdir()},
                    index_sha256=digest(cache / 'index.json'))
    manifest['fingerprint'] = hashlib.sha256(canonical(manifest)).hexdigest()
    (cache / 'manifest.json').write_text(json.dumps(manifest))
    loaded = session.InferenceSession(path, cache, prepared, device='cpu')
    result = loaded.predict('test', 0)
    assert result['sample_id'] == 'test-0'
    assert result['camera_order'] == CAMERA_ORDER
    assert result['question'] == row['question']
    assert isinstance(result['prediction'], str)
    assert loaded.describe()['metadata'] == {'epoch': 2}
