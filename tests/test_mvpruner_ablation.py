from hashlib import sha256
import math

import pytest
import torch

from test_smolvlm_model import tiny_components
from smolvlm_attention.mvpruner_ablation.collector import (
    QuestionBatch, collect, reduce_visual, reference,
)
from smolvlm_attention.mvpruner_ablation.data import camera_label
from smolvlm_attention.mvpruner_ablation.runtime import frame_features
from smolvlm_inference.model import CachedSmolVLM


@pytest.mark.parametrize(('question', 'expected'), [
    ('Is <c1,CAM_BACK,1015.8,520.8> a traffic sign?', ('CAM_BACK', 'tag')),
    ('Are <c1,CAM_BACK,1,2> and <c2,CAM_BACK,2,4> moving?', ('CAM_BACK', 'tag')),
    ('Is <c1,CAM_BACK,1,2> near <c2,CAM_FRONT,2,4>?', (None, 'multiple')),
    ('Are there moving pedestrians to the back left of the ego car?', ('CAM_BACK_LEFT', 'prose')),
    ('Is there any traffic element in the front view?', ('CAM_FRONT', 'prose')),
    ('Can the car turn left?', (None, 'none')),
    ('Is <c1,CAM_BACK,1,2> in the front view?', (None, 'multiple')),
])
def test_camera_labels(question, expected):
    assert camera_label(question)[:2] == expected


def test_entropy_is_reduced_in_the_requested_order():
    visual = torch.tensor([[[0.25, 0], [0, 0.75], [0, 0]],
                           [[0.2, 0.2], [0.1, 0.1], [0.3, 0.3]]])
    valid_queries = torch.tensor([[True, True, False], [True, False, False]])
    mean, token_level, question_level = reduce_visual(visual, valid_queries)
    torch.testing.assert_close(mean, torch.tensor([[0.125, 0.375], [0.2, 0.2]]))
    torch.testing.assert_close(token_level, torch.tensor([0., 1.]))
    unequal_entropy = -(0.25 * math.log2(0.25) + 0.75 * math.log2(0.75))
    torch.testing.assert_close(question_level, torch.tensor([unequal_entropy, 1.]))


def test_collector_matches_eager_and_uses_only_question_tokens():
    native, processor = tiny_components()
    model = CachedSmolVLM(native, processor)
    encoder = QuestionBatch(model)
    questions = ['What?', 'What is visible? A road. A car.']
    encoded = [encoder.encode(question) for question in questions]
    features = torch.randn(6, 16, 8)
    batch = encoder.batch(encoded, features)
    standard = encoder.collator([dict(question=q, visual_features=features) for q in questions])
    assert torch.equal(batch['input_ids'], standard['input_ids'])
    assert torch.equal(batch['attention_mask'], standard['attention_mask'])
    for question, ids, selected in zip(questions, batch['input_ids'], batch['question_mask']):
        assert processor.tokenizer.decode(ids[selected]).strip() == question
    actual, expected = collect(model, batch), reference(model, batch)
    assert actual['visual'].shape == (2, 2, 24)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=1e-6, rtol=1e-4)
    for row, single in enumerate(encoded):
        for key, value in collect(model, encoder.batch([single], features)).items():
            torch.testing.assert_close(actual[key][row:row + 1], value, atol=1e-6, rtol=1e-4)


def test_frame_features_match_native_vision(tmp_path):
    from PIL import Image

    native, processor = tiny_components()
    model = CachedSmolVLM(native, processor)
    images, paths, assets = [], [], {}
    for i in range(6):
        image = Image.new('RGB', (16 + i, 12), (40 + 20 * i, 100, 120))
        path = tmp_path / f'{i}.png'
        image.save(path)
        paths.append(path.name)
        assets[path.name] = dict(sha256=sha256(path.read_bytes()).hexdigest())
        images.append(image)
    prompt = QuestionBatch(model)._render('What is visible?')
    ids = processor.tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)
    pixels = processor.image_processor(images=[images], return_tensors='pt')
    tensor = frame_features(model, paths, assets, tmp_path)
    captured = []
    handle = native.model.connector.register_forward_pre_hook(lambda _, args: captured.append(args[0]))
    try:
        with torch.inference_mode():
            native(**ids, **pixels, use_cache=False)
    finally:
        handle.remove()
    torch.testing.assert_close(tensor, captured[0])
