import json

import torch
import pytest
from transformers import GenerationConfig

from smolvlm_attention.mvpruner_ablation.collector import QuestionBatch
from smolvlm_attention.visipruner_ablation.decoder import Decoder, visual_weights
from smolvlm_attention.visipruner_ablation.design import VARIANTS, Variant, merge_positions, select_indices
from smolvlm_attention.visipruner_ablation.handoff import PART1, handoff
from smolvlm_attention.visipruner_ablation import launcher, runtime
from smolvlm_attention.mvpruner_ablation.data import file_sha256
from smolvlm_inference.model import CachedSmolVLM
from test_smolvlm_model import tiny_components


def test_matrix_sampling_and_merge_strata():
    assert len({variant.name for variant in VARIANTS}) == len(VARIANTS) == 23
    selected = list(merge_positions().values())
    assert len({item['camera'] for item in selected}) == 3
    assert [item['row'] in band for item, band in zip(selected, (range(3), range(3, 5), range(5, 8)))] == [True] * 3
    records = [(dict(question_type='perception', A='Yes.'), {})] * 12
    records += [(dict(question_type='planning', A='Turn left.'), {})] * 4
    sample = select_indices(records, 4)
    assert sample == select_indices(records, 4)
    assert len(sample) == 4 and sum(index < 12 for index in sample) == 3


def test_part1_handoff_autodetects_completed_run_and_preserves_part2(tmp_path):
    source = tmp_path / 'input' / 'saved-notebook' / 'visipruner_ablation_outputs'
    output = tmp_path / 'working'
    frame = 'a' * 64

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    write(source / 'manifest.json', dict(frames={frame: {}}, samples=1))
    write(source / 'run.json', dict(manifest=file_sha256(source / 'manifest.json')))
    for variant in PART1:
        folder = source / 'predictions' / variant.name
        write(folder / f'{frame}.json', [{'id': 0}])
        write(folder / 'complete.json', dict(samples=1, variant=variant.name))
    write(source / 'rankings' / f'{frame}.npz', [0])
    with pytest.raises(ValueError, match='Missing Part 1 artifact'):
        handoff(output, source)
    assert not output.exists()
    write(source / 'projected' / f'{frame}.npy', [1])

    handoff(output, input_root=tmp_path / 'input')
    assert (output / 'rankings' / f'{frame}.npz').read_bytes() == (source / 'rankings' / f'{frame}.npz').read_bytes()
    part2 = output / 'predictions' / VARIANTS[12].name / f'{frame}.json'
    write(part2, [{'id': 7}])
    handoff(output, source)
    assert json.loads(part2.read_bytes()) == [{'id': 7}]

    write(output / 'run.json', dict(manifest='different'))
    with pytest.raises(ValueError, match='Existing Part 1 artifact differs'):
        handoff(output, source)


@pytest.mark.parametrize('devices', [1, 2])
def test_pair_launches_two_workers_only_when_two_gpus_are_available(monkeypatch, devices):
    variants = ('shallow-top10', 'shallow-keep10')
    launches = []
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: devices)

    def record(command):
        launches.append((command[command.index('--variant') + 1], command[-1]))

    class Worker:
        def __init__(self, command):
            record(command)

        def wait(self):
            assert len(launches) == 2
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(launcher.subprocess, 'Popen', Worker)
    monkeypatch.setattr(launcher.subprocess, 'run', lambda cmd, check: record(cmd))
    launcher.run_pair(variants, output='out', image_root='images', batch_size=8, limit=128, checkpoint=None)
    assert launches == [(variants[0], '0'), (variants[1], str(devices - 1))]


def test_single_gpu_run_metadata_can_resume_with_pair_runner(tmp_path):
    path = tmp_path / 'run.json'
    current = dict(manifest='data', checkpoint='weights', gpu='Tesla T4',
                   code=runtime._code_identity(), max_new_tokens=128)
    legacy = dict(current, code=runtime._code_identity(runtime._SINGLE_GPU_RUNTIME))
    path.write_text(json.dumps(legacy), encoding='utf-8')
    runtime._bind(path, current)
    assert json.loads(path.read_bytes()) == legacy
    with pytest.raises(ValueError, match='measurement code changed'):
        runtime._bind(path, dict(current, max_new_tokens=512))


def test_mass_preserved_with_underflow_and_zero_visual_mass():
    probs = torch.tensor([[0.4, 0.0, 0.0], [0.4, 0.1, 0.1], [0.0, 0.0, 0.0]])
    logits = torch.tensor([[0.0, -140.0, -141.0], [0.0, -1.0, -1.0], [0.0, -2.0, -2.0]])
    keep = torch.tensor([[False, True, True]])
    edited, fallback = visual_weights(probs, logits, keep, 'top10', preserve_mass=True)
    assert fallback.flatten().tolist() == [True, False, False]
    assert not torch.any(edited[:, 0])
    assert edited[0, 1] > edited[0, 2] > 0
    torch.testing.assert_close(edited.sum(1), probs.sum(1))
    merge, _ = visual_weights(probs, logits, None, 'merge', recipient=torch.zeros((3, 1), dtype=torch.long))
    torch.testing.assert_close(merge[:, 0], probs.sum(1))
    assert not torch.any(merge[:, 1:])


def test_baseline_greedy_matches_native_and_deep_cache_lengths():
    native, processor = tiny_components()
    model = CachedSmolVLM(native, processor)
    encoder = QuestionBatch(model)
    decoder = Decoder(model, encoder)
    encoded = [encoder.encode(q) for q in ('What?', 'What is visible? A car.')]
    features = torch.randn(6, 16, 8)
    with torch.no_grad():
        projected = model._project(features.unsqueeze(0), 1)
    batch = encoder.batch(encoded, features)
    source = native.generation_config
    config = GenerationConfig(bos_token_id=source.bos_token_id, eos_token_id=source.eos_token_id,
                              pad_token_id=source.pad_token_id, max_new_tokens=3, do_sample=False, num_beams=1)
    native_tokens = model.generate_from_features(**{key: batch[key] for key in
        ('input_ids', 'attention_mask', 'visual_features')}, generation_config=config)
    actual, ranks, _, _ = decoder.generate(encoded, projected, VARIANTS[0], 3)
    torch.testing.assert_close(actual, native_tokens)
    decoder.verify_native(encoded, projected, VARIANTS[0])
    assert ranks[1].shape == (2, 24)
    intervention = Variant('top', 'Top', (1,), 'top10', preserve_mass=True)
    _, _, mass, _ = decoder.generate(encoded, projected, intervention, 3, rankings=ranks)
    assert (mass[:, 0] > 0).all()

    width = batch['input_ids'].shape[1]
    for mode, expected in (('skip', [width - 24, width]), ('discard', [width - 24, width - 24])):
        seen = []
        def capture(_, args, kwargs):
            seen.append((args[0].shape[1], kwargs['attention_mask'].shape[-1],
                         kwargs['position_ids'][:, -1].tolist()))
        handles = [layer.register_forward_pre_hook(capture, with_kwargs=True) for layer in decoder.text.layers]
        try:
            decoder.generate(encoded, projected, Variant(mode, mode, (1,), mode), 3)
        finally:
            for handle in handles:
                handle.remove()
        assert [entry[0] for entry in seen[:2]] == expected
        assert [entry[1] for entry in seen[2:4]] == [n + 1 for n in expected]
        assert seen[0][2] == seen[1][2] == [len(ids) - 1 for ids, _ in encoded]
