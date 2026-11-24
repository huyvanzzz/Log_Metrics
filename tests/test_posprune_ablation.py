import json
from types import SimpleNamespace

import numpy as np
import torch

from smolvlm_attention.collector import collect_last_query_attention
from smolvlm_attention.posprune_ablation.decoder import attention_difference, generate, mmr
from smolvlm_inference.cache import CAMERA_ORDER, record_key
from smolvlm_inference.text import CachedBatchCollator
from test_smolvlm_model import fixture


def test_calibration_matches_existing_attention_collector():
    model, batch, _ = fixture()
    expected = collect_last_query_attention(model, batch)
    observed = attention_difference(model, batch, 1, 2)
    np.testing.assert_allclose(observed, (expected[:, 0] - expected[:, 1]).numpy(), atol=1e-8)


def test_mmr_uses_relevance_then_penalizes_redundancy():
    scores = torch.tensor([0.05, 0.048, 0.045])
    states = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
    indices, penalties = mmr(scores, states, 2, .9)
    assert indices.tolist() == [0, 2]
    assert penalties.tolist() == [0., 0.]


def test_native_beam_generation_uses_compact_layer_local_cache():
    model, batch, _ = fixture()
    inputs = {key: batch[key] for key in ('input_ids', 'attention_mask', 'visual_features')}
    settings = dict(num_beams=3, max_new_tokens=4, min_new_tokens=4, length_penalty=1., early_stopping=True)
    tokens = model.generate_from_features(**inputs, **settings)
    curve = torch.zeros(model.visual_token_len)
    baseline, timing, _ = generate(model, inputs, settings, 1)
    assert torch.equal(tokens, baseline) and timing['first_beam_step_ms'] > 0
    all_kept, _, _ = generate(model, inputs, settings, 1, method='raw', curve=curve, count=len(curve))
    assert torch.equal(tokens, all_kept)

    text = model.native.model.text_model
    layer_lengths = []
    handle = text.register_forward_hook(lambda _, args, output: layer_lengths.append(
        tuple(output.past_key_values.get_seq_length(layer) for layer in range(2))))
    try:
        pruned, _, details = generate(model, inputs, settings, 1, method='mmr', curve=curve, count=5)
    finally:
        handle.remove()
    assert pruned.shape == tokens.shape
    assert len(set(details['picks'])) == 5
    width = inputs['input_ids'].shape[1]
    assert layer_lengths == [(width + i, width + i - len(curve) + 5) for i in range(4)]
    assert torch.equal(model.generate_from_features(**inputs, **settings), tokens)


def test_paired_run_exports_all_figures_and_selection_indices(tmp_path, monkeypatch):
    from smolvlm_attention.posprune_ablation import report, runtime

    model, _, fixture_item = fixture()
    frames = [dict(sample_id=f'id-{i}', scene_id=scene, question='What is visible?',
                   answer='A road.', camera_paths={camera: f'{i}/{camera}.jpg' for camera in CAMERA_ORDER})
              for i, scene in enumerate(['calibration', 'evaluation', 'evaluation'])]
    for split, rows in [('val', frames[:2]), ('test', frames[2:])]:
        (tmp_path / f'{split}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    features = np.stack([fixture_item['visual_features'].numpy().astype(np.float16)] * len(frames))
    cache = SimpleNamespace(prepared=tmp_path, index={record_key(row): i for i, row in enumerate(frames)}, array=features)
    session = SimpleNamespace(model=model, cache=cache, device=torch.device('cpu'), processor=model.processor,
                              describe=lambda: {'fixture': True},
                              collator=CachedBatchCollator(model.processor, image_seq_len=4, max_length=512,
                                                           max_new_tokens=3))
    monkeypatch.setattr(runtime, 'SHALLOW', 1)
    monkeypatch.setattr(runtime, 'DEEP', 2)
    monkeypatch.setattr(runtime, 'GENERATION', dict(runtime.GENERATION, max_new_tokens=3))
    output = runtime.run(session, tmp_path / 'results', calibration_size=1, evaluation_size=1, keep_counts=(5,))
    config = json.loads((output / 'config.json').read_text(encoding='utf-8'))
    assert config['calibration_ids'] == ['id-0']
    assert config['evaluation_ids'] == ['id-2']
    monkeypatch.setattr(report, '_quality', lambda rows, names: {name: {'overall': 1.} for name in names})
    paths = report.report(output)
    assert len(paths) == 8 and all(path.stat().st_size for path in paths)
    assert len((output / 'indexed_selections.csv').read_text(encoding='utf-8').splitlines()) == 16
    assert 'mmr_5' in (output / 'paired_answers.csv').read_text(encoding='utf-8')
