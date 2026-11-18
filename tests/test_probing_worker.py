"""CPU orchestration tests; all generated artifacts stay in pytest temp dirs."""
import json
from types import SimpleNamespace

import pytest
import torch


class CharacterTokenizer:
    eos_token_id = 2
    all_special_ids = [0, 1, 2, 3, 4]

    def encode(self, text, **kwargs):
        return [ord(char) + 10 for char in text]

    def decode(self, ids, **kwargs):
        return ''.join(chr(token-10) for token in ids if token not in self.all_special_ids)


def target(phrase, camera='Front'):
    return dict(object_tag=phrase, reference_span=dict(text=phrase, start=0, end=len(phrase)),
                camera=camera, bbox=[0, 0, 16, 16], image_size=[16, 16])


@pytest.fixture
def harness(tmp_path, monkeypatch, request):
    from smolvlm_probing import measurement, preflight, runtime, worker
    from smolvlm_probing.config import CAMERA_ORDER, RANKING_LAYERS
    from smolvlm_probing.state import load_stage
    tokenizer = CharacterTokenizer()
    rows = [dict(sample_id=sid, input_ids=[20+index], feature_index=index,
                 frame_id=sid, question='What?', answer='car bus',
                 camera_paths={camera: camera+'.jpg' for camera in CAMERA_ORDER},
                 localization_targets=dict(with_coordinates=[target('car')],
                                          without_coordinates=[target('bus', 'Front-Left')] if index == 0 else []))
            for index, sid in enumerate(('valid', 'empty', 'invalid'))]
    raw = {
        20: tokenizer.encode('ca') + [1] + tokenizer.encode('r bus') + [4],
        21: [4],
        22: tokenizer.encode('car') + [3, 4],
    }
    generation_calls, baseline_calls, causal_calls = [], [], []
    def generate(**kwargs):
        codes = kwargs['input_ids'][:, 0].tolist()
        generation_calls.append(codes)
        width = max(len(raw[code]) for code in codes)
        return torch.tensor([raw[code] + [0]*(width-len(raw[code])) for code in codes])
    model = SimpleNamespace(processor=SimpleNamespace(tokenizer=tokenizer),
        native=SimpleNamespace(config=SimpleNamespace(image_token_id=3),
            generation_config=SimpleNamespace(eos_token_id=[4], bos_token_id=1, pad_token_id=0)),
        generate_from_features=generate)
    monkeypatch.setattr(worker, '_inference_context', lambda run, selected: (model, None))
    def batch(reader, selected, processor, device):
        return dict(input_ids=torch.tensor([row['input_ids'] for row in selected]),
                    attention_mask=torch.ones((len(selected), 1), dtype=torch.long),
                    visual_features=torch.zeros((len(selected), 1)))
    monkeypatch.setattr(worker, '_batch', batch)
    monkeypatch.setattr(preflight, 'numerical_preflight', lambda *args, **kwargs: {})
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *args: 'CPU fixture')
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda *args: 0)
    def collect(model, batch, generated_ids, *, target_queries):
        baseline_calls.append((generated_ids, target_queries))
        size = len(generated_ids)
        maps = []
        for targets in target_queries:
            values = torch.zeros((len(targets), 6, 384))
            for index in range(len(targets)):
                values[index, :, index*64] = .5
            maps.append(values)
        return dict(scores=[dict(mean_logprob=-1., token_logprobs=[-1.] * sum(token not in tokenizer.all_special_ids for token in ids),
                                token_count=sum(token not in tokenizer.all_special_ids for token in ids)) for ids in generated_ids],
                    ranking=torch.zeros((size, 5, 384)), target_maps=maps,
                    ranking_layers=list(RANKING_LAYERS))
    monkeypatch.setattr(measurement, 'collect_baseline', collect)
    def causal(model, batch, generated_ids, baseline, *, sample_ids):
        causal_calls.append(sample_ids)
        def condition():
            return dict(indices=list(range(32)), remove_score=-2., keep_score=-2., comp=.5, sgap=.5)
        return [dict(baseline_score=-1., token_count=score['token_count'],
            layers=[dict(layer=layer, top=condition(), bottom=condition()) for layer in RANKING_LAYERS],
            random=dict(repeats=[condition() for _ in range(3)], comp=.5, sgap=.5)) for score in baseline['scores']]
    monkeypatch.setattr(measurement, 'measure_causal', causal)
    config = dict(version=1, output_root=str(tmp_path), shard_count=getattr(request, 'param', 1),
                  identity={stage: stage+'_identity' for stage in ('generation', 'baseline', 'causal', 'localization')},
                  measurement_sources=runtime.measurement_sources(), versions=runtime._library_versions(),
                  dataset_fingerprint='fixture_dataset')
    path = runtime.write_run(tmp_path, config, rows)
    def loaded(stage):
        return load_stage(tmp_path, stage, [row['sample_id'] for row in rows], config['identity'][stage])
    return SimpleNamespace(path=path, root=tmp_path, rows=rows, worker=worker, measurement=measurement,
        runtime=runtime, loaded=loaded, collect=collect, generation_calls=generation_calls,
        baseline_calls=baseline_calls, causal_calls=causal_calls)


@pytest.mark.parametrize('failure', ['error', 'oom'])
def test_generation_is_durable_before_replay_failure_and_never_regenerated(harness, monkeypatch, failure):
    h = harness
    attempts = []
    def measured(*args, **kwargs):
        attempts.append(1)
        assert set(h.loaded('generation')) == {'valid', 'empty', 'invalid'}
        if len(attempts) == 1:
            if failure == 'oom':
                raise torch.cuda.OutOfMemoryError('fixture OOM after durable generation')
            raise RuntimeError('fixture replay failure')
        return h.collect(*args, **kwargs)
    monkeypatch.setattr(h.measurement, 'collect_baseline', measured)
    if failure == 'error':
        with pytest.raises(RuntimeError, match='fixture replay failure'):
            h.worker.run_worker(h.path, 'baseline', 0)
        assert not h.loaded('baseline')
        assert json.loads((h.root/'logs/baseline_0.json').read_text())['status'] == 'failed'
    h.worker.run_worker(h.path, 'baseline', 0)
    assert len(h.generation_calls) == 1 and len(attempts) == 2
    assert set(h.loaded('baseline')) == {'valid', 'empty', 'invalid'}
    before = {path: path.read_bytes() for path in (h.root/'state').rglob('*.npz')}
    h.worker.run_worker(h.path, 'baseline', 0)
    assert len(attempts) == 2 and len(h.generation_calls) == 1
    assert all(path.read_bytes() == payload for path, payload in before.items())


def test_stage_pipeline_audit_maps_and_invalid_empty_exclusions(harness):
    h = harness
    for stage in ('baseline', 'causal', 'localization'):
        h.worker.run_worker(h.path, stage, 0)
    baseline = h.loaded('baseline')
    valid, arrays = baseline['valid']
    assert valid['generated_ids'][2] == 1 and valid['special_token_mask'][2] is True
    assert valid['stop_reason'] == 'eos'
    assert baseline['empty'][0]['generation_status'] == 'empty'
    assert baseline['invalid'][0]['generation_status'] == 'invalid_multimodal_generation'
    assert arrays['target_maps'].shape == (2, 6, 384)
    assert h.baseline_calls[0][1] == [[[0, 1, 3], [5, 6, 7]]]
    assert h.causal_calls == [['valid']]
    causal = h.loaded('causal')
    assert causal['empty'][0] == dict(sample_id='empty', status='excluded', reason='empty')
    assert causal['invalid'][0]['reason'] == 'invalid_multimodal_generation'
    complete = causal['valid'][0]
    assert len(complete['layers'])*4 + len(complete['random']['repeats'])*2 == 26
    localized = h.loaded('localization')['valid'][0]['partitions']
    assert localized['with_coordinates'][0]['map_index'] == 0
    assert localized['without_coordinates'][0]['map_index'] == 1
    results = h.runtime.load_results(h.path)
    assert results['metadata']['causal_valid_questions'] == 1
    assert results['metadata']['causal_excluded_questions'] == 2
    assert results['summary']['with_coordinates']['reasons'] == {'eligible': 1, 'empty_generation': 1, 'invalid_generation': 1}
    assert all(group['total'] == 1 and group['correct'] == 1 for group in results['summary']['without_coordinates']['groups'])
    bus = next(row for row in results['localization'] if row['sample_id'] == 'valid' and row['partition'] == 'without_coordinates')['targets'][0]
    assert bus['attention_maps'][0][64] == .5 and bus['group_results'][0]['peak_index'] == 64
    audit = json.loads((h.root/'localization_audit.json').read_text())
    assert sum(len(row['targets']) for row in audit) == 4
    assert all('attention_maps' not in entry for row in audit for entry in row['targets'])


@pytest.mark.parametrize('harness', [2], indirect=True)
def test_shards_accept_peer_records_and_resume_without_double_counting(harness):
    h = harness
    for stage in ('baseline', 'causal', 'localization'):
        h.worker.run_worker(h.path, stage, 0)
        h.worker.run_worker(h.path, stage, 1)
        assert set(h.loaded(stage)) == {'valid', 'empty', 'invalid'}
    calls = len(h.generation_calls)
    for stage in ('baseline', 'causal', 'localization'):
        h.worker.run_worker(h.path, stage, 1)
        h.worker.run_worker(h.path, stage, 0)
    assert len(h.generation_calls) == calls == 2
    results = h.runtime.load_results(h.path)
    assert results['metadata']['input_questions'] == 3
    assert results['metadata']['causal_valid_questions'] == 1
    assert results['summary']['with_coordinates']['total_targets'] == 3
