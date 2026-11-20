import numpy as np
from test_attention_collector import fixture


def test_tiny_preflight_native_pruned_and_batch(monkeypatch):
    from smolvlm_topk_benchmark.preflight import numerical_preflight
    model, _, items, collator = fixture(questions=('What?', 'What is visible? A road. A car.'))
    rows = [dict(item, row_index=index, input_ids=collator([item])['input_ids'][0].tolist()) for index, item in enumerate(items)]
    snapshots = {name: parameter.clone() for name, parameter in model.named_parameters()}
    def no_vision(*args, **kwargs):
        raise AssertionError('cached feature preflight must not run vision')
    monkeypatch.setattr(model.native.model.vision_model, 'forward', no_vision)
    batch_sizes = []
    def build_batch(selected):
        batch_sizes.append(len(selected))
        return collator(selected)
    result = numerical_preflight(model, rows, build_batch,
        {'base': None, 'selected': np.array([[0, 5, 10, 23], [1, 6, 11, 22]])})
    assert result['passed']
    assert result['probe_row_indices'] == [0, 1]
    assert result['pruned_config'] == 'selected'
    assert batch_sizes == [1, 1, 2]
    import torch
    assert all(torch.equal(parameter, snapshots[name]) for name, parameter in model.named_parameters())
