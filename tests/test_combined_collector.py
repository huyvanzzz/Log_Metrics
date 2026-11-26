import pytest
import torch
from test_attention_collector import fixture


def test_native_parity_chunk_and_batch():
    from smolvlm_attention.combined_collector import collect_combined_visual_attention, reference_combined_visual_attention
    model, batch, items, collator = fixture(questions=('What?', 'What is visible? A road. A car.'))
    actual = collect_combined_visual_attention(model, batch, (1, 0), 8)
    expected = reference_combined_visual_attention(model, batch, (1, 0))
    assert actual.shape == (2, 2, 24) and actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(actual, collect_combined_visual_attention(model, batch, (1, 0), 1))
    for i, item in enumerate(items):
        torch.testing.assert_close(actual[i:i+1], collect_combined_visual_attention(model, collator([item]), (1, 0)), atol=1e-5, rtol=1e-4)
    assert all(not l.self_attn._forward_hooks and not l.self_attn._forward_pre_hooks for l in model.native.model.text_model.layers)


def test_mean_before_product_and_query_exclusion():
    from smolvlm_attention.combined_collector import combine_attention
    a = torch.tensor([[[[1.,0,0],[.8,.2,0],[.2,.4,.4]], [[1.,0,0],[.2,.8,0],[.4,.2,.4]]]])
    value = combine_attention(a, torch.tensor([[1,1,1]]), torch.tensor([2]), torch.tensor([[0]]))
    torch.testing.assert_close(value, torch.tensor([[.45]]))


@pytest.mark.parametrize('layers,chunk', [((0,0),8), ((2,),8), ((),8), ((0,),0)])
def test_malformed(layers,chunk):
    from smolvlm_attention.combined_collector import collect_combined_visual_attention
    model,batch,_,_=fixture()
    with pytest.raises(ValueError): collect_combined_visual_attention(model,batch,layers,chunk)


def test_no_intermediates_is_direct_and_padding_is_ignored():
    from smolvlm_attention.combined_collector import combine_attention
    a = torch.tensor([[[[1.,0,0],[.6,.4,0],[.2,.3,.5]]]])
    value = combine_attention(a, torch.tensor([[1,1,0]]), torch.tensor([1]), torch.tensor([[0]]))
    torch.testing.assert_close(value, torch.tensor([[.6]]))


def test_right_padding_and_cleanup_on_failure(monkeypatch):
    from smolvlm_attention.combined_collector import collect_combined_visual_attention, reference_combined_visual_attention
    model,batch,_,_=fixture()
    expected=collect_combined_visual_attention(model,batch,(0,1))
    batch['input_ids']=torch.nn.functional.pad(batch['input_ids'],(0,3),value=model.processor.tokenizer.pad_token_id)
    batch['attention_mask']=torch.nn.functional.pad(batch['attention_mask'],(0,3),value=0)
    torch.testing.assert_close(expected,collect_combined_visual_attention(model,batch,(0,1)))
    def forbidden(*a,**kw): raise AssertionError('forbidden')
    monkeypatch.setattr(model.native.model.vision_model,'forward',forbidden)
    monkeypatch.setattr(model.native.lm_head,'forward',forbidden)
    collect_combined_visual_attention(model,batch,(0,1))
    monkeypatch.setattr(model.native.model.text_model.layers[1].self_attn,'forward',forbidden)
    for fn in (collect_combined_visual_attention,reference_combined_visual_attention):
        with pytest.raises(AssertionError): fn(model,batch,(0,1))
        assert all(not l.self_attn._forward_hooks and not l.self_attn._forward_pre_hooks for l in model.native.model.text_model.layers)


def test_preflight():
    from smolvlm_attention.combined_collector import numerical_preflight
    model,_,items,collator=fixture(questions=('What?', 'What is visible? A road. A car.'))
    rows=[dict(item,input_ids=collator([item])['input_ids'][0].tolist()) for item in items]
    result=numerical_preflight(model,rows,collator,layer_ids=(0,1))
    assert result['status']=='passed' and result['samples']==2


def test_selected_single_layer_oracle_and_no_renormalization():
    from smolvlm_attention.combined_collector import collect_combined_visual_attention, reference_combined_visual_attention
    model,batch,_,_=fixture()
    actual=collect_combined_visual_attention(model,batch,(1,))
    torch.testing.assert_close(actual,reference_combined_visual_attention(model,batch,(1,)),atol=1e-6,rtol=1e-4)
    assert actual.shape==(1,1,24) and (actual.sum(-1)<1).all()
