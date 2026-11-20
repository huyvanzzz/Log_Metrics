import numpy as np
import pytest
from smolvlm_topk_benchmark.data import topk_indices, aligned_rows


def test_topk_stable_ties_and_original_order():
    values=np.zeros((2,384),np.float32)
    values[0,380:]=.1
    result=topk_indices(values)
    assert result.shape==(2,32) and result.dtype==np.uint16
    np.testing.assert_array_equal(result[1],np.arange(32))
    np.testing.assert_array_equal(result[0],list(range(28))+list(range(380,384)))
    assert all(np.diff(row).min()>0 for row in result)
    values[0,0]=np.nan
    with pytest.raises(ValueError):topk_indices(values)


def test_alignment_uses_ids_not_position():
    expected=[dict(row_index=i,sample_id=f's{i}',eval_id=i, question='Q',answer='A',camera_paths={},
                   input_ids=[9,3],image_token_id=9,feature_key='x',feature_index=0,
                   prompt_length=2,query_position=1,query_token_id=3) for i in range(2)]
    exported=[dict(expected[1],row_index=0),dict(expected[0],row_index=1)]
    merged=aligned_rows(expected,exported)
    assert [r['attention_row'] for r in merged]==[1,0]
    exported[0]['input_ids']=[9,4]
    with pytest.raises(ValueError,match='input_ids'):aligned_rows(expected,exported)
