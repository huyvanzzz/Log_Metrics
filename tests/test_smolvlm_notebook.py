import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_notebook_is_inference_only_and_reuses_loaded_session():
    notebook = json.loads((ROOT / 'notebooks/kaggle_smolvlm_checkpoint_inference.ipynb').read_text())
    code = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
    for source in code:
        ast.parse(source)
    combined = '\n'.join(code)
    assert 'huyvanzzz/smolvlm256M_base' in combined
    assert 'f9d4e9a4d704078f9e7f39f061d13602cf251c78' in combined
    assert 'huyhuy12334/cache-smolvlm' in combined
    assert 'huyvanzzz/prepared_data' in combined
    assert "'smolvlm-attention'" in combined
    assert '.rng' not in combined
    assert 'finetune_latest' not in combined
    assert combined.count('InferenceSession(') == 1
    assert 'session.predict' in code[-1]
    assert 'InferenceSession(' not in code[-1]
    assert 'dataset_download' in combined
    assert 'dataset_upload' not in combined
