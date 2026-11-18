import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dino_notebook_has_independent_pull_and_ten_valid_cells():
    notebook = json.loads((ROOT/'notebooks/kaggle_dinov2_base_attention.ipynb').read_text(encoding='utf-8'))
    cells = [cell for cell in notebook['cells'] if cell['cell_type']=='code']
    assert len(cells)==10
    for cell in cells:
        assert cell['outputs']==[] and cell['execution_count'] is None
        ast.parse(''.join(cell['source']))
    source = [''.join(cell['source']) for cell in cells]
    helper = (ROOT/'scripts/kaggle_source_setup.py').read_text(encoding='utf-8').strip()
    assert helper in source[0]
    assert 'https://github.com/huyvanzzz/Log_Metrics.git' in source[0]
    assert 'smolvlm-attention' in source[0]
    assert source[0].index("os.environ['USE_TF']") < source[0].index('import torch')
    assert "packages=('dinov2_attention', 'smolvlm_attention')" in source[0]
    assert 'dinov2_attention/requirements.txt' in source[0]
    assert 'cuda.is_available' not in source[0]
    setup_tree = ast.parse(source[0])
    mode_guard = next(node for node in setup_tree.body if isinstance(node, ast.If)
                      and ast.unparse(node.test)=="BOOT_MODE != 'plot'")
    guarded_source = ast.unparse(mode_guard)
    assert "'pip', 'install'" in guarded_source
    assert 'import torch, transformers, tokenizers' in guarded_source
    outside = ast.unparse(ast.Module(body=[node for node in setup_tree.body if node is not mode_guard],type_ignores=[]))
    assert 'import torch' not in outside
    assert "'pip', 'install'" not in outside
    assert 'resolve_run_mode' in source[1]
    for i in (2,3,4):
        assert "MODE != 'plot'" in source[i]
    assert 'load_results' in source[5]
    assert 'plot_token_atlas' in source[6]
    assert 'plot_spatial_atlas' in source[7]
    assert 'plot_layer_histogram' in source[8]
    assert 'make_figures_zip' in source[9]
    assert all('resolve_artifacts' not in cell for cell in source[5:])


def test_builder_generates_exact_checked_in_notebook():
    import importlib.util
    path = ROOT/'scripts/build_dinov2_attention_notebook.py'
    spec = importlib.util.spec_from_file_location('build_dino_notebook',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    notebook = json.loads((ROOT/'notebooks/kaggle_dinov2_base_attention.ipynb').read_text(encoding='utf-8'))
    assert module.make_notebook()==notebook
