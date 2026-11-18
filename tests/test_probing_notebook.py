import ast
import importlib.util
import json
from pathlib import Path


def test_notebook_builder_exact_cells_and_plot_paths():
    builder = Path('scripts/build_probing_notebook.py')
    spec = importlib.util.spec_from_file_location('probing_builder', builder)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    notebook = json.loads(Path('notebooks/kaggle_smolvlm_causal_localization.ipynb').read_text(encoding='utf-8'))
    assert module.make_notebook() == notebook
    code = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
    assert len(code) == 11
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            assert cell['execution_count'] is None and cell['outputs'] == []
            ast.parse(''.join(cell['source']))
    assert 'git' in code[0] and 'smolvlm_probing' in code[0]
    assert Path('scripts/kaggle_source_setup.py').read_text(encoding='utf-8').strip() in code[0]
    assert 'launch_stage' in code[4] and "'baseline'" in code[4]
    assert "'causal'" in code[5] and "'localization'" in code[6]
    assert 'load_results' in code[7]
    assert 'fetch_example_images' in code[9] and 'bundle_figures' in code[10]
    assert all('resolve_artifacts' not in c and 'launch_stage' not in c for c in code[7:])
