import ast
import json
from pathlib import Path
import subprocess
import sys


def test_notebook_syntax_and_explicit_input_resume_protocol():
    path = Path('notebooks/kaggle_smolvlm_map_predictor_after2_lastquery_train_test.ipynb')
    notebook = json.loads(path.read_text(encoding='utf-8'))
    sources = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type']=='code']
    for source in sources:
        ast.parse(source)
    text = '\n'.join(sources)
    assert text.index("os.environ['PYTORCH_ALLOC_CONF']") < text.index('import torch')
    assert "TARGETS_INPUT = " in text and "'--targets-input', TARGETS_INPUT" in text
    assert "for kind in ('latest','best')" in text and "filename+'.rng'" in text
    assert "status['training_complete']" in text
    assert "run('diagnose'" in text and "run('train'" in text and "run('evaluate'" in text
    assert 'RESUME_TRAINING' not in text
    assert 'smolvlm_map_predictor_after2_lastquery_index22.json' in text
    assert 'smolvlm_map_predictor_after2_lastquery_index22_outputs' in text
    assert 'map_predictor_after2_artifacts.zip' in text


def test_cli_help_and_required_arguments():
    from smolvlm_map_predictor_after2.cli import parser
    args = parser().parse_args(['train','--run','example/run.json','--resume','latest.pt'])
    assert args.resume == 'latest.pt'
    result = subprocess.run([sys.executable,'-m','smolvlm_map_predictor_after2','--help'],
                            capture_output=True,text=True,timeout=30)
    assert result.returncode == 0
    for command in ('validate-inputs','diagnose','train','verify-resume-checkpoint','evaluate','bundle'):
        assert command in result.stdout
