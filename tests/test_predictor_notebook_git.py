"""Exercise the actual Git bootstrap embedded in both Kaggle notebooks."""
import json
from pathlib import Path
import subprocess
import pytest

NOTEBOOKS=['kaggle_smolvlm_map_predictor_train_test.ipynb','kaggle_smolvlm_map_predictor_after2_lastquery_train_test.ipynb']

def git(*args):
    return subprocess.check_output(['git',*map(str,args)],stderr=subprocess.STDOUT,text=True).strip()

def bootstrap(path,source):
    notebook=json.loads(path.read_text(encoding='utf-8'))
    cell=''.join(next(c['source'] for c in notebook['cells'] if c['cell_type']=='code'))
    code=cell.split('assert (SOURCE/')[0]
    return code.replace("SOURCE = Path('/kaggle/working/Log_Metrics')",f'SOURCE = Path({str(source)!r})')

@pytest.mark.parametrize('name',NOTEBOOKS)
def test_existing_clone_pulls_new_commit_and_refuses_local_edits(tmp_path,name):
    remote=tmp_path/'remote.git';author=tmp_path/'author';clone=tmp_path/'clone'
    git('init','--bare',remote);git('init','-b','smolvlm-attention',author)
    git('-C',author,'config','user.email','fixture@example.invalid');git('-C',author,'config','user.name','Fixture')
    (author/'code.txt').write_text('old')
    git('-C',author,'add','.');git('-C',author,'commit','-m','initial')
    git('-C',author,'remote','add','origin',remote);git('-C',author,'push','-u','origin','smolvlm-attention')
    git('clone','--branch','smolvlm-attention',remote,clone)
    (author/'code.txt').write_text('new');git('-C',author,'commit','-am','update');git('-C',author,'push')
    code=bootstrap(Path('notebooks')/name,clone)
    exec(compile(code,name,'exec'),{})
    assert (clone/'code.txt').read_text()=='new'
    assert git('-C',clone,'rev-parse','HEAD')==git('-C',author,'rev-parse','HEAD')
    (clone/'code.txt').write_text('my local edit')
    with pytest.raises(RuntimeError,match='local'):
        exec(compile(code,name,'exec'),{})
    assert (clone/'code.txt').read_text()=='my local edit'
    git('-C',clone,'restore','code.txt');git('-C',clone,'switch','-c','different')
    with pytest.raises((AssertionError,RuntimeError),match='branch'):
        exec(compile(code,name,'exec'),{})
