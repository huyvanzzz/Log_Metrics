"""Build a Kaggle notebook that can run the current source without a Git push."""

from base64 import b64encode
from io import BytesIO
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def source_bundle():
    files = list((ROOT / 'smolvlm_inference').glob('*.py'))
    attention = ROOT / 'smolvlm_attention'
    files += [attention / name for name in ('__init__.py', 'collector.py', 'config.py', 'plotting.py')]
    files += list((attention / 'posprune_ablation').glob('*.py'))
    files += [ROOT / 'smolvlm_inference/requirements.txt', attention / 'requirements.txt']
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), (2026, 1, 1, 0, 0, 0))
            archive.writestr(info, path.read_bytes().replace(b'\r\n', b'\n'), compress_type=zipfile.ZIP_DEFLATED)
    return b64encode(buffer.getvalue()).decode('ascii')


def cell(kind, content, tags=None):
    result = dict(cell_type=kind, metadata={} if tags is None else {'tags': tags},
                  source=content.strip().splitlines(keepends=True))
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def main():
    cells = [
        cell('markdown', '''
# SmolVLM decoder-only PosPrune ablation

Enable a Kaggle GPU and Internet. Attach `huyhuy12334/cache-smolvlm` version 1 as an Input if available. The notebook uses the pinned base checkpoint and prepared data and reads the mounted cache directly. The first cache validation scans the full payload.

The cubic is fitted once on validation attention differences from layers 3 and 22. Generation compares the unpruned checkpoint with raw top-k, corrected top-k, and corrected + MMR at each budget, using the existing FP32 and three-beam settings. This notebook includes its Python source snapshot and can run independently of a Git branch. Regenerate it with `python scripts/build_posprune_notebook.py` after editing the source.

The first-beam-step timing ends when beam search selects its first candidates. Generation timings start after the inputs are on the GPU and include projection, pruning, and decoding in the model.
'''),
        cell('code', '''
CALIBRATION_SIZE = 800
EVALUATION_SIZE = 100  # Set to None for the full test split.
CALIBRATION_BATCH = 4  # Answer generation always uses batch 1.
KEEP_COUNTS = (192, 96, 48)
SEED = 42
SELECTION_EXAMPLES = 3  # Save full ranked selections for the first evaluation examples.
CACHE_INPUT_DIR = ''   # Optional directory within /kaggle/input containing the cache files.
'''),
        cell('code', f'''
# Embedded source snapshot. Run this cell before importing Transformers.
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
from pathlib import Path
from base64 import b64decode
from io import BytesIO
import shutil
import subprocess
import sys
import tempfile
import zipfile

if shutil.which('java') is None:
    raise RuntimeError('Java is required by the existing COCO caption evaluation metrics.')
SOURCE = Path(tempfile.mkdtemp(prefix='log-metrics-posprune-'))
SOURCE_B64 = {source_bundle()!r}
with zipfile.ZipFile(BytesIO(b64decode(SOURCE_B64))) as archive:
    archive.extractall(SOURCE)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary=tokenizers',
                '-r', str(SOURCE / 'smolvlm_attention/requirements.txt'), 'pycocoevalcap==1.2'], check=True)
sys.path.insert(0, str(SOURCE))
''', tags=['hide-input']),
        cell('code', '''
import torch
import transformers
import tokenizers
from huggingface_hub import hf_hub_download
import kagglehub
from smolvlm_attention.config import MODELS, CACHE_DATASET, CACHE_FINGERPRINT, PREPARED_REPO, PREPARED_REVISION
from smolvlm_inference.session import InferenceSession

if (transformers.__version__, tokenizers.__version__) != ('4.46.3', '0.20.4'):
    raise RuntimeError('Restart the kernel after dependency installation and Run All.')
if not torch.cuda.is_available():
    raise RuntimeError('Enable a Kaggle GPU accelerator.')

work = Path('/kaggle/working/smolvlm-posprune')
prepared_dir = work / 'prepared_data'
for name in ('manifest.json', 'train.jsonl', 'val.jsonl', 'test.jsonl'):
    hf_hub_download(repo_id=PREPARED_REPO, repo_type='dataset', revision=PREPARED_REVISION,
                    filename=name, local_dir=prepared_dir, token=False)
base = MODELS['base']
checkpoint = hf_hub_download(repo_id=base['repo'], revision=base['revision'],
                             filename='finetune_best.pt', token=False)
cache_root = Path(CACHE_INPUT_DIR) if CACHE_INPUT_DIR else Path(kagglehub.dataset_download(CACHE_DATASET))
caches = [path.parent for path in cache_root.rglob('features.float16.memmap')
          if (path.parent / 'manifest.json').is_file() and (path.parent / 'index.json').is_file()]
if len(caches) != 1:
    raise RuntimeError(f'Expected exactly one complete cache under {cache_root}, found {len(caches)}.')
session = InferenceSession(checkpoint, caches[0], prepared_dir, device='cuda:0')
if session.cache.manifest['fingerprint'] != CACHE_FINGERPRINT or session.model.prompt_camera_order != base['camera_order']:
    raise RuntimeError('The cache identity or checkpoint camera order differs from the pinned baseline.')
print(session.describe())
'''),
        cell('code', '''
from smolvlm_attention.posprune_ablation.runtime import run

run_dir = Path(tempfile.mkdtemp(prefix='run-', dir=work))
output = run(session, run_dir / 'results', calibration_size=CALIBRATION_SIZE,
             evaluation_size=EVALUATION_SIZE, calibration_batch=CALIBRATION_BATCH,
             keep_counts=KEEP_COUNTS, seed=SEED, selection_examples=SELECTION_EXAMPLES)
print(output)
'''),
        cell('code', '''
from IPython.display import FileLink, Image, display
from smolvlm_attention.posprune_ablation.report import report

for path in report(output):
    if path.suffix == '.png':
        display(Image(filename=str(path)))
archive = Path(shutil.make_archive(str(run_dir / 'posprune-results'), 'zip', root_dir=output))
tables = [output / name for name in ('indexed_selections.csv', 'comparison.csv', 'paired_answers.csv')]
for path in (*tables, archive):
    display(FileLink(os.path.relpath(path)))
'''),
    ]
    notebook = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                          'accelerator': 'GPU'}, nbformat=4, nbformat_minor=4)
    path = ROOT / 'notebooks/kaggle_smolvlm_posprune_ablation.ipynb'
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
