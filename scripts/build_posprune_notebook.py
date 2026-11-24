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
    files += [attention / name for name in ('__init__.py', 'collector.py', 'config.py', 'data.py', 'plotting.py')]
    files += list((attention / 'posprune_ablation').glob('*.py'))
    files.append(attention / 'posprune_ablation/test_question_types.json')
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

Select **GPU T4 ×2** in Kaggle, enable Internet, and use **Save & Run All**. The notebook downloads the pinned checkpoint, prepared data, and the images it needs from Hugging Face without adding any Kaggle Inputs. The GPUs run the frozen vision encoder once per selected six-camera frame and store temporary FP16 pre-connector features shared by all variants.

Evaluation uses a seeded, stratified 50% test sample with the original question-type proportions. The notebook prints the full and selected counts for perception, prediction, planning, and behavior. Calibration still uses 800 separate validation questions.

The cubic is fitted once on validation attention differences from layers 3 and 22. The unpruned baseline and all pruning variants use FP32 greedy decoding, with up to 512 new tokens. Each GPU starts with four questions per generation batch. If a CUDA out-of-memory error occurs, that worker halves its generation batch and retries the same questions for all variants.

The notebook includes its Python source snapshot. Regenerate it with `python scripts/build_posprune_notebook.py` after editing the source.

First-token and generation latency are measured once per batch, after inputs reach the GPU. The comparison table also reports model throughput in questions per GPU-second.
`parallel_timing.json` records elapsed time for feature preparation, prefill, and paired generation across both GPUs. For comparisons against earlier runs, `config.json` distinguishes freshly extracted features from a mounted cache.
'''),
        cell('code', '''
GPU_COUNT = 2
CALIBRATION_SIZE = 800
EVALUATION_FRACTION = 0.50  # Stratified by question type. Set to 1.0 for the full test split.
CALIBRATION_BATCH = 4  # Per GPU.
GENERATION_BATCH = 4   # Per GPU, with automatic CUDA OOM backoff.
KEEP_COUNTS = (192, 96, 48)
SEED = 42
SELECTION_EXAMPLES = 3  # Save full ranked selections for the first evaluation examples.
CACHE_INPUT_DIR = ''   # Leave blank to fetch images and build features automatically.
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
from smolvlm_attention.config import MODELS, CACHE_DATASET, CACHE_FINGERPRINT, PREPARED_REPO, PREPARED_REVISION
from smolvlm_attention.posprune_ablation.cache_input import resolve_cache_input

if (transformers.__version__, tokenizers.__version__) != ('4.46.3', '0.20.4'):
    raise RuntimeError('Restart the kernel after dependency installation and Run All.')
visible_gpus = torch.cuda.device_count()
if visible_gpus == 0:
    raise RuntimeError('Enable a Kaggle GPU accelerator.')
if GPU_COUNT < 1:
    raise ValueError('GPU_COUNT must be at least 1.')
worker_count = min(GPU_COUNT, visible_gpus)
print(f'Using {worker_count} of {visible_gpus} visible GPU(s).')

cache_dir = None
if CACHE_INPUT_DIR:
    cache_dir = resolve_cache_input('/kaggle/input', CACHE_INPUT_DIR, dataset=CACHE_DATASET,
                                   fingerprint=CACHE_FINGERPRINT)
    print(f'Using mounted feature cache: {cache_dir}')
work = Path('/kaggle/working/smolvlm-posprune')
prepared_dir = work / 'prepared_data'
for name in ('manifest.json', 'train.jsonl', 'val.jsonl', 'test.jsonl'):
    hf_hub_download(repo_id=PREPARED_REPO, repo_type='dataset', revision=PREPARED_REVISION,
                    filename=name, local_dir=prepared_dir, token=False)
base = MODELS['base']
checkpoint = hf_hub_download(repo_id=base['repo'], revision=base['revision'],
                             filename='finetune_best.pt', token=False)
'''),
        cell('code', '''
from smolvlm_attention.posprune_ablation.parallel import run_parallel

run_dir = Path(tempfile.mkdtemp(prefix='run-', dir=work))
output = run_parallel(checkpoint, cache_dir, prepared_dir, run_dir / 'results',
                      cache_fingerprint=CACHE_FINGERPRINT, camera_order=base['camera_order'], gpu_count=worker_count,
                      calibration_size=CALIBRATION_SIZE, evaluation_size=None, evaluation_fraction=EVALUATION_FRACTION,
                      calibration_batch=CALIBRATION_BATCH, generation_batch=GENERATION_BATCH,
                      keep_counts=KEEP_COUNTS, seed=SEED,
                      selection_examples=SELECTION_EXAMPLES)
print((output / 'parallel_timing.json').read_text())
print(output)
'''),
        cell('code', '''
from IPython.display import FileLink, Image, display
from smolvlm_attention.posprune_ablation.report import report

for path in report(output):
    if path.suffix == '.png':
        display(Image(filename=str(path)))
archive = run_dir / 'posprune-results.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(output.iterdir()):
        if path.is_file():
            bundle.write(path, path.name)
tables = [output / name for name in ('indexed_selections.csv', 'comparison.csv', 'paired_answers.csv',
                                      'question_type_distribution.csv')]
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
