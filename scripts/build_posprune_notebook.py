"""Build a Kaggle notebook that can run the current source without a Git push."""

from base64 import b64encode
import argparse
from io import BytesIO
import json
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PREDICTOR_SOURCE = 'b63c7c8fac9ee39d388b5618ba853f561b4fba0f'


def source_bundle(extra_files=None):
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
        for name, payload in (extra_files or {}).items():
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED)
    return b64encode(buffer.getvalue()).decode('ascii')


def cell(kind, content, tags=None):
    result = dict(cell_type=kind, metadata={} if tags is None else {'tags': tags},
                  source=content.strip().splitlines(keepends=True))
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def main(*, camera_bias=False, camera_spline=False, predictor_comparison=False):
    camera_bias = camera_bias or camera_spline or predictor_comparison
    overview = '''
# SmolVLM decoder-only PosPrune ablation

Select **GPU T4 ×2** in Kaggle, enable Internet, and use **Save & Run All**. The notebook downloads the pinned checkpoint, prepared data, and the images it needs from Hugging Face without adding any Kaggle Inputs. The GPUs run the frozen vision encoder once per selected six-camera frame and store temporary FP16 pre-connector features shared by all variants.

Evaluation uses a seeded, stratified 50% test sample with the original question-type proportions. The notebook prints the full and selected counts for perception, prediction, planning, and behavior. Calibration still uses 800 separate validation questions.

The cubic is fitted once on validation attention differences from layers 3 and 22. The unpruned baseline and all pruning variants use FP32 greedy decoding, with up to 512 new tokens. Each GPU starts with four questions per generation batch. If a CUDA out-of-memory error occurs, that worker halves its generation batch and retries the same questions for all variants.

The notebook includes its Python source snapshot. Regenerate it with `python scripts/build_posprune_notebook.py` after editing the source.

First-token and generation latency are measured once per batch, after inputs reach the GPU. The comparison table also reports model throughput in questions per GPU-second.
`parallel_timing.json` records elapsed time for feature preparation, prefill, and paired generation across both GPUs. For comparisons against earlier runs, `config.json` distinguishes freshly extracted features from a mounted cache.
'''
    settings = '''
GPU_COUNT = 2
CALIBRATION_SIZE = 800
EVALUATION_FRACTION = 0.50  # Stratified by question type. Set to 1.0 for the full test split.
CALIBRATION_BATCH = 4  # Per GPU.
GENERATION_BATCH = 4   # Per GPU, with automatic CUDA OOM backoff.
KEEP_COUNTS = (192, 96, 48)
SEED = 42
SELECTION_EXAMPLES = 3  # Save full ranked selections for the first evaluation examples.
CACHE_INPUT_DIR = ''   # Leave blank to fetch images and build features automatically.
'''
    extra_files = None
    notebook_name = 'ablation'
    run_folder = 'smolvlm-posprune'
    run_options = 'selection_examples=SELECTION_EXAMPLES'
    if camera_bias:
        overview = '''
# SmolVLM PosPrune with per-camera bias fits

Select **GPU T4 ×2**, enable Internet, and use **Save & Run All**. The notebook downloads the pinned checkpoint and data, then extracts temporary vision features. No Kaggle Input is required.

For each of the six cameras, fit a cubic on its 64 positions using the mean attention difference from block 2 to block 23 (`layers[1]` minus `layers[22]`). Fit on 800 validation questions. Subtract the fitted value at each visual position from block-2 attention. Evaluate residuals separately on the same 4,204 stratified test questions as the recent MMR notebooks. The report compares held-out RMSE with a single global cubic fitted at the same layers for reference.

All four methods prune immediately after block 2, keeping 32 or 64 visual tokens across the six cameras. Raw and corrected top-k use their respective attention scores. Raw + MMR and corrected + MMR use the corresponding scores after per-question min–max normalization over all 384 tokens, with block-2 output hidden-state cosine for redundancy and λ = 0.9. The eight configurations use FP32 greedy decoding. Deltas use raw attention at the same budget as the reference.
'''
        settings = settings.replace('KEEP_COUNTS = (192, 96, 48)\nSEED = 42', '''KEEP_COUNTS = (32, 64)
CAMERA_BIAS_LAYERS = (2, 23)  # One-based decoder block numbers.
MMR_LAMBDA = 0.9
EVALUATION_SUBSET_SIZE = 4204''')
        settings = '\n'.join(line for line in settings.splitlines() if not line.startswith('EVALUATION_FRACTION'))
        config = json.loads((ROOT / 'docs/posprune/results/config.json').read_bytes())
        cohort = {key: config[key] for key in ('evaluation_ids', 'evaluation_fraction', 'seed')}
        extra_files = {'camera_posprune_cohort.json': json.dumps(cohort, separators=(',', ':'))}
        notebook_name = 'camera_bias_2_23'
        run_folder = 'smolvlm-posprune-camera-bias-2-23'
        run_options = '''selection_examples=SELECTION_EXAMPLES, camera_bias_layers=CAMERA_BIAS_LAYERS,
                      mmr_weight=MMR_LAMBDA, attention_normalization='minmax',
                      evaluation_subset_size=EVALUATION_SUBSET_SIZE, expected_evaluation_ids=cohort['evaluation_ids']'''
    if camera_spline:
        overview = overview.replace('per-camera bias fits', 'per-camera smoothing splines').replace(
            'fit a cubic on its 64 positions', 'fit a natural cubic smoothing spline on its 64 positions').replace(
            'The report compares held-out RMSE with a single global cubic fitted at the same layers for reference.',
            'The smoothing parameters were selected with recording-wise cross-validation on these 800 validation '
            'questions in the local CUDA run. The report compares held-out RMSE with per-camera and global cubic fits.')
        settings += '''
# Fixed by recording-wise validation on the unit-spaced camera grid.
SPLINE_SMOOTHING = {
    'Front': 0.01,
    'Front-Left': 1000.0,
    'Front-Right': 100.0,
    'Back': 0.01,
    'Back-Left': 10000.0,
    'Back-Right': 3.0,
}
'''
        notebook_name = 'camera_spline_2_23'
        run_folder = 'smolvlm-posprune-camera-spline-2-23'
        run_options += ',\n                      spline_smoothing=SPLINE_SMOOTHING'
    predictor_setup = ''
    if predictor_comparison:
        overview = '''
# SmolVLM spline + MMR versus MLP + MMR at block 3

Set `PREDICTOR_CHECKPOINT` to your trained after2 `predictor_best.pt`, or set a Hugging Face repository and revision with that file. Select **GPU T4 ×2**, enable Internet, and use **Save & Run All**. A Hugging Face predictor can be run without attaching a Kaggle Input.

Both variants keep 32 of 384 visual tokens immediately after decoder block 3 (`layers[2]`). Both run on the same 4,204 stratified DriveLM test questions with the same six-camera features, backbone, batch, FP32 greedy decoding, token budget, MMR λ = 0.9, min–max relevance normalization, and cosine redundancy from block-3 outputs.

- **Spline + MMR** subtracts a separate smoothing spline per camera from the attention at block 3. The fitted difference is block 3 minus block 23. Smoothing is selected on two disjoint recording groups from the 800 validation questions, then fit on all 800.
- **MLP + MMR** uses the frozen after2 MAP predictor from the original branch, with the last prompt query and the visual hidden states after block 3. It uses the existing trained weights and interprets its log probabilities as relevance probabilities before MMR.

The report includes paired answers, selected token indices, quality and latency under the same timing procedure, and the spline plots. Only the source of MMR relevance changes during evaluation.
'''
        settings = settings.replace('KEEP_COUNTS = (32, 64)', 'KEEP_COUNTS = (32,)').replace(
            'CAMERA_BIAS_LAYERS = (2, 23)', 'CAMERA_BIAS_LAYERS = (3, 23)') + '''
PREDICTOR_CHECKPOINT = ''  # Exact local path to the trained after2 predictor_best.pt.
PREDICTOR_HF_REPO = ''  # Alternative Hugging Face location for the existing predictor.
PREDICTOR_HF_REVISION = ''  # Required when using the Hugging Face option.
PREDICTOR_HF_FILENAME = 'predictor_best.pt'
PREDICTOR_HF_REPO_TYPE = 'model'
'''
        for package in ('smolvlm_map_predictor', 'smolvlm_map_predictor_after2'):
            for name in ('__init__.py', 'model.py'):
                file = f'{package}/{name}'
                extra_files[file] = subprocess.check_output(
                    ['git', '-c', f'safe.directory={ROOT.as_posix()}', 'show', f'{PREDICTOR_SOURCE}:{file}'], cwd=ROOT)
        notebook_name = 'spline_vs_mlp_after3_32'
        run_folder = 'smolvlm-posprune-spline-vs-mlp-after3-32'
        run_options += ',\n                      predictor_checkpoint=predictor_checkpoint'
        predictor_setup = '''
if PREDICTOR_CHECKPOINT and PREDICTOR_HF_REPO:
    raise ValueError('Choose a local predictor checkpoint or a Hugging Face repository.')
if PREDICTOR_CHECKPOINT:
    predictor_checkpoint = Path(PREDICTOR_CHECKPOINT).resolve(strict=True)
elif PREDICTOR_HF_REPO and PREDICTOR_HF_REVISION:
    predictor_checkpoint = Path(hf_hub_download(repo_id=PREDICTOR_HF_REPO, revision=PREDICTOR_HF_REVISION,
        filename=PREDICTOR_HF_FILENAME, repo_type=PREDICTOR_HF_REPO_TYPE))
else:
    raise ValueError('Set the path to your trained after2 predictor_best.pt, or its Hugging Face repository and revision.')
'''
    cells = [
        cell('markdown', overview),
        cell('code', settings),
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
SOURCE_B64 = {source_bundle(extra_files)!r}
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

PREDICTOR_SETUP
cache_dir = None
if CACHE_INPUT_DIR:
    cache_dir = resolve_cache_input('/kaggle/input', CACHE_INPUT_DIR, dataset=CACHE_DATASET,
                                   fingerprint=CACHE_FINGERPRINT)
    print(f'Using mounted feature cache: {cache_dir}')
work = Path('/kaggle/working/RUN_FOLDER')
prepared_dir = work / 'prepared_data'
for name in ('manifest.json', 'train.jsonl', 'val.jsonl', 'test.jsonl'):
    hf_hub_download(repo_id=PREPARED_REPO, repo_type='dataset', revision=PREPARED_REVISION,
                    filename=name, local_dir=prepared_dir, token=False)
base = MODELS['base']
checkpoint = hf_hub_download(repo_id=base['repo'], revision=base['revision'],
                             filename='finetune_best.pt', token=False)
'''.replace('RUN_FOLDER', run_folder).replace('PREDICTOR_SETUP', predictor_setup)),
        cell('code', ('''
import json
cohort = json.loads((SOURCE / 'camera_posprune_cohort.json').read_bytes())
''' if camera_bias else '') + '''
from smolvlm_attention.posprune_ablation.parallel import run_parallel

run_dir = Path(tempfile.mkdtemp(prefix='run-', dir=work))
output = run_parallel(checkpoint, cache_dir, prepared_dir, run_dir / 'results',
                      cache_fingerprint=CACHE_FINGERPRINT, camera_order=base['camera_order'], gpu_count=worker_count,
                      calibration_size=CALIBRATION_SIZE, evaluation_size=None, evaluation_fraction=EVALUATION_FRACTION,
                      calibration_batch=CALIBRATION_BATCH, generation_batch=GENERATION_BATCH,
                      keep_counts=KEEP_COUNTS, seed=SEED,
                      RUN_OPTIONS)
print((output / 'parallel_timing.json').read_text())
print(output)
'''.replace('RUN_OPTIONS', run_options).replace('evaluation_fraction=EVALUATION_FRACTION,',
    "evaluation_fraction=cohort['evaluation_fraction']," if camera_bias else 'evaluation_fraction=EVALUATION_FRACTION,').replace(
    'seed=SEED,', "seed=cohort['seed']," if camera_bias else 'seed=SEED,')),
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
''' + ('''\ndisplay(FileLink(os.path.relpath(output / 'camera_fit_metrics.csv')))\n''' if camera_bias else '')),
    ]
    notebook = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                          'accelerator': 'GPU'}, nbformat=4, nbformat_minor=4)
    path = ROOT / f'notebooks/kaggle_smolvlm_posprune_{notebook_name}.ipynb'
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--camera-bias', action='store_true', help='Build the per-camera cubic experiment')
    mode.add_argument('--camera-spline', action='store_true', help='Build the per-camera smoothing spline experiment')
    mode.add_argument('--spline-vs-mlp', action='store_true', help='Build the paired block-3 spline and MAP predictor experiment')
    args = parser.parse_args()
    main(camera_bias=args.camera_bias, camera_spline=args.camera_spline, predictor_comparison=args.spline_vs_mlp)
