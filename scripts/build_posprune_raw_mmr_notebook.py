"""Bundle the 32- and 64-token MMR notebook and the previous evaluation cohort."""

import json
from pathlib import Path

from build_posprune_notebook import ROOT, cell, source_bundle


RESULTS = ROOT / 'docs/posprune/results'


def evaluation_protocol(results):
    config = json.loads((results / 'config.json').read_bytes())
    ids = config['evaluation_ids']
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('The previous evaluation cohort needs unique question IDs')
    return dict(evaluation_ids=ids, seed=config['seed'],
                evaluation_fraction=config['evaluation_fraction'], generation=config['generation'],
                shallow=config['shallow'], camera_order=config['session']['camera_order'],
                checkpoint_sha256=Path(config['session']['checkpoint']).name, feature_source=config['feature_source'])


def main():
    protocol = json.dumps(evaluation_protocol(RESULTS), ensure_ascii=False, separators=(',', ':'))
    packed = source_bundle({'raw_mmr_protocol.json': protocol})
    cells = [
        cell('markdown', '''
# SmolVLM MMR ablation at 32 and 64 visual tokens

Choose **GPU T4 ×2**, enable Internet, and use **Save & Run All**. The notebook downloads the pinned model, test data, and camera images. It builds temporary pre-connector features with the frozen vision encoder and needs no Kaggle Inputs.

Each method is run at two budgets, keeping 32 or 64 of the 384 visual tokens after layer 3. Both methods start from attention from the final prompt token.

- **Raw top-k** retains the highest raw-attention scores.
- **Raw-attention MMR** applies per-question min–max scaling over all visual tokens for relevance. It measures redundancy with cosine similarity between layer-3 token states. Set `MMR_ATTENTION_NORMALIZATION` to `none` for unscaled relevance and set `MMR_LAMBDA` to tune the trade-off (default 0.9).

All four configurations use the same stratified subset of 4,204 questions from the previous 8,408-question cohort. Each question type contributes exactly half of its previous count. The run uses FP32 greedy decoding and skips positional bias calibration. Quality and latency deltas are computed against raw top-k at the matching budget from this run.
'''),
        cell('code', '''
GPU_COUNT = 2
GENERATION_BATCH = 4  # Per GPU, with automatic CUDA OOM backoff.
KEEP_COUNTS = (32, 64)
EVALUATION_SUBSET_SIZE = 4204  # Set to None to use the full 8,408-question cohort.
MMR_ATTENTION_NORMALIZATION = 'minmax'  # 'minmax' or 'none'; applies only to MMR relevance.
MMR_LAMBDA = 0.9
SELECTION_EXAMPLES = 3
'''),
        cell('code', f'''
import json
import os
from base64 import b64decode
from io import BytesIO
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
if shutil.which('java') is None:
    raise RuntimeError('Java is needed for COCO caption evaluation.')
SOURCE = Path(tempfile.mkdtemp(prefix='log-metrics-raw-mmr-'))
SOURCE_B64 = {packed!r}
with zipfile.ZipFile(BytesIO(b64decode(SOURCE_B64))) as archive:
    archive.extractall(SOURCE)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary=tokenizers',
                '-r', str(SOURCE / 'smolvlm_attention/requirements.txt'), 'pycocoevalcap==1.2'], check=True)
sys.path.insert(0, str(SOURCE))
protocol = json.loads((SOURCE / 'raw_mmr_protocol.json').read_bytes())
''', tags=['hide-input']),
        cell('code', '''
import tokenizers
import torch
import transformers
from huggingface_hub import hf_hub_download
from smolvlm_attention.config import MODELS, PREPARED_REPO, PREPARED_REVISION
from smolvlm_attention.posprune_ablation.image_features import IMAGE_SOURCE
from smolvlm_attention.posprune_ablation.runtime import GENERATION, SHALLOW
from smolvlm_inference.cache import digest

if (transformers.__version__, tokenizers.__version__) != ('4.46.3', '0.20.4'):
    raise RuntimeError('Restart the kernel after dependency installation and Run All.')
worker_count = min(GPU_COUNT, torch.cuda.device_count())
if worker_count < 1:
    raise RuntimeError('Enable a Kaggle GPU accelerator.')
base = MODELS['base']
if (GENERATION != protocol['generation'] or SHALLOW != protocol['shallow']
        or base['camera_order'] != protocol['camera_order']
        or any(protocol['feature_source'].get(key) != value for key, value in IMAGE_SOURCE.items())
        or protocol['feature_source']['mode'] != 'regenerated'):
    raise RuntimeError('The notebook configuration differs from the historical experiment.')

work = Path('/kaggle/working/smolvlm-raw-mmr-32-64')
prepared = work / 'prepared_data'
for name in ('manifest.json', 'test.jsonl'):
    hf_hub_download(repo_id=PREPARED_REPO, repo_type='dataset', revision=PREPARED_REVISION,
                    filename=name, local_dir=prepared, token=False)
checkpoint = Path(hf_hub_download(repo_id=base['repo'], revision=base['revision'],
                                 filename='finetune_best.pt', token=False))
if digest(checkpoint) != protocol['checkpoint_sha256']:
    raise RuntimeError('The checkpoint SHA-256 differs from the historical experiment.')

print(f'Using {worker_count} GPU(s). Budgets {KEEP_COUNTS}, MMR normalization {MMR_ATTENTION_NORMALIZATION}, λ={MMR_LAMBDA}.')
'''),
        cell('code', '''
from smolvlm_attention.posprune_ablation.parallel import run_parallel

run_dir = Path(tempfile.mkdtemp(prefix='run-', dir=work))
output = run_parallel(checkpoint, None, prepared, run_dir / 'results', cache_fingerprint=None,
    camera_order=base['camera_order'], gpu_count=worker_count, evaluation_size=None,
    evaluation_fraction=protocol['evaluation_fraction'], seed=protocol['seed'],
    evaluation_subset_size=EVALUATION_SUBSET_SIZE,
    generation_batch=GENERATION_BATCH, keep_counts=KEEP_COUNTS, selection_examples=SELECTION_EXAMPLES,
    raw_mmr=True, expected_evaluation_ids=protocol['evaluation_ids'],
    attention_normalization=MMR_ATTENTION_NORMALIZATION, mmr_weight=MMR_LAMBDA)
print((output / 'parallel_timing.json').read_text())
'''),
        cell('code', '''
import csv
from IPython.display import FileLink, Image, display
from smolvlm_attention.posprune_ablation.raw_mmr_report import report_raw_mmr

for path in report_raw_mmr(output):
    if path.suffix == '.png':
        display(Image(filename=str(path)))
with (output / 'mmr_vs_raw.csv').open() as stream:
    for row in csv.DictReader(stream):
        print(f"Keep {row['kept_tokens']} · {row['method']}: Overall {float(row['raw_overall']):.3f} → "
              f"{float(row['mmr_overall']):.3f} (Δ {float(row['delta_overall']):+.3f})")
archive = run_dir / 'raw-mmr-results.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != 'generation_ready.json':
            bundle.write(path, path.name)
for path in (output / 'comparison.csv', output / 'mmr_vs_raw.csv', output / 'paired_answers.csv', archive):
    display(FileLink(os.path.relpath(path)))
'''),
    ]
    notebook = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                          'accelerator': 'GPU'}, nbformat=4, nbformat_minor=4)
    target = ROOT / 'notebooks/kaggle_smolvlm_posprune_raw_mmr.ipynb'
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
