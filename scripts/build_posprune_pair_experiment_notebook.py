"""Build a self-contained Kaggle notebook for visual-token pair analysis."""

import json

from build_posprune_notebook import ROOT, cell, source_bundle


def main():
    cells = [
        cell('markdown', '''
# Visual-token pair similarity and single-token ablation

Select **GPU T4 ×2**, enable Internet, and use **Save & Run All**. The notebook downloads the pinned SmolVLM checkpoint and DriveLM test data from Hugging Face. The two workers build temporary visual features from the camera images. No Kaggle Input is needed.

For each stratified test question, compare the hidden-state cosine and incoming-attention cosine of all 73,536 unordered pairs of visual tokens at layers **3, 15, 22, and 26**. A token's attention pattern concatenates probabilities from every head and up to the final 16 prompt tokens after the last image token. Softmax includes all allowed keys. Pearson and Spearman are calculated per question and summarized per layer.

The ablation removes one visual token **after layer 22**, retaining its partner and all other tokens. At layer 22, four pair groups combine the top and bottom quartiles of hidden-state and attention similarity. For each question, a seeded choice within each group uses a common target raw-attention level for the removed token. Questions without all four groups contribute to the correlation results. Ablation summaries use only questions where all four removals were possible.

Each removal has the same question and an unpruned greedy baseline. Outputs include Jensen–Shannon divergence of the first-token distributions, first-token changes, exact decoded-answer changes, and the original indices and cameras of the two tokens. Similarity groups are observational. Many other visual tokens also remain after the removal.
'''),
        cell('code', '''
TEST_QUESTIONS = 800
SEED = 42
GPU_COUNT = 2
BATCH_SIZE = 4  # Per GPU. A worker halves its batch and retries on CUDA OOM.
'''),
        cell('code', f'''
import os
from base64 import b64decode
from io import BytesIO
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
source = Path(tempfile.mkdtemp(prefix='log-metrics-pair-experiment-'))
SOURCE_B64 = {source_bundle()!r}
with zipfile.ZipFile(BytesIO(b64decode(SOURCE_B64))) as archive:
    archive.extractall(source)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary=tokenizers',
                '-r', str(source / 'smolvlm_attention/requirements.txt')], check=True)
sys.path.insert(0, str(source))
''', tags=['hide-input']),
        cell('code', '''
import tokenizers
import torch
import transformers
from huggingface_hub import hf_hub_download
from smolvlm_attention.config import MODELS, PREPARED_REPO, PREPARED_REVISION

if (transformers.__version__, tokenizers.__version__) != ('4.46.3', '0.20.4'):
    raise RuntimeError('Restart the kernel after dependency installation and Run All.')
workers = min(GPU_COUNT, torch.cuda.device_count())
if workers < 1:
    raise RuntimeError('Enable a Kaggle GPU accelerator.')
work = Path('/kaggle/working/smolvlm-token-pair-experiment')
prepared = work / 'prepared_data'
for filename in ('manifest.json', 'test.jsonl'):
    hf_hub_download(repo_id=PREPARED_REPO, repo_type='dataset', revision=PREPARED_REVISION,
                    filename=filename, local_dir=prepared, token=False)
base = MODELS['base']
checkpoint = hf_hub_download(repo_id=base['repo'], revision=base['revision'],
                             filename='finetune_best.pt', token=False)
'''),
        cell('code', '''
from smolvlm_attention.posprune_ablation.pair_experiment import run_pair_experiment

run_dir = Path(tempfile.mkdtemp(prefix='run-', dir=work))
output = run_pair_experiment(checkpoint, prepared, run_dir / 'results', camera_order=base['camera_order'],
                             questions=TEST_QUESTIONS, seed=SEED, gpu_count=workers, batch_size=BATCH_SIZE)
print((output / 'timing.json').read_text())
'''),
        cell('code', '''
from IPython.display import FileLink, Image, display
from smolvlm_attention.posprune_ablation.pair_report import report_pairs

for path in report_pairs(output):
    if path.suffix == '.png':
        display(Image(filename=str(path)))
archive = run_dir / 'token-pair-experiment.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(output.iterdir()):
        if path.is_file():
            bundle.write(path, path.name)
for path in (output / 'correlation_by_layer.csv', output / 'ablation_by_group.csv', archive):
    display(FileLink(os.path.relpath(path)))
'''),
    ]
    notebook = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                          'accelerator': 'GPU'}, nbformat=4, nbformat_minor=4)
    target = ROOT / 'notebooks/kaggle_smolvlm_visual_token_pairs.ipynb'
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
