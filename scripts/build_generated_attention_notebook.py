"""Build standalone generation notebook and refresh source bundled in the legacy notebook."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for package in ('smolvlm_inference', 'smolvlm_attention'):
            for path in sorted((ROOT / package).iterdir()):
                if path.suffix in ('.py', '.txt'):
                    entry = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), (2026, 1, 1, 0, 0, 0))
                    entry.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(entry, path.read_bytes())
    payload = buffer.getvalue()
    encoded, sha = base64.b64encode(payload).decode(), hashlib.sha256(payload).hexdigest()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    cells = []
    def cell(kind, source):
        first, *rest = source.splitlines()
        normalized = first + '\n' + textwrap.dedent('\n'.join(rest))
        value = dict(cell_type=kind, metadata={}, source=normalized.strip().splitlines(keepends=True))
        if kind == 'code':
            value.update(execution_count=None, outputs=[])
        cells.append(value)
    def code(source):
        cell('code', source)
    cell('markdown', '''# SmolVLM base: generated-token attention on 1,000 test questions
    Select GPU (one GPU or T4 ×2), enable Internet, attach `huyhuy12334/cache-smolvlm`, then Run All.
    This notebook bundles its source; no GitHub checkout is needed. Only the base checkpoint is loaded.

    Random 1,000 test questions without replacement, seed 42. Greedy, at most 32 new tokens.
    Query = each generated content-token position, including self-attention; EOS is excluded.
    Mean heads → generated content tokens within each answer → equal-weight nonempty samples.
    Image = 384 prompt image-token positions; Prompt text = all other prompt tokens; Generated = answer tokens.
    The two figures cover all 30 layers. Empty answers are reported and excluded from the denominator.
    Shading is ±1 sample SD. Histogram population is 384 averaged visual positions per layer.

    Output: `/kaggle/working/generated_attention_outputs/`. Measurement results are saved after each batch
    as individual atomic sample files. Keep this output directory to resume an interrupted session.''')
    code('''# 1. Install pinned dependencies and unpack bundled source; retain Kaggle CUDA Torch.
    import os
    os.environ['USE_TF'] = '0'
    os.environ['USE_FLAX'] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['HF_HOME'] = '/kaggle/working/.cache/huggingface'
    os.environ['PIP_CACHE_DIR'] = '/kaggle/working/.cache/pip'
    from pathlib import Path
    import base64, hashlib, io, json, subprocess, sys, zipfile
    SOURCE = Path('/kaggle/working/smolvlm-generated-attention-source')
    SOURCE_COMMIT = '__COMMIT__'
    SOURCE_ARCHIVE_SHA256 = '__SHA__'
    SOURCE_B64 = '__PAYLOAD__'
    payload = base64.b64decode(SOURCE_B64)
    assert hashlib.sha256(payload).hexdigest() == SOURCE_ARCHIVE_SHA256
    SOURCE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            if not (SOURCE / name).resolve().is_relative_to(SOURCE.resolve()):
                raise ValueError('Unsafe bundled source path')
        archive.extractall(SOURCE)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary=tokenizers',
                    '-r', str(SOURCE / 'smolvlm_attention/requirements.txt')], check=True)
    sys.path.insert(0, str(SOURCE))
    import torch, transformers, tokenizers
    if transformers.__version__ != '4.46.3' or tokenizers.__version__ != '0.20.4':
        raise RuntimeError('Restart kernel after package installation, then Run All')
    if not torch.cuda.is_available():
        raise RuntimeError('Enable a GPU accelerator for this notebook')
    print({'torch': torch.__version__, 'precision': 'float32',
           'gpus': [torch.cuda.get_device_name(i) for i in range(min(2, torch.cuda.device_count()))]})
    '''.replace('__COMMIT__', commit).replace('__SHA__', sha).replace('__PAYLOAD__', encoded))
    code('''# 2. Download pinned prepared data and BASE checkpoint; use the existing Kaggle feature cache.
    from smolvlm_attention.generated_runtime import resolve_generated_artifacts, prepare_generated_run
    OUTPUT = Path('/kaggle/working/generated_attention_outputs')
    CACHE_INPUT_DIR = ''  # Optional: explicit mounted cache root if automatic discovery fails.
    CACHE_RECEIPT = None  # Optional: path to a previous validated_cache.json; inputs are rechecked.
    artifacts = resolve_generated_artifacts(OUTPUT, CACHE_INPUT_DIR)
    print({key: value for key, value in artifacts.items() if key != 'model'})
    print(artifacts['model'])''')
    code('''# 3. Validate cache ONCE (or reuse matching receipt), select 1,000 questions and prepare prompts.
    # CPU/disk work: progress bars appear here; GPUs are used in the next cells.
    RUN_CONFIG = prepare_generated_run(artifacts, source_archive_sha256=SOURCE_ARCHIVE_SHA256,
        source_commit=SOURCE_COMMIT, sample_count=1000, seed=42, max_new_tokens=32,
        receipt_path=CACHE_RECEIPT)
    print(RUN_CONFIG)''')
    code('''# 4. Launch base model on available GPUs: two 500-question shards or one 1,000-question shard.
    from smolvlm_attention.generated_runtime import launch_generated_workers, wait_generated_workers
    workers = launch_generated_workers(RUN_CONFIG)''')
    code('''# 5. Live progress for each GPU and total. Batch starts at 16 per GPU; OOM halves it.
    # Interrupting this WAIT cell leaves workers running; rerun this cell to reconnect.
    wait_generated_workers(RUN_CONFIG)''')
    code('''# 6. Verify full selected-ID coverage, combine samples equally and save numeric results/responses.
    from smolvlm_attention.generated_runtime import load_generated_results
    result, figure_metadata = load_generated_results(RUN_CONFIG)
    FIGURES = OUTPUT / 'figures' / 'base'
    from IPython.display import display, Image, FileLink
    from smolvlm_attention.generated_plotting import (
        plot_attention_fractions, plot_generated_histogram, make_generated_zip)
    print({key: result[key] for key in ('processed_count', 'valid_count', 'empty_count')})''')
    code('''# 7. Figure 1: Image / Prompt text / Generated attention, mean ±1 SD across valid samples.
    fraction_files = plot_attention_fractions(result['mean_groups'], result['std_groups'], FIGURES)
    display(Image(filename=str(next(path for path in fraction_files if path.suffix == '.png'))))''')
    code('''# 8. Figure 2: 384 averaged image-token attention values per layer, log-count histogram.
    histogram_files = plot_generated_histogram(result['mean_visual'], FIGURES)
    display(Image(filename=str(next(path for path in histogram_files if path.suffix == '.png'))))''')
    code('''# 9. Download both figures as PNG/PDF, metadata and captions; no model/cache payload in ZIP.
    zip_path = make_generated_zip(FIGURES, OUTPUT / 'generated_attention_figures.zip', figure_metadata)
    display(FileLink(str(zip_path)))
    print('Numeric summary:', OUTPUT / 'summary.npz')
    print('Answers and token IDs:', OUTPUT / 'responses.jsonl')''')
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python', version='3.11')), nbformat=4, nbformat_minor=5)
    target = ROOT / 'notebooks/kaggle_smolvlm_generated_attention.ipynb'
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    legacy = ROOT / 'notebooks/kaggle_smolvlm_attention_distribution.ipynb'
    old = json.loads(legacy.read_text(encoding='utf-8'))
    source_cell = next(c for c in old['cells'] if c['cell_type'] == 'code')
    lines = ''.join(source_cell['source']).splitlines(keepends=True)
    for statement in ast.parse(''.join(lines)).body:
        if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name):
            name = statement.targets[0].id
            if name in ('SOURCE_B64', 'SOURCE_ARCHIVE_SHA256'):
                value = encoded if name == 'SOURCE_B64' else sha
                lines[statement.lineno - 1] = f'{name} = {value!r}\n'
    source_cell['source'] = ''.join(lines).splitlines(keepends=True)
    legacy.write_text(json.dumps(old, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'Built {target.name}; refreshed legacy source bundle; SHA256={sha}')


if __name__ == '__main__':
    build()
