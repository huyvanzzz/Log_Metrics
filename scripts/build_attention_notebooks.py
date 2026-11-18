"""Refresh both attention notebooks with a shared, real Git clone/pull setup cell."""
import ast
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def setup_cell(generated=False):
    output = 'generated_attention_outputs' if generated else 'attention_outputs'
    run_name = 'generated_run.json' if generated else 'run.json'
    header = textwrap.dedent('''\
        # 1. PULL ALL SOURCE from GitHub, install dependencies, and activate the updated code.
        # Rerun this cell after pushing code; finish running workers before updating source.
        import os
        from pathlib import Path
        SOURCE_REPOSITORY = "https://github.com/huyvanzzz/Log_Metrics.git"
        SOURCE_BRANCH = "smolvlm-attention"  # Change only if your pushed code is on another branch.
        SOURCE = Path('/kaggle/working/visualize-github-source')
        os.environ['USE_TF'] = '0'
        os.environ['USE_FLAX'] = '0'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        os.environ['HF_HOME'] = '/kaggle/working/.cache/huggingface'
        os.environ['PIP_CACHE_DIR'] = '/kaggle/working/.cache/pip'
        ''')
    helper = (ROOT / 'scripts/kaggle_source_setup.py').read_text(encoding='utf-8').strip()
    tail = textwrap.dedent('''\

        ensure_no_workers([Path('/kaggle/working/attention_outputs'),
                           Path('/kaggle/working/generated_attention_outputs')])
        identity = sync_source(SOURCE_REPOSITORY, SOURCE_BRANCH, SOURCE)
        SOURCE_COMMIT = identity["commit"]
        SOURCE_ARCHIVE_SHA256 = identity["sha256"]
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary=tokenizers',
                        '-r', str(SOURCE / 'smolvlm_attention/requirements.txt')], check=True)
        activate_source(SOURCE)
        import torch, transformers, tokenizers
        if transformers.__version__ != '4.46.3' or tokenizers.__version__ != '0.20.4':
            raise RuntimeError('Restart kernel after installing pinned packages, then run this setup cell again')
        ''')
    gpu = ("if not torch.cuda.is_available():\n    raise RuntimeError('Enable a GPU accelerator')\n" if generated else
           "if torch.cuda.device_count() < 2:\n    raise RuntimeError('Enable two GPUs, e.g. T4 x2')\n")
    finish = textwrap.dedent(f'''\
        OUTPUT = Path('/kaggle/working/{output}')
        RUN_CONFIG = OUTPUT / '{run_name}'  # Existing results can be redrawn without rerunning preparation.
        print({{'source': str(SOURCE), 'commit': SOURCE_COMMIT, 'source_sha256': SOURCE_ARCHIVE_SHA256,
               'batch_per_gpu': 16, 'precision': 'float32',
               'gpus': [torch.cuda.get_device_name(i) for i in range(min(2, torch.cuda.device_count()))]}})
        print('Source updated. New run: continue cells in order. Redraw existing results: skip preparation/workers.')
        ''')
    source = header + '\n' + helper + '\n' + tail + gpu + finish
    ast.parse(source)
    return source


def build():
    for generated, filename in ((False, 'kaggle_smolvlm_attention_distribution.ipynb'),
                                (True, 'kaggle_smolvlm_generated_attention.ipynb')):
        path = ROOT / 'notebooks' / filename
        notebook = json.loads(path.read_text(encoding='utf-8'))
        code = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
        code[0]['source'] = setup_cell(generated).splitlines(keepends=True)
        if generated:
            notebook['cells'][0]['source'] = textwrap.dedent('''\
                # SmolVLM base: generated-token attention on 1,000 test questions
                Enable Internet and one GPU or T4 ×2; attach `huyhuy12334/cache-smolvlm`.
                **Cell 1 clones/pulls the entire GitHub repository** and activates the updated source.
                Push changes to the configured branch before rerunning cell 1. Batch starts at **16 per GPU**;
                only CUDA OOM reduces it. Progress bars cover validation, preparation and measurement.

                Seed 42 selects 1,000 test questions without replacement. Greedy, at most 32 new tokens.
                Query = generated content-token position, including self; EOS is excluded.
                Mean heads → generated tokens within each answer → equal-weight nonempty questions.
                Empty answers are reported, not replaced. Three curves show mean ±1 sample SD.
                Histogram: 384 already-averaged visual-token values per layer; all 30 layers.

                New run: run cells in order. **Redraw saved results: cell 1, then cells 6–9 only.**
                Outputs remain in `/kaggle/working/generated_attention_outputs/`, outside the Git checkout.
                Source changes affect measurement fingerprints; do not overwrite an existing run configuration
                merely to update figures. Keep the original config/results for redraws.
                ''').splitlines(keepends=True)
            code[4]['source'] = ("# 5. tqdm per GPU and total; batch starts at 16/GPU, reduced only on OOM.\n"
                                 "# Interrupting this wait leaves workers running; rerun to reconnect.\n"
                                 "wait_generated_workers(RUN_CONFIG)").splitlines(keepends=True)
        else:
            notebook['cells'][0]['source'] = textwrap.dedent('''\
                # SmolVLM: test-set mean visual attention (base vs reverse)
                Enable Internet, select T4 ×2 and attach `huyhuy12334/cache-smolvlm`.
                **Cell 1 clones/pulls the entire GitHub repository** and activates the updated source.
                Push changes to the configured branch before rerunning cell 1.
                Both base and reverse start at **batch 16/GPU**; only CUDA OOM reduces it.
                tqdm shows cache validation, prompt preparation, each model and total measurement progress.

                FP32. Mean heads then all 16,817 test questions. Final prompt query predicts the first answer token.
                Token Y axes and spatial colors use log scales shared across both models.
                Each model/plot type has its own cell; the last cell exports a ZIP.

                New run: run cells in order. **Redraw saved results: cell 1, then cells 7–14 only.**
                Outputs remain in `/kaggle/working/attention_outputs/`, outside the Git checkout.
                Keep the original config/results when updating only figures; rerunning preparation after
                source changes creates a new source fingerprint and is not needed for redraws.
                ''').splitlines(keepends=True)
            code[5]['source'] = ("# 6. tqdm for base, reverse and total; stage, processed count and batch are shown.\n"
                                 "# Interrupting this wait leaves workers running; rerun to reconnect.\n"
                                 "wait_workers(RUN_CONFIG)").splitlines(keepends=True)
            plot_setup = ''.join(code[6]['source'])
            if 'from smolvlm_attention.runtime import load_results\n' not in plot_setup:
                plot_setup = 'from smolvlm_attention.runtime import load_results\n' + plot_setup
            plot_setup = plot_setup.replace("orders = {name: spec['camera_order'] for name, spec in artifacts['models'].items()}",
                "saved_run = json.loads(Path(RUN_CONFIG).read_text())\norders = {name: spec['camera_order'] for name, spec in saved_run['models'].items()}")
            code[6]['source'] = plot_setup.splitlines(keepends=True)
        for index, cell in enumerate(notebook['cells']):
            cell.setdefault('id', f'attention-{index}')
            if cell['cell_type'] == 'code':
                cell.update(execution_count=None, outputs=[])
                ast.parse(''.join(cell['source']))
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        print(f'Updated {filename}: first-cell Git clone/pull, batch16 and tqdm')


if __name__ == '__main__':
    build()
