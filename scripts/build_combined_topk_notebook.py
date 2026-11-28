"""Reproducible single-notebook projected attention → Top-32 benchmark workflow."""
import ast
import json
from pathlib import Path
import textwrap

ROOT=Path(__file__).resolve().parents[1]


def setup_cell():
    header=textwrap.dedent('''\
        # 1. Push source first; this cell clones/pulls the complete repository.
        import os
        from pathlib import Path
        SOURCE_REPOSITORY = 'https://github.com/huyvanzzz/Log_Metrics.git'
        SOURCE_BRANCH = 'smolvlm-attention'
        SOURCE = Path('/kaggle/working/visualize-github-source')
        OUTPUT = Path('/kaggle/working/smolvlm_combined_topk_outputs')
        RUN_MODE = 'auto'  # auto / measure / resume / score
        os.environ['USE_TF'] = '0'
        os.environ['USE_FLAX'] = '0'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        os.environ['HF_HOME'] = '/kaggle/working/.cache/huggingface'
        os.environ['PIP_CACHE_DIR'] = '/kaggle/working/.cache/pip'
        ''')
    helper=(ROOT/'scripts/kaggle_source_setup.py').read_text(encoding='utf-8').strip()
    tail=textwrap.dedent('''\

        ensure_no_workers([OUTPUT])
        PACKAGES = ('smolvlm_combined_topk', 'smolvlm_predictor_io',
                    'smolvlm_topk_benchmark', 'smolvlm_attention', 'smolvlm_inference')
        identity = sync_source(SOURCE_REPOSITORY, SOURCE_BRANCH, SOURCE, packages=PACKAGES)
        SOURCE_COMMIT, SOURCE_SHA256 = identity['commit'], identity['sha256']
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary=tokenizers',
            '-r', str(SOURCE/'smolvlm_topk_benchmark/requirements.txt')], check=True)
        import shutil
        if shutil.which('java') is None:
            if shutil.which('apt-get') is None:
                raise RuntimeError('METEOR requires Java 8+. Install Java, then rerun cell 1.')
            subprocess.run(['apt-get', 'update', '-qq'], check=True)
            subprocess.run(['apt-get', 'install', '-y', '-qq', 'openjdk-17-jre-headless'], check=True)
        activate_source(SOURCE, packages=PACKAGES)
        import transformers, tokenizers
        if transformers.__version__ != '4.46.3' or tokenizers.__version__ != '0.20.4':
            raise RuntimeError('Restart kernel after installing dependencies, then rerun cell 1.')
        from smolvlm_combined_topk.runtime import resolve_run_mode
        BOOT_MODE = resolve_run_mode(OUTPUT, RUN_MODE)
        RUN_CONFIG = OUTPUT/'combined_run.json'
        BENCHMARK_CONFIG = OUTPUT/'benchmark'/'benchmark_run.json'
        print(dict(mode=BOOT_MODE, output=str(OUTPUT), source_commit=SOURCE_COMMIT))
        ''')
    return header+'\n'+helper+'\n'+tail


def make_notebook():
    intro='''# SmolVLM base: combined attention → global Top-32 benchmark

Enable **Internet + T4 x2** and attach your **completed projected cache** containing
`projected.npy`, `index.json`, `manifest.json`. No raw feature cache or images are
needed. Push this implementation to `smolvlm-attention` before running cell 1.

Run **all ten cells**. Stage A splits the 16,817 test questions across GPUs and saves
FP32 `[16817,10,384]` scores for layers **10,15,16,17,21,22,23,25,26,27**.
For each sample: `S(q,v)=mean_heads(A)[q,v]+sum_t mean_heads(A)[q,t]*mean_heads(A)[t,v]`,
within the same layer. `q` is the last native-prompt token including assistant suffix;
`t` includes every valid non-visual token except `q`, including camera names/specials.
Full causal-key softmax, no image renormalization or averaging across samples/layers.

Stage B keeps global Top-32 and physically removes other visual embeddings before LLM
layer 0, preserving non-visual tokens and original RoPE positions. GPU 0 finishes
**10 → 15 → 16 → 17 → 21**; GPU 1 finishes **22 → 23 → 25 → 26 → 27**.
Every config uses all 16,817 questions; **no full-token base config** is run.
Both stages use **fixed batch 64**, FP32/no TF32; OOM fails without reducing batch.
Generation is native cached greedy, at most **512 new tokens**, stopping normally at EOS.

Each wait cell shows tqdm. Interrupting a wait leaves workers running; rerun that cell
to reconnect. Never pull while workers are alive. Save the **entire OUTPUT folder** for
partial resume across sessions. The final uncompressed ZIP contains completed attention,
selections, predictions and corpus metrics; no cache/checkpoint/work shards.

`auto` resumes/skips completed phases. For rescoring complete predictions without GPU,
checkpoint or projected payload, select `RUN_MODE='score'` and run cells **1,2,9**.
Cell 10 additionally needs the attention export to package it. Extract ZIP before memmapping.
There are no figures, predictor, training or teacher targets in this notebook.
'''
    sources=[setup_cell(),textwrap.dedent('''\
        # 2. Fixed experiment; only artifact paths are optional.
        BATCH_SIZE = 64
        LAYER_IDS = (10,15,16,17,21,22,23,25,26,27)
        PROJECTED_CACHE_INPUT = ''  # Auto-detect; set exact directory if multiple are found.
        PREPARED_INPUT = None  # Existing prepared folder, otherwise pinned HF download.
        CHECKPOINT_INPUT = None  # Existing base .pt file, otherwise pinned HF download.
        from smolvlm_combined_topk.config import BATCH_SIZE as REQUIRED_BATCH, LAYER_IDS as REQUIRED_LAYERS
        if BATCH_SIZE != REQUIRED_BATCH or LAYER_IDS != REQUIRED_LAYERS:
            raise ValueError('Requires fixed batch 64 and the ten specified layers.')
        from smolvlm_combined_topk.runtime import prepare_run, launch_attention, wait_workers
        from smolvlm_combined_topk.runtime import finalize_attention, launch_benchmark
        from smolvlm_combined_topk.benchmark import score_results, bundle_results
        ATTENTION_EXPORT = OUTPUT/'attention_export'
        print(dict(batch=BATCH_SIZE, layers=LAYER_IDS, mode=BOOT_MODE))
        '''),textwrap.dedent('''\
        # 3. Check Java/scorers BEFORE expensive measurement, then validate artifacts/prompts.
        if BOOT_MODE != 'score':
            from smolvlm_topk_benchmark.runtime import _scorer_command
            _scorer_command(OUTPUT/'metric_setup_preflight.json')
            RUN_CONFIG = prepare_run(OUTPUT, mode=BOOT_MODE,
                projected_input=PROJECTED_CACHE_INPUT, prepared_dir=PREPARED_INPUT,
                checkpoint=CHECKPOINT_INPUT, source_commit=SOURCE_COMMIT,
                source_sha256=SOURCE_SHA256)
        print(RUN_CONFIG)
        '''),textwrap.dedent('''\
        # 4. One attention worker/GPU; each question collected only once, all ten layers.
        if BOOT_MODE != 'score':
            launch_attention(RUN_CONFIG)
        '''),textwrap.dedent('''\
        # 5. Progress reflects durable committed rows. Interrupt leaves workers running.
        if BOOT_MODE != 'score':
            wait_workers(RUN_CONFIG, phase='attention')
        '''),textwrap.dedent('''\
        # 6. Verify/merge per-sample FP32 attention; prepare stable global Top-32.
        if BOOT_MODE != 'score':
            ATTENTION_EXPORT, BENCHMARK_CONFIG = finalize_attention(RUN_CONFIG)
        print(dict(attention=str(ATTENTION_EXPORT), benchmark=str(BENCHMARK_CONFIG)))
        '''),textwrap.dedent('''\
        # 7. Five configs/GPU. Complete all questions of a config before the next.
        if BOOT_MODE != 'score':
            launch_benchmark(BENCHMARK_CONFIG)
        '''),textwrap.dedent('''\
        # 8. tqdm per configuration, per GPU and total; rerun to reconnect.
        if BOOT_MODE != 'score':
            wait_workers(BENCHMARK_CONFIG, phase='benchmark')
        '''),textwrap.dedent('''\
        # 9. Full-corpus scoring only after all ten configs complete. CPU-only rerunnable.
        SUMMARY = score_results(BENCHMARK_CONFIG)
        import pandas as pd
        display(pd.DataFrame(SUMMARY['results']))
        '''),textwrap.dedent('''\
        # 10. ZIP_STORED; contains completed attention + predictions/metrics, no large caches.
        ARCHIVE = bundle_results(BENCHMARK_CONFIG, ATTENTION_EXPORT)
        from IPython.display import FileLink, display
        display(FileLink(str(ARCHIVE)))
        display(FileLink(str(ATTENTION_EXPORT/'attention.npy')))
        display(FileLink(str(OUTPUT/'benchmark'/'summary.csv')))
        display(FileLink(str(OUTPUT/'benchmark'/'summary.json')))
        ''')]
    cells=[dict(cell_type='markdown',metadata={},source=intro.splitlines(keepends=True))]
    for source in sources:
        ast.parse(source)
        cells.append(dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=source.splitlines(keepends=True)))
    return dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python',version='3.11')),nbformat=4,nbformat_minor=5)


if __name__=='__main__':
    path=ROOT/'notebooks/kaggle_smolvlm_combined_attention_topk_benchmark.ipynb'
    path.write_text(json.dumps(make_notebook(),ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
