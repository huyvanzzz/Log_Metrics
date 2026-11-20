"""Build two eight-config waves, four sequential full-test configs per GPU."""
import ast
import json
from pathlib import Path
import textwrap

ROOT=Path(__file__).resolve().parents[1]


def setup_cell():
    header=textwrap.dedent('''\
        # 1. Push code first. For the later wave change ONLY CONFIG_SET to 'second'.
        import os
        from pathlib import Path
        SOURCE_REPOSITORY = 'https://github.com/huyvanzzz/Log_Metrics.git'
        SOURCE_BRANCH = 'smolvlm-attention'
        SOURCE = Path('/kaggle/working/visualize-github-source')
        OUTPUT = Path('/kaggle/working/smolvlm_topk_benchmark_outputs_batch64_two_waves')
        CONFIG_SET = 'first'  # first: 0,1,2,9,10,11,15,16; second: 17,21,22,23,25,26,27,base
        RUN_MODE = 'auto'  # auto, measure, resume, score
        os.environ['USE_TF'] = '0'
        os.environ['USE_FLAX'] = '0'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        os.environ['HF_HOME'] = '/kaggle/working/.cache/huggingface'
        os.environ['PIP_CACHE_DIR'] = '/kaggle/working/.cache/pip'
        ''')
    helper=(ROOT/'scripts/kaggle_source_setup.py').read_text(encoding='utf-8').strip()
    tail=textwrap.dedent('''\

        ensure_no_workers([OUTPUT, Path('/kaggle/working/attention_outputs'),
                           Path('/kaggle/working/generated_attention_outputs'),
                           Path('/kaggle/working/dinov2_attention_outputs')])
        PACKAGES = ('smolvlm_topk_benchmark', 'smolvlm_attention', 'smolvlm_inference')
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
        from smolvlm_topk_benchmark.runtime import resolve_run_mode
        BOOT_MODE = resolve_run_mode(OUTPUT, CONFIG_SET, RUN_MODE)
        RUN_CONFIG = OUTPUT/'benchmark_run.json'
        print(dict(config_set=CONFIG_SET, mode=BOOT_MODE, output=str(OUTPUT), commit=SOURCE_COMMIT))
        ''')
    return header+'\n'+helper+'\n'+tail


def make_notebook():
    intro='''# SmolVLM base: single-layer Top-32 answer-quality benchmark

Enable Internet and **T4 x2**. Attach the existing SmolVLM feature cache and the
**complete per-question attention export** (attention.npy, samples.jsonl,
metadata.json, manifest.json, README.md), extracted as a Kaggle Input directory.
Push the complete implementation before cell 1. No raw images are downloaded.

Default `CONFIG_SET='first'`: GPU 0 runs **0,1,2,9**; GPU 1 runs **10,11,15,16**.
Each GPU finishes all test questions for one config before starting the next.
Later set `CONFIG_SET='second'`: GPU 0 runs **17,21,22,23**;
GPU 1 runs **25,26,27,base**. Base runs last on GPU 1, only in the second wave.
Keep the same OUTPUT or restore it before
starting a new session. Optional PREVIOUS_RESULTS reads a prior extracted ZIP
without writing into Input and includes it in the combined comparison.

Two resident workers, one model per GPU and four sequential configs each. Every
config processes all 16,817 test questions. **Fixed batch 64**, last batch 49;
OOM stops the affected worker, never reduces batch. Greedy, max 512 new tokens,
FP32/no TF32, EOS may stop earlier. All non-visual tokens are retained. Top-32
is global over six cameras at one ranking layer; actual deletion occurs before
LLM layer 0. Original RoPE positions are preserved; physical cache slots are compact.

New/partial measurement: run cells 1–8. Complete generation: cells **1,2,7,8**
rescore/export without loading a model or reading cache/attention. Interrupting
cell 6 leaves workers alive; rerun it to reconnect. Never pull while workers run.
Keep the full OUTPUT for partial-generation resume. ZIP exports complete results
and inputs/provenance, excludes large artifacts and partial batch journals.

Quality metrics only: BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr. No timing claims while
GPUs are shared. Base is measured in wave two; no old beam-search baseline is reused.
Prior batch-32 or 15-config scheduling state is incompatible: use this new OUTPUT.
'''
    sources=[setup_cell(),textwrap.dedent('''\
        # 2. These optional paths are normally discovered automatically.
        BATCH_SIZE = 64  # Fixed. OOM is an error; no automatic reduction.
        from smolvlm_topk_benchmark.config import BATCH_SIZE as REQUIRED_BATCH_SIZE
        if BATCH_SIZE != REQUIRED_BATCH_SIZE:
            raise ValueError(f'This experiment requires batch {REQUIRED_BATCH_SIZE} for every config.')
        ATTENTION_INPUT = None  # Extracted complete attention export; set if multiple are found.
        CACHE_INPUT = ''  # Existing mounted cache; set if multiple matches are found.
        PREPARED_INPUT = None  # Existing manifest/train/val/test; otherwise pinned HF download.
        PREVIOUS_RESULTS = None  # Optional extracted prior result ZIP root for combined scoring.
        from smolvlm_topk_benchmark.runtime import prepare_run, launch_workers, wait_workers
        from smolvlm_topk_benchmark.runtime import score_results, bundle_results
        print(dict(batch=BATCH_SIZE, greedy=True, max_new_tokens=512, set=CONFIG_SET, mode=BOOT_MODE))
        '''),textwrap.dedent('''\
        # 3. Locate artifacts only for new/partial generation; no model weights are loaded here.
        ARTIFACTS = None
        if BOOT_MODE != 'score':
            from smolvlm_topk_benchmark.data import find_attention
            from smolvlm_attention.sample_runtime import resolve_artifacts
            ATTENTION_INPUT, CHECKED_ATTENTION = find_attention(ATTENTION_INPUT)
            del CHECKED_ATTENTION
            ARTIFACTS = resolve_artifacts(OUTPUT, CACHE_INPUT, PREPARED_INPUT)
            print(dict(attention=str(ATTENTION_INPUT), cache=ARTIFACTS['cache_dir']))
        '''),textwrap.dedent('''\
        # 4. Validate + align all samples, prepare Top-32, and smoke-test all seven CPU scorers.
        # Native numerical generation preflight runs inside each worker before full measurement.
        if BOOT_MODE != 'score':
            RUN_CONFIG = prepare_run(OUTPUT, config_set=CONFIG_SET, mode=BOOT_MODE,
                attention_input=ATTENTION_INPUT, cache_input_dir=CACHE_INPUT,
                prepared_dir=PREPARED_INPUT, artifacts=ARTIFACTS,
                source_commit=SOURCE_COMMIT, source_sha256=SOURCE_SHA256)
        print(RUN_CONFIG)
        '''),textwrap.dedent('''\
        # 5. Two workers; each GPU finishes four configs sequentially. Base is in wave two only.
        if BOOT_MODE != 'score':
            launch_workers(RUN_CONFIG, CONFIG_SET)
        '''),textwrap.dedent('''\
        # 6. Progress per config + GPU + total. Interrupting this cell keeps workers running.
        if BOOT_MODE != 'score':
            wait_workers(RUN_CONFIG, CONFIG_SET)
        '''),textwrap.dedent('''\
        # 7. Validate complete coverage and score each entire test corpus, never batch averages.
        SUMMARY = score_results(RUN_CONFIG, CONFIG_SET, previous_results=PREVIOUS_RESULTS)
        import html
        from IPython.display import HTML, display
        columns = ['config', 'k', 'count', 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr']
        table = '<table><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in columns)+'</tr>'
        for row in SUMMARY['results']:
            table += '<tr>'+''.join('<td>'+html.escape(str(round(row[c], 6) if isinstance(row[c], float) else row[c]))+'</td>' for c in columns)+'</tr>'
        display(HTML(table+'</table>'))
        print(dict(baseline_available=SUMMARY['baseline_available'], raw_metrics=True, summary=str(OUTPUT/'summary.csv')))
        '''),textwrap.dedent('''\
        # 8. ZIP portable complete results; no model/cache/attention or partial journals.
        ZIP = bundle_results(RUN_CONFIG)
        from IPython.display import FileLink, display
        display(FileLink(str(ZIP)))
        display(FileLink(str(OUTPUT/'summary.csv')))
        display(FileLink(str(OUTPUT/'summary.json')))
        ''')]
    cells=[dict(cell_type='markdown',id='topk-intro',metadata={},source=intro.splitlines(keepends=True))]
    for i,source in enumerate(sources,1):
        ast.parse(source)
        cells.append(dict(cell_type='code',id=f'topk-{i}',metadata={},execution_count=None,outputs=[],source=source.splitlines(keepends=True)))
    return dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python',version='3.11')),nbformat=4,nbformat_minor=5)


def build():
    path=ROOT/'notebooks/kaggle_smolvlm_topk_benchmark.ipynb'
    path.write_text(json.dumps(make_notebook(),ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(f'Built {path}')


if __name__=='__main__':build()
