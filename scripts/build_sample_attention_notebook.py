"""Build the seven-cell full-test per-question attention export notebook."""
import ast
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def setup_cell():
    header = textwrap.dedent('''\
        # 1. Push Visualize first; this cell clones/pulls source without deleting outputs.
        import os
        from pathlib import Path
        SOURCE_REPOSITORY = 'https://github.com/huyvanzzz/Log_Metrics.git'
        SOURCE_BRANCH = 'smolvlm-attention'
        SOURCE = Path('/kaggle/working/visualize-github-source')
        OUTPUT = Path('/kaggle/working/smolvlm_prompt_attention_outputs')
        RUN_MODE = 'auto'  # auto, measure, resume, export
        os.environ['USE_TF'] = '0'
        os.environ['USE_FLAX'] = '0'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        os.environ['HF_HOME'] = '/kaggle/working/.cache/huggingface'
        os.environ['PIP_CACHE_DIR'] = '/kaggle/working/.cache/pip'
        ''')
    helper = (ROOT / 'scripts/kaggle_source_setup.py').read_text(encoding='utf-8').strip()
    tail = textwrap.dedent('''\

        ensure_no_workers([OUTPUT, Path('/kaggle/working/attention_outputs'),
                           Path('/kaggle/working/generated_attention_outputs'),
                           Path('/kaggle/working/dinov2_attention_outputs')])
        identity = sync_source(SOURCE_REPOSITORY, SOURCE_BRANCH, SOURCE)
        SOURCE_COMMIT, SOURCE_SHA256 = identity['commit'], identity['sha256']
        activate_source(SOURCE)
        from smolvlm_attention.sample_runtime import resolve_run_mode
        BOOT_MODE = resolve_run_mode(OUTPUT, RUN_MODE)
        if BOOT_MODE != 'export':
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary=tokenizers',
                            '-r', str(SOURCE/'smolvlm_attention/sample_requirements.txt')], check=True)
            import transformers, tokenizers
            if transformers.__version__ != '4.46.3' or tokenizers.__version__ != '0.20.4':
                raise RuntimeError('Restart kernel after dependency installation, then rerun cell 1.')
        RUN_CONFIG = OUTPUT/'sample_run.json'
        print(dict(mode=BOOT_MODE, output=str(OUTPUT), commit=SOURCE_COMMIT))
        ''')
    return header + '\n' + helper + '\n' + tail


def make_notebook():
    intro = '''# SmolVLM base: per-question prompt attention export

Enable Internet and GPU (one GPU or T4 x2). Attach your existing SmolVLM cache;
automatic discovery is supported. Push the implementation before cell 1.

All **16,817 test questions**, all **30 layers**. Query = last valid native prompt
token, including the assistant suffix, predicting the first answer token.
Softmax over all valid prompt keys including self; select 384 visual positions;
mean heads only. No generation, pruning, image-only normalization or sample mean.

**Initial batch 32 per GPU**; CUDA OOM halves it. FP32, no TF32. Two GPUs process
8,409/8,408 questions. Separate shards are committed every 100 successful batches,
at completion and on handled measurement errors. Progress distinguishes computed
from committed rows. A hard interruption can require repeating uncommitted work.

New/partial run: cells 1–7. Complete shards or existing export: **cells 1, 2, 6, 7**;
these do not download a model or scan features. Interrupting cell 5 leaves workers
running; rerun cell 5 to reconnect. Do not pull while workers run.

Keep the entire OUTPUT to resume. Changing GPU shard count or measurement identity
requires a new OUTPUT. Raw attention is ~739 MiB; shards + export + uncompressed ZIP
need ~2.2 GiB plus mapping/artifacts. No files are automatically deleted.
'''
    sources = [setup_cell(), textwrap.dedent('''\
        # 2. Defaults work without editing when one matching cache is attached.
        BATCH_SIZE = 32  # Per GPU; only CUDA OOM reduces it.
        GPU_COUNT = None  # Auto: up to two GPUs. Resume must retain original topology.
        CACHE_INPUT = ''  # Set only if multiple matching mounted caches exist.
        PREPARED_INPUT = None  # Optional existing prepared-data directory; otherwise pinned HF.
        from smolvlm_attention.sample_runtime import prepare_run, launch_workers, wait_workers
        from smolvlm_attention.sample_store import finalize_export, open_prompt_attention, bundle_export
        print(dict(initial_batch_per_gpu=BATCH_SIZE, mode=BOOT_MODE))
        '''), textwrap.dedent('''\
        # 3. Validate cache once and tokenize every test prompt without answer/truncation.
        if BOOT_MODE != 'export':
            RUN_CONFIG = prepare_run(OUTPUT, mode=BOOT_MODE, batch_size=BATCH_SIZE,
                gpu_count=GPU_COUNT, cache_input_dir=CACHE_INPUT, prepared_dir=PREPARED_INPUT,
                source_commit=SOURCE_COMMIT, source_sha256=SOURCE_SHA256)
        print(RUN_CONFIG)
        '''), textwrap.dedent('''\
        # 4. Each worker strict-loads base and checks native eager/batch parity before measuring.
        if BOOT_MODE != 'export':
            launch_workers(RUN_CONFIG)
        '''), textwrap.dedent('''\
        # 5. Live tqdm per GPU + total. Interrupting does not cancel workers.
        if BOOT_MODE != 'export':
            wait_workers(RUN_CONFIG)
        '''), textwrap.dedent('''\
        # 6. Verify shard checksums and merge in original test order. No model/cache needed.
        EXPORT = finalize_export(RUN_CONFIG)
        RESULTS = open_prompt_attention(EXPORT)
        print(dict(shape=RESULTS.attention.shape, dtype=str(RESULTS.attention.dtype),
                   first_sample=RESULTS.samples[0]['sample_id'], export=str(EXPORT)))
        print('Example: sample 0, layer 21, first eight visual tokens:', RESULTS.attention[0, 21, :8])
        '''), textwrap.dedent('''\
        # 7. ZIP only portable export (no checkpoint/cache/shards). Extract before memory mapping.
        ZIP = bundle_export(EXPORT)
        from IPython.display import FileLink, display
        display(FileLink(str(ZIP)))
        for name in ('attention.npy', 'samples.jsonl', 'metadata.json', 'manifest.json', 'README.md'):
            display(FileLink(str(EXPORT/name)))
        ''')]
    cells = [dict(cell_type='markdown', id='sample-intro', metadata={}, source=intro.splitlines(keepends=True))]
    for index, source in enumerate(sources, 1):
        ast.parse(source)
        cells.append(dict(cell_type='code', id=f'sample-{index}', metadata={}, execution_count=None,
                          outputs=[], source=source.splitlines(keepends=True)))
    return dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python', version='3.11')), nbformat=4, nbformat_minor=5)


def build():
    path = ROOT / 'notebooks/kaggle_smolvlm_prompt_attention_export.ipynb'
    path.write_text(json.dumps(make_notebook(), ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    print(f'Built {path}')


if __name__ == '__main__':
    build()
