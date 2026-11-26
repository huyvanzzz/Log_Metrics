"""Reproducible ten-cell Kaggle notebook for base combined visual attention."""
import ast
import json
from pathlib import Path
import textwrap

ROOT=Path(__file__).resolve().parents[1]


def setup_cell():
    header=textwrap.dedent('''\
        # 1. Push code first; this cell clones/pulls ALL source and refreshes imports.
        import os
        from pathlib import Path
        SOURCE_REPOSITORY = 'https://github.com/huyvanzzz/Log_Metrics.git'
        SOURCE_BRANCH = 'smolvlm-attention'
        SOURCE = Path('/kaggle/working/visualize-github-source')
        OUTPUT = Path('/kaggle/working/smolvlm_combined_visual_attention_outputs')
        RUN_MODE = 'auto'  # auto / measure / resume / plot
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
        identity = sync_source(SOURCE_REPOSITORY, SOURCE_BRANCH, SOURCE)
        SOURCE_COMMIT, SOURCE_SHA256 = identity['commit'], identity['sha256']
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary=tokenizers',
            '-r', str(SOURCE/'smolvlm_attention/requirements.txt'),
            '-r', str(SOURCE/'smolvlm_attention/sample_requirements.txt')], check=True)
        activate_source(SOURCE)
        import transformers, tokenizers
        if transformers.__version__ != '4.46.3' or tokenizers.__version__ != '0.20.4':
            raise RuntimeError('Restart kernel after installing dependencies, then rerun cell 1.')
        from smolvlm_attention.combined_runtime import resolve_run_mode
        BOOT_MODE = resolve_run_mode(OUTPUT, RUN_MODE)
        RUN_CONFIG = OUTPUT/'combined_run.json'
        print(dict(mode=BOOT_MODE, output=str(OUTPUT), source_commit=SOURCE_COMMIT))
        ''')
    return header+'\n'+helper+'\n'+tail


def make_notebook():
    intro='''# SmolVLM base: same-layer combined visual attention

Enable Internet and **T4 x2**, attach the existing `huyhuy12334/cache-smolvlm` feature cache.
Base only, full 16,817 test questions. Prepared data/checkpoint are downloaded at pinned
revisions if missing; no raw images, reverse checkpoint, generation or benchmark.

**Layers: 0,2,4,...,28 only. Fixed batch 64/GPU; OOM fails, no reduction or truncation.**
Two shards contain 8,409 and 8,408 questions (last batches 25 and 24). FP32, no TF32.
For each sample, mean heads FIRST, then compute in the SAME layer:

`S(q,v) = A_mean[q,v] + sum_t A_mean[q,t] * A_mean[t,v]`.

`q` is the last valid native prompt position, including the assistant suffix. `t` includes
all valid non-visual tokens, camera labels and special tokens; excludes padding and `q`.
Native causal masking and softmax over all allowed keys including self. No image-only
renormalization, rollout or value scaling. This is a defined combined score, not causal
contribution. Equal sample weighting after computing each sample's score.

New/resume run: cells **1–10**. Redraw complete results: cells **1,2,6–10**, with
`RUN_MODE='plot'` or auto after a completed merge. Plot mode does not load model/cache.
Cell 4 launches workers in the background; cell 5 waits. Interrupting cell 5 leaves
workers running; rerun it to reconnect. Never pull while workers are active.
Save the entire OUTPUT to resume a partial run across sessions. Figures ZIP contains
only six PNG/PDF figures, metadata and caption; no model/cache or shard snapshots.
The older last-query-only attention export cannot reconstruct this new score.
'''
    sources=[setup_cell(),textwrap.dedent('''\
        # 2. Experiment settings. Optional paths are discovered automatically.
        BATCH_SIZE = 64
        LAYER_IDS = tuple(range(0, 29, 2))
        QUERY_CHUNK_SIZE = 8
        CACHE_INPUT = ''  # Set an explicit mounted cache path if several are found.
        PREPARED_INPUT = None  # Existing prepared folder; otherwise pinned HF download.
        from smolvlm_attention.combined_state import BATCH_SIZE as REQUIRED_BATCH, LAYER_IDS as REQUIRED_LAYERS
        if BATCH_SIZE != REQUIRED_BATCH or LAYER_IDS != REQUIRED_LAYERS or QUERY_CHUNK_SIZE != 8:
            raise ValueError('This experiment requires batch 64, layers 0,2,...,28 and query chunks of 8.')
        from smolvlm_attention.combined_runtime import prepare_run, launch_workers, wait_workers
        from smolvlm_attention.combined_runtime import finalize_results, load_results
        print(dict(model='huyvanzzz/smolvlm256M_base', batch=64, layers=LAYER_IDS, mode=BOOT_MODE))
        '''),textwrap.dedent('''\
        # 3. Resolve base artifacts, validate feature cache ONCE, prepare all test prompts.
        if BOOT_MODE != 'plot':
            RUN_CONFIG = prepare_run(OUTPUT, mode=BOOT_MODE, cache_input_dir=CACHE_INPUT,
                prepared_dir=PREPARED_INPUT, source_commit=SOURCE_COMMIT, source_sha256=SOURCE_SHA256)
        print(RUN_CONFIG)
        '''),textwrap.dedent('''\
        # 4. Two independent GPU workers. Each performs native numerical preflight first.
        if BOOT_MODE != 'plot':
            launch_workers(RUN_CONFIG)
        '''),textwrap.dedent('''\
        # 5. Committed progress per GPU + total. Interrupt does NOT kill workers.
        if BOOT_MODE != 'plot':
            wait_workers(RUN_CONFIG)
        '''),textwrap.dedent('''\
        # 6. Verify both shards, weighted merge, and read saved results (no GPU needed).
        finalize_results(RUN_CONFIG)
        MEAN, METADATA, SCALES, FIGURE_METADATA = load_results(RUN_CONFIG, plot_source_commit=SOURCE_COMMIT)
        from smolvlm_inference.cache import CAMERA_ORDER
        from smolvlm_attention.plotting import plot_token_atlas, plot_spatial_atlas, plot_layer_histogram
        FIGURES = OUTPUT/'figures'
        TITLE = 'SmolVLM base — same-layer combined visual attention'
        LABEL = 'Mean combined attention score'
        print(dict(count=METADATA['count'], shape=MEAN.shape, figures=str(FIGURES)))
        '''),textwrap.dedent('''\
        # 7. Fifteen layers in a 5x3 atlas, 384 bars per panel, log value/color scales.
        TOKEN_FILES = plot_token_atlas(MEAN, CAMERA_ORDER, FIGURES/'base', title=TITLE,
            scales=SCALES, layer_ids=LAYER_IDS, value_label=LABEL)
        from IPython.display import Image, display
        display(Image(filename=str(TOKEN_FILES[0])))
        '''),textwrap.dedent('''\
        # 8. Same layers: each panel keeps six original-orientation 8x8 camera maps in one row.
        SPATIAL_FILES = plot_spatial_atlas(MEAN, CAMERA_ORDER, FIGURES/'base', title=TITLE,
            scales=SCALES, layer_ids=LAYER_IDS, value_label=LABEL)
        display(Image(filename=str(SPATIAL_FILES[0])))
        '''),textwrap.dedent('''\
        # 9. Only layers 0,2,...,28; 384 sample-averaged scores/layer, 100 shared linear bins.
        HISTOGRAM_FILES = plot_layer_histogram(MEAN, CAMERA_ORDER, FIGURES/'base', title=TITLE,
            scales=SCALES, layer_ids=LAYER_IDS, value_label=LABEL)
        display(Image(filename=str(HISTOGRAM_FILES[0])))
        '''),textwrap.dedent('''\
        # 10. ZIP exactly six figures + metadata + caption, no model/cache/work state.
        from smolvlm_attention.plotting import make_figures_zip
        ZIP = make_figures_zip(FIGURES, OUTPUT/'combined_attention_figures.zip', FIGURE_METADATA,
            models=('base',), caption=FIGURE_METADATA['caption'])
        from IPython.display import FileLink
        display(FileLink(str(ZIP)))
        ''')]
    cells=[dict(cell_type='markdown',id='combined-intro',metadata={},source=intro.splitlines(keepends=True))]
    for i,source in enumerate(sources,1):
        ast.parse(source)
        cells.append(dict(cell_type='code',id=f'combined-{i}',metadata={},execution_count=None,
                          outputs=[],source=source.splitlines(keepends=True)))
    return dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python',version='3.11')),nbformat=4,nbformat_minor=5)


def build():
    path=ROOT/'notebooks/kaggle_smolvlm_combined_visual_attention.ipynb'
    path.write_text(json.dumps(make_notebook(),ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(f'Built {path}')


if __name__=='__main__':build()
