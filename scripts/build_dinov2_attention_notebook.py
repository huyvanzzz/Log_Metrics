"""Build the standalone DINOv2/T5 inference notebook with current Git setup."""
import ast
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def setup_cell():
    header = textwrap.dedent('''\
        # 1. Pull the complete Visualize source before importing inference code.
        # Push updates first; finish all attention workers before rerunning this cell.
        import os
        from pathlib import Path
        SOURCE_REPOSITORY = 'https://github.com/huyvanzzz/Log_Metrics.git'
        SOURCE_BRANCH = 'smolvlm-attention'
        SOURCE = Path('/kaggle/working/visualize-github-source')
        OUTPUT = Path('/kaggle/working/dinov2_attention_outputs')
        RUN_MODE = 'auto'  # auto, measure, resume, or plot
        BATCH_PER_GPU = 32  # Fixed initial batch; CUDA OOM halves it.
        os.environ['USE_TF'] = '0'
        os.environ['USE_FLAX'] = '0'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        os.environ['HF_HOME'] = '/kaggle/working/.cache/huggingface'
        os.environ['PIP_CACHE_DIR'] = '/kaggle/working/.cache/pip'
        ''')
    helper = (ROOT/'scripts/kaggle_source_setup.py').read_text(encoding='utf-8').strip()
    tail = textwrap.dedent('''\

        ensure_no_workers([Path('/kaggle/working/attention_outputs'),
                           Path('/kaggle/working/generated_attention_outputs'), OUTPUT])
        identity = sync_source(SOURCE_REPOSITORY, SOURCE_BRANCH, SOURCE,
                               packages=('dinov2_attention', 'smolvlm_attention'))
        SOURCE_COMMIT, SOURCE_SHA256 = identity['commit'], identity['sha256']
        activate_source(SOURCE, packages=('dinov2_attention', 'smolvlm_attention'))
        from dinov2_attention.runtime import resolve_run_mode
        BOOT_MODE = resolve_run_mode(OUTPUT, RUN_MODE)
        if BOOT_MODE != 'plot':
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary=tokenizers',
                            '-r', str(SOURCE/'dinov2_attention/requirements.txt')], check=True)
            import torch, transformers, tokenizers
            if transformers.__version__ != '4.46.3' or tokenizers.__version__ != '0.20.4':
                raise RuntimeError('Restart the kernel after installing pinned packages, then rerun setup.')
        from dinov2_attention.config import BATCH_SIZE
        if BATCH_PER_GPU != BATCH_SIZE:
            raise ValueError('This experiment uses batch 32/GPU; only CUDA OOM reduces it.')
        RUN_CONFIG = OUTPUT/'run.json'
        print(dict(source=str(SOURCE), commit=SOURCE_COMMIT, output=str(OUTPUT),
                   mode=BOOT_MODE, batch_per_gpu=BATCH_SIZE,
                   gpus=('not required for plot' if BOOT_MODE == 'plot' else
                         [torch.cuda.get_device_name(i) for i in range(min(2,torch.cuda.device_count()))])))
        ''')
    return header+'\n'+helper+'\n'+tail


def make_notebook():
    markdown = textwrap.dedent('''\
        # DINOv2-T5 unpruned base: decoder-start visual cross-attention

        Enable Internet and one GPU or T4 x2 for measurement. All artifacts download from pinned
        Hugging Face revisions; raw images and the training repository are not required.
        **Cell 1 clones/pulls the entire Visualize repository.** Push code to the configured branch first.
        Source updates preserve the separate output directory and are blocked while workers run.

        Measure all 16,817 test questions, batch **32/GPU**, FP32. CUDA OOM halves the batch.
        Query is decoder-start `<pad>` (ID 0) with mask 1, predicting the first answer token.
        Mean eight heads, then equal-weight questions. Softmax spans all valid encoder keys;
        the 384 image positions are not renormalized. Four decoder layers are measured.
        These encoder positions already contain bidirectionally mixed information, not raw patches.

        **New/resumed run:** run all cells with `RUN_MODE='auto'`.
        **Redraw only:** set `RUN_MODE='plot'`, run cells 1-2, then 6-10 (no GPU/model/cache scan).
        Partial resume requires the same GPU count and compatible measurement code/library versions.
        Updating only plotting does not invalidate saved measurements.
        tqdm shows validation and worker progress; interrupting the wait cell leaves workers running.
        Figures use shared log attention scales; histogram attention bins are linear and count color is log.
        The last cell downloads a ZIP containing figures, metadata and the measurement caption only.
        ''')
    sources = [setup_cell(), '''# 2. Decide whether to measure, resume, or only redraw saved results.
from dinov2_attention.runtime import resolve_run_mode
MODE = resolve_run_mode(OUTPUT, RUN_MODE)
RUN_CONFIG = OUTPUT/'run.json'
print('Selected mode:', MODE)
''', '''# 3. Download/validate once and prepare all test questions. Plot mode skips this.
if MODE != 'plot':
    from dinov2_attention.runtime import prepare_run
    RUN_CONFIG = prepare_run(OUTPUT, mode=MODE, source_commit=SOURCE_COMMIT,
                             source_sha256=SOURCE_SHA256)
print('Run configuration:', RUN_CONFIG)
''', '''# 4. Launch one/two independent workers; each strict-loads and preflights the model.
if MODE != 'plot':
    from dinov2_attention.runtime import launch_workers
    WORKERS = launch_workers(RUN_CONFIG)
''', '''# 5. Live tqdm per GPU and total. Rerun this cell to reconnect after interruption.
if MODE != 'plot':
    from dinov2_attention.runtime import wait_workers
    wait_workers(RUN_CONFIG)
''', '''# 6. Verify full coverage and load saved attention without loading the model.
from dinov2_attention.runtime import load_results
from dinov2_attention.plotting import shared_scales
from dinov2_attention.config import CAMERA_ORDER
MEAN, METADATA = load_results(RUN_CONFIG, plot_source_commit=SOURCE_COMMIT,
                              plot_source_sha256=SOURCE_SHA256)
SCALES = shared_scales(MEAN)
FIGURES = OUTPUT/'figures'
TITLE = 'DINOv2-T5 base: decoder-start visual cross-attention'
print('Measured questions:', METADATA['count'], 'Mean shape:', MEAN.shape)
''', '''# 7. Four layers, 2x2 panels; token attention Y and color use log scales.
from IPython.display import display, Image, FileLink
from dinov2_attention.plotting import plot_token_atlas
paths = plot_token_atlas(MEAN, CAMERA_ORDER, FIGURES, title=TITLE, scales=SCALES)
display(Image(filename=str(paths[0])))
''', '''# 8. Four layers, 2x2 panels; six unflipped 8x8 camera maps per horizontal row.
from dinov2_attention.plotting import plot_spatial_atlas
paths = plot_spatial_atlas(MEAN, CAMERA_ORDER, FIGURES, title=TITLE, scales=SCALES)
display(Image(filename=str(paths[0])))
''', '''# 9. Histogram: 384 averaged visual positions/layer, logarithmic count color.
from dinov2_attention.plotting import plot_layer_histogram
paths = plot_layer_histogram(MEAN, CAMERA_ORDER, FIGURES, title=TITLE, scales=SCALES)
display(Image(filename=str(paths[0])))
''', '''# 10. Download six PNG/PDF figures, metadata, and the measurement caption.
from dinov2_attention.plotting import make_figures_zip
ZIP = make_figures_zip(FIGURES, OUTPUT/'dinov2_base_attention_figures.zip', METADATA)
display(FileLink(str(ZIP)))
''']
    cells = [dict(cell_type='markdown',id='dino-intro',metadata={},source=markdown.splitlines(keepends=True))]
    for index, source in enumerate(sources,1):
        ast.parse(source)
        cells.append(dict(cell_type='code',id=f'dino-{index}',metadata={},
                          execution_count=None,outputs=[],source=source.splitlines(keepends=True)))
    return dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
                language_info=dict(name='python',version='3.11')),nbformat=4,nbformat_minor=5)


def build():
    path=ROOT/'notebooks/kaggle_dinov2_base_attention.ipynb'
    path.write_text(json.dumps(make_notebook(),ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(f'Updated {path.name}')


if __name__=='__main__':
    build()
