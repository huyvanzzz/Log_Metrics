"""Build the standalone two-task SmolVLM base Kaggle notebook."""
import ast
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def setup_cell():
    header = textwrap.dedent('''\
        # 1. Push changes first. Pull is blocked while any attention worker runs.
        import os
        from pathlib import Path
        SOURCE_REPOSITORY = 'https://github.com/huyvanzzz/Log_Metrics.git'
        SOURCE_BRANCH = 'smolvlm-attention'
        SOURCE = Path('/kaggle/working/visualize-github-source')
        OUTPUT = Path('/kaggle/working/smolvlm_probing_outputs')
        RUN_MODE = 'auto'  # auto, measure, resume, plot
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
        identity = sync_source(SOURCE_REPOSITORY, SOURCE_BRANCH, SOURCE,
                               packages=('smolvlm_probing', 'smolvlm_inference', 'smolvlm_attention'))
        SOURCE_COMMIT, SOURCE_SHA256 = identity['commit'], identity['sha256']
        activate_source(SOURCE, packages=('smolvlm_probing', 'smolvlm_inference', 'smolvlm_attention'))
        from smolvlm_probing.runtime import resolve_run_mode
        BOOT_MODE = resolve_run_mode(OUTPUT, RUN_MODE)
        if BOOT_MODE != 'plot':
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary=tokenizers',
                            '-r', str(SOURCE/'smolvlm_probing/requirements.txt')], check=True)
            import torch, transformers, tokenizers
            if transformers.__version__ != '4.46.3' or tokenizers.__version__ != '0.20.4':
                raise RuntimeError('Restart the kernel after installing pinned dependencies, then rerun cell 1.')
        RUN_CONFIG = OUTPUT/'probing_run.json'
        print(dict(source=str(SOURCE), commit=SOURCE_COMMIT, output=str(OUTPUT), mode=BOOT_MODE,
                   model='SmolVLM-256M base', initial_batch_per_gpu=16))
        ''')
    return header + '\n' + helper + '\n' + tail


def make_notebook():
    intro = '''# SmolVLM base: causal perturbation and object localization

Enable Internet and a GPU (T4 x2 is supported). Attach the existing SmolVLM feature
cache. **Push the complete implementation and frozen data folder before using cell 1.**
Cell 1 clones/pulls Visualize, clears old package imports, and preserves outputs.

2,581 fixed test questions; greedy max 64 new tokens; FP32; initial batch 16/GPU,
CUDA OOM halves it. Top/Bottom/Random-32 (three Random repeats), ranking layers
1/7/13/19/25 (zero-based). Fixed generated answer is teacher-forced after zeroing
post-connector visual embeddings. Scores use geometric-mean token probabilities.

Localization uses both fixed partitions (749/652 targets), exact case-sensitive
literal phrases in actual generated output, global peak over 384 positions,
and inclusive bbox grid-center hits in six five-layer groups. Missing/repeated/
ambiguous/unalignable targets are audited. This is not segmentation accuracy.

Run new/partial measurements in order. **Redraw complete results: cell 1, then
cells 8–11 only**; these cells do not load a model or scan features. Only the few
raw camera images selected for overlays are downloaded, serially and cached.
Interrupting a wait cell leaves workers running; rerun that cell to reconnect.
Keep the whole output folder to resume. Changing measurement identity requires
a new OUTPUT; plotting-only changes do not require new measurement.
'''
    sources = [setup_cell(), textwrap.dedent('''\
        # 2. Existing mounted cache/prepared paths are optional; automatic discovery is the default.
        from smolvlm_probing.runtime import prepare_run, launch_stage, wait_stage, load_results
        from smolvlm_probing.state import atomic_json
        CACHE_INPUT = ''  # Set only if more than one matching cache is mounted.
        PREPARED_INPUT = None  # Existing directory with manifest/train/val/test, otherwise pinned HF files.
        print(dict(mode=BOOT_MODE, cache=CACHE_INPUT or 'auto', prepared=PREPARED_INPUT or 'pinned HF'))
        '''), textwrap.dedent('''\
        # 3. Validate fixed dataset/cache once, reuse matching receipt, and prepare all 2,581 prompts.
        if BOOT_MODE != 'plot':
            RUN_CONFIG = prepare_run(OUTPUT, mode=BOOT_MODE, cache_input_dir=CACHE_INPUT,
                                     prepared_dir=PREPARED_INPUT, source_commit=SOURCE_COMMIT,
                                     source_sha256=SOURCE_SHA256)
        print(RUN_CONFIG)
        '''), textwrap.dedent('''\
        # 4. Strict loading, native eager/scoring/zero-ablation and cached-decode preflight.
        if BOOT_MODE != 'plot':
            launch_stage(RUN_CONFIG, 'preflight')
            wait_stage(RUN_CONFIG, 'preflight')
        '''), textwrap.dedent('''\
        # 5. Generate once, save exact original IDs, then collect baseline ranks and target maps.
        if BOOT_MODE != 'plot':
            launch_stage(RUN_CONFIG, 'baseline')
            wait_stage(RUN_CONFIG, 'baseline')
        '''), textwrap.dedent('''\
        # 6. Causal: 26 perturbation conditions per nonempty valid answer; no new generation.
        if BOOT_MODE != 'plot':
            launch_stage(RUN_CONFIG, 'causal')
            wait_stage(RUN_CONFIG, 'causal')
        '''), textwrap.dedent('''\
        # 7. CPU localization over saved target maps. Same eligible denominator for all six groups.
        if BOOT_MODE != 'plot':
            launch_stage(RUN_CONFIG, 'localization')
            wait_stage(RUN_CONFIG, 'localization')
        '''), textwrap.dedent('''\
        # 8. Load complete checked results and draw five causal scatter figures.
        from smolvlm_probing.runtime import load_results
        from smolvlm_probing.plotting import plot_causal, plot_localization, select_examples
        from smolvlm_probing.plotting import fetch_example_images, plot_examples, bundle_figures
        from smolvlm_probing.state import atomic_json
        RESULTS = load_results(RUN_CONFIG, plot_source_commit=SOURCE_COMMIT, plot_source_sha256=SOURCE_SHA256)
        FIGURES = OUTPUT/'figures'
        FIGURE_PATHS = plot_causal(RESULTS['causal'], FIGURES, metadata=RESULTS['metadata'])
        print(RESULTS['metadata'])
        '''), textwrap.dedent('''\
        # 9. Accuracy by layer group, both coordinate partitions, with coverage and counts.
        FIGURE_PATHS += plot_localization(RESULTS['summary'], FIGURES, metadata=RESULTS['metadata'])
        print(RESULTS['summary'])
        '''), textwrap.dedent('''\
        # 10. Post-hoc examples from each partition's best layer group; fetch ONLY these image paths.
        EXAMPLES = select_examples(RESULTS['localization'], RESULTS['summary'], seed=42)
        atomic_json(OUTPUT/'selected_examples.json', EXAMPLES)
        IMAGE_PATHS, MISSING_IMAGES = fetch_example_images(EXAMPLES, OUTPUT/'illustration_images')
        atomic_json(OUTPUT/'missing_images.json', MISSING_IMAGES)
        FIGURE_PATHS += plot_examples(EXAMPLES, IMAGE_PATHS, FIGURES, metadata=RESULTS['metadata'])
        print(dict(missing_images=MISSING_IMAGES, figures=[str(p) for p in FIGURE_PATHS]))
        '''), textwrap.dedent('''\
        # 11. ZIP figures/captions/summary/audit/provenance; no checkpoint, features or source-image archive.
        extras = [OUTPUT/name for name in ('summary.json', 'causal_summary.json', 'figure_metadata.json',
                  'localization_audit.json', 'selected_examples.json', 'missing_images.json')]
        extras += list(FIGURES.glob('*_caption.txt')) + list(FIGURES.glob('*_metadata.json'))
        ZIP = bundle_figures(OUTPUT, FIGURE_PATHS, RESULTS['metadata'], extra_paths=extras)
        from IPython.display import FileLink, display
        display(FileLink(str(ZIP)))
        ''')]
    cells = [dict(cell_type='markdown', id='probing-intro', metadata={}, source=intro.splitlines(keepends=True))]
    for index, source in enumerate(sources, 1):
        ast.parse(source)
        cells.append(dict(cell_type='code', id=f'probing-{index}', metadata={}, execution_count=None,
                          outputs=[], source=source.splitlines(keepends=True)))
    return dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python', version='3.11')), nbformat=4, nbformat_minor=5)


def build():
    path = ROOT/'notebooks/kaggle_smolvlm_causal_localization.ipynb'
    path.write_text(json.dumps(make_notebook(), ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    print(f'Built {path}')


if __name__ == '__main__':
    build()
