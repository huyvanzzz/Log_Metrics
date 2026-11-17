# Kaggle: sample-averaged SmolVLM visual attention

Upload `notebooks/kaggle_smolvlm_attention_distribution.ipynb` to Kaggle. Enable
Internet and **two GPUs** (for example T4 x2); attach
`huyhuy12334/cache-smolvlm` version 1 as Input. Run All. The notebook bundles the
measurement source and pins both HF best checkpoints and prepared data; it does
not require publishing local changes to GitHub or cloning figures4papers in Kaggle.
Keep Kaggle's CUDA Torch. If Transformers was already imported before installing
the pinned dependencies, restart the kernel and Run All.

Base launches on physical GPU 0 and reverse on GPU 1. Both launch cells return
immediately; the next cell waits and shows progress. Workers read the same cache
without copying it and independently run the entire test set. Inspect
`attention_outputs/logs/base.log` or `reverse.log` on failure. Each worker verifies
native attention parity first, selects an FP32 batch size on long prompts, and
reduces it on CUDA OOM without skipping samples. A snapshot saved every 100
successful batches resumes only when source, artifacts and sample IDs match.
Interrupting the wait cell leaves workers running; rerun that cell to reconnect.
Do not rerun the setup/data cells while workers are active.

The quantity is the softmax attention from the final valid prompt position to
384 image-token positions, averaged over heads, then equally over all 16,817 test
records. The softmax denominator includes all valid text and visual keys. Records
sharing images remain distinct questions. No answer text, image extraction,
generation, beam search, training or KV cache is used. Base best is epoch 2;
reverse best is epoch 1, so this compares two fine-tuned checkpoints rather than
isolating camera order on identical weights.

Each model has three plotting cells and PNG/PDF exports:

- Token atlas: layers 0,2,...,28 in a 5-by-3 grid; 384 token bars per layer.
- Spatial atlas: the same 15 layers; six unflipped 8-by-8 maps in prompt order
  on one horizontal row within every panel.
- Histogram: all 30 layers on X, attention value on Y. Counts describe the 384
  **already averaged token positions** per layer, not individual samples/heads.

Color/axis limits and histogram bins are shared across models. Re-run plotting
cells without remeasuring. Results and metadata live in `attention_outputs/data/`;
figures in `attention_outputs/figures/base/` and `figures/reverse/`. The final cell
creates `attention_outputs/attention_figures.zip` containing both PNG/PDF folders
and measurement metadata, then displays a download link. Weights and cache are
excluded. The plotting style follows the local figures4papers reference (white
background, sans-serif typography, vector text), with the requested viridis and
magma colormaps; no third-party plot helpers or LaTeX installation are needed.

Token-atlas Y axes and viridis attention colors use a shared logarithmic scale
across all layers and both models. Attention values remain raw probabilities.
Zero has no bar and is shown as gray in spatial maps. Histogram attention bins
remain linear; only histogram counts use logarithmic colors.

To redraw an existing Kaggle run, update `smolvlm_attention/plotting.py` in the
actual source folder, reload that module with `importlib.reload`, then rerun
cells 7–14. Preserve the original `RUN_CONFIG` and output files; do not rerun
bootstrap or measurement cells merely to change figure scales.

Local verification: `python -m pytest -q tests backend/test_parser.py` after
installing Torch, pytest and `smolvlm_attention/requirements.txt` in a virtual
environment. Synthetic CPU tests establish collector/order/batch parity and
output correctness. They do not establish full-checkpoint GPU parity, Kaggle
VRAM, throughput or completion; the notebook reports those at runtime.
