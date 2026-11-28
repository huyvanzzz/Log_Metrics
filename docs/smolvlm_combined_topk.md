# Combined-attention Top-32 benchmark on Kaggle

Open `notebooks/kaggle_smolvlm_combined_attention_topk_benchmark.ipynb`.
Push the new source to the `smolvlm-attention` branch before running its first cell.
Enable Internet and **T4 x2**, attach the **completed projected cache** from the
predictor pipeline (`projected.npy`, `index.json`, `manifest.json`), then Run All.
This notebook never builds projected cache or downloads raw features/images.

`PROJECTED_CACHE_INPUT=''` searches `/kaggle/input` and
`/kaggle/working/smolvlm_projected_visual_cache`. Set the exact cache directory in
cell 2 if detection finds none or more than one. `PREPARED_INPUT=None` and
`CHECKPOINT_INPUT=None` download the existing pinned prepared data and base checkpoint.
The first cell pulls source and installs pinned inference/scoring dependencies,
preserving Kaggle's CUDA Torch. Java is required for METEOR and is installed if missing.

## What runs

Only `huyvanzzz/smolvlm256M_base`, revision
`f9d4e9a4d704078f9e7f39f061d13602cf251c78`, and all 16,817 prepared test questions.
Native prompt expansion, camera order and processor policy remain unchanged.

For each question and selected layer, the collector averages heads **before**
computing:

`S(q,v) = A_mean[q,v] + sum_t A_mean[q,t] * A_mean[t,v]`.

`q` is the last valid native prompt position, including the assistant suffix;
`t` includes **all valid non-visual positions except q**, including camera names,
separators and special tokens. Padding is excluded. Both attention rows come from
the same layer and use native causal masking and softmax across all allowed keys,
including self. No question-mask restriction, image-only renormalization, rollout
or value scaling is applied. This is the agreed combined score, not a causal
contribution estimate. Answers remain metadata and never enter the prompt.

Stage A splits questions into 8,409 / 8,408 rows across GPUs, saves FP32 per-sample
scores in `attention_export/attention.npy` with shape **[16817,10,384]**, then merges
the shards in original prepared-test order. The layer axis is explicitly
**[10,15,16,17,21,22,23,25,26,27]**; axis 0 is layer 10, not layer 0.
Camera order is Front, Front-Left, Front-Right, Back, Back-Left, Back-Right;
each camera occupies 64 visual indices in an unchanged 8×8 orientation.

Stage B selects stable global Top-32 separately for each sample/layer. Ties prefer
lower visual indices, and selected tokens return to prompt order. Other 352 image
embeddings are physically removed before LLM layer 0. All non-visual tokens remain;
original RoPE positions are retained while physical KV slots are compacted.

| GPU | Full-test configuration order |
| --- | --- |
| 0 | 10 → 15 → 16 → 17 → 21 |
| 1 | 22 → 23 → 25 → 26 → 27 |

Each GPU completes one configuration before starting the next. One model is loaded
per GPU per stage; there is no base/full-token benchmark configuration. The small
all-keep native parity probe is a preflight check only.

Both stages use **batch 64 fixed**, FP32, no TF32/autocast. Last batches are 25/24
for attention and 49 for each benchmark. Generation uses the shared-weight native
SDPA decoder, KV cache, greedy decoding and at most **512 new tokens**, stopping
at EOS normally. The upstream `num_logits_to_keep=1` generation optimization is
preserved. CUDA OOM stops the worker; batch size/precision/prompt length are never
silently changed.

## Progress, resume and outputs

Output defaults to `/kaggle/working/smolvlm_combined_topk_outputs/`.
Attention batches flush/fsync their private memmap before publishing checksummed
progress. Predictions are committed atomically by batch. Restarting repeats only
uncommitted work; committed content is verified. tqdm covers validation/preparation,
measurement, benchmark configurations, merge and scoring. Worker status and errors
are in `logs/`; both wait cells may be interrupted and rerun without stopping workers.
Wait for workers to finish before pulling source.

- `RUN_MODE='auto'`: new run or compatible resume; skip completed phases.
- `measure`: refuse an existing run.
- `resume`: require an existing run and compatible measurement/generation identity.
- `score`: complete predictions only; run cells **1,2,9** without GPU, checkpoint or
  projected payload. Cell 10 additionally needs the attention export for packaging.

Save the **entire output folder** to resume a partial run across sessions. A completed
ZIP contains portable results and excludes partial work journals; it is not a partial
resume snapshot. Measurement, generation and scoring identities are separate:
editing scoring or ZIP does not require remeasuring attention or generating again.
Incompatible identities stop before overwriting existing results.

After extracting a completed ZIP elsewhere, rescore its portable benchmark directly:

```python
from smolvlm_combined_topk.benchmark import score_results
summary = score_results('/path/to/extracted/benchmark/benchmark_run.json')
```

This API needs the completed prediction files and scoring dependencies, without
checkpoint, projected cache, attention payload or partial batch journals.

Scoring begins after all ten configurations complete: corpus BLEU-1–4, METEOR,
ROUGE-L and CIDEr using the previous direct-string pipeline. No per-batch metric
averaging, fallback scorer or comparison with an old baseline is introduced.
Raw metrics, diagnostics, `benchmark/summary.json` and `benchmark/summary.csv` are saved.

The uncompressed ZIP contains attention/mapping/metadata/hashes, Top-32 selections,
predictions, metrics and summaries. It excludes checkpoint, projected/raw cache and
work shards. Separate links expose attention and summaries. Extract before using
`numpy.load(..., mmap_mode='r', allow_pickle=False)` or the NumPy-only
`smolvlm_combined_topk.store.open_attention` reader.

The attention payload alone is about **246.34 MiB**; keep space for its shards,
final export and ZIP as well as prediction journals. No shards are automatically deleted.

CPU fixtures and regression tests verify the new path. Real checkpoint numerical
preflight runs on **both GPUs before each stage**. Batch-64 VRAM, wall time and
full Kaggle execution still require validation on the actual attached artifacts.
