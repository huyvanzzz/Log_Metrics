# MAP-style predictor for SmolVLM

This experiment trains a small predictor before the language model. It uses the
existing base checkpoint and teacher targets from **index 22, layer 23**. It does
not implement MAP's teacher selection or diversity selection, and is not a
reproduction of the paper's complete recipe.

The two-branch MLP, Q/K scoring, RoPE, visual softmax and question-query average
follow the idea in [MAP, Cross-Modal Attention Distillation](https://arxiv.org/html/2608.06411v2#S4.SS2).
The fixed teacher, dimensions and optimization recipe below are this experiment's
choices. Teacher queries use the last native generation-prefix token; predictor
queries use all question tokens. These policies intentionally differ.

## Computation

Training/validation:

```text
FP16 raw cache --> FP32 frozen connector (one-time cache build)
projected FP32 cache --FP32 H2D-----------------------------> [B,384,576]
question token IDs ----------> frozen native embedding -----> [B,Lq,576]
                                      |
                         trainable MAP-style predictor
                                      |
                           log probabilities [B,384]
                                      |
                     KL(teacher index22 || predictor)
```

Each modality has `Linear(576,128) -> GELU -> Linear(128,128)`;
separate bias-free projections produce Q/K of dimension 64. Native RoPE uses
original full-prompt positions, independently for question and visual lengths.
There are **197,120 trainable parameters**. No additional camera embeddings,
dropout, V/O projections, residual prediction or online teacher are used.

The predictor applies visual softmax **per question query**, then averages valid
queries. Computation stays in log space. Loss sums teacher-to-predictor KL over
the 384 visual positions and averages samples, without division by 384 or
question length. Targets are checked but never renormalized/clamped. This uses
[PyTorch KL with `log_target=True`](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.kl_div.html).

Frozen extraction uses `no_grad`, not `inference_mode`, so predictor parameter
gradients remain valid. Training guards reject calls to decoder blocks, full
native model, vision encoder and LM head. Only native embedding lookups run once the projected cache is active; the connector runs only during cache construction.

After training, predictor best selects stable global Top-32 **before block 0**.
Ties prefer smaller visual indices; chosen indices then return to source order.
Generation gathers the original embeddings, keeps all valid non-image tokens
and original positions, and uses the existing greedy compaction implementation.
Selection stays fixed during decoding. Test does not read teacher targets.

## Inputs

Use `notebooks/kaggle_smolvlm_map_predictor_train_test.ipynb` on two Kaggle GPUs.
Set `TARGETS_INPUT` to the unpacked complete portable export:

```text
TARGETS_INPUT/
  train/log_probs.npy
  train/samples.jsonl
  train/manifest.json
  val/log_probs.npy
  val/samples.jsonl
  val/manifest.json
```

Partial exporter shards are not accepted. Targets and vision cache are read-only
memmaps directly from Input. No hidden-state cache or copied vision cache is
created. Target rows join by `(split,sample_id)`; prompt/question hashes, prepared
row, cache key/index, checkpoint identity and normalization must all match.
Producer worker/prefetch/scheduling settings are not trainer compatibility keys.

Pinned base: `huyvanzzz/smolvlm256M_base/finetune_best.pt`, revision
`f9d4e9a4d704078f9e7f39f061d13602cf251c78`. Native model revision:
`7e3e67edbbed1bf9888184d9df282b700a323964`. Prepared data:
`huyvanzzz/prepared_data`; cache: `huyhuy12334/cache-smolvlm/versions/1`.
Camera order is **Front, Front-Left, Front-Right, Back, Back-Left, Back-Right**.
The reverse-camera checkpoint is rejected.

Parent validates full hashes/finiteness once when preparing a run; workers use
file receipts, lazy offsets/memmaps and batch checks. Each CPU batch preallocates projected FP32 features. Compact native prompt/token mappings are prepared once, and workers use lazy read-only memmaps without a tokenizer or per-sample prepared/target JSON reads.

## Training and resume

The separate config is `configs/smolvlm_map_predictor_index22.json`:

| Setting | Value |
|---|---|
| Train | 2 GPUs, predictor-only DDP, batch 32/GPU, accumulation 1 |
| Precision | FP32 compute, autocast/TF32 off, no GradScaler |
| Duration | 5 epochs continuously |
| Optimizer | AdamW, LR 1e-4, weight decay .05, default betas/eps |
| Scheduler | ExponentialLR gamma .9 after completed epoch |
| Clip / seed | 1.0 / 42 |
| Loader | 2 spawned persistent workers/GPU, pin memory, prefetch 1 |
| Validation | 32/GPU, exact nonduplicated shards, sample-mean KL |
| QA test | 64/GPU, two independent processes, greedy, max 512 new tokens |

Selective lifecycle reference:
`efficient_vlm_ad/smolvlm/pipeline.py` at commit
`4d7dbd238c7960d305c882b75220fb95beb37644` in the source SmolVLM repository.
There is no runtime dependency on another local checkout. This implementation
retains seeded shuffle, Accelerate seedable sampler/set_epoch, even_batches and
drop_last=False, accumulation-v2 and successful update counters. See
[Accelerate's batching semantics](https://huggingface.co/docs/accelerate/v1.15.0/en/package_reference/torch_wrappers).
Train tail samples may repeat to fill even batches; validation/test never do.
Post-prepare RNG seeds are `42 + rank` for fresh training.

Latest and best contain only predictor weights plus optimizer/scheduler/counters
and identity. Each has an accompanying `.pt.rng/` directory with immutable save
generations and Python/NumPy/Torch/CUDA RNG for each rank. The `.pt` is atomically
published after sidecars. Training resume requires matching code/library versions, recipe/world size,
sidecars and counters; RNG is restored after initialization and iterator seek.
Inference requires only predictor weights and matching base/input identity.

Checkpoints are saved at epoch boundaries. If interrupted mid-epoch, resume the
last saved completed epoch. Do not modify `epochs`, precision or other recipe
fields to resume. Keep **both latest and best, including `.rng`**, when moving
sessions. Notebook `RESUME_INPUT` copies these small predictor artifacts and
revalidates current Input paths; it does not copy base/cache/targets. An existing
output is never silently treated as a fresh run.

## CLI

```bash
python -m pip install -r smolvlm_map_predictor/requirements.txt
python -m smolvlm_map_predictor validate-inputs \
  --targets-input /kaggle/input/YOUR-TARGETS/export \
  --cache-input /kaggle/input/YOUR-CACHE/cache \
  --output /kaggle/working/smolvlm_map_predictor_index22_optimized_outputs
python -m smolvlm_map_predictor diagnose --run /kaggle/working/smolvlm_map_predictor_index22_optimized_outputs/run.json
python -m smolvlm_map_predictor train --run /kaggle/working/smolvlm_map_predictor_index22_optimized_outputs/run.json
```

Resume: rerun `validate-inputs` with `--resume` and the same contents/recipe, run
diagnose, then `train --run RUN --resume PATH/predictor_latest.pt`. Read-only
verification: `verify-resume-checkpoint --run RUN --predictor PATH`.
Evaluate: `evaluate --run RUN --predictor PATH/predictor_best.pt`.
Bundle: `bundle --run RUN`.

The notebook checks Java/METEOR before training and runs real two-rank training
stress before launching epochs. Evaluation uses a fresh shared stress gate;
both independent GPU workers must pass their longest-batch check before either
commits test results. Errors stop peers and retain committed outputs. Batch size,
workers, precision and prefetch are never automatically reduced.

## Outputs and verification

`debug/train_progress.jsonl` has flat timestamp/event records, microbatch loss,
LR, counters, elapsed/rate/ETA and CUDA memory. Epoch train summaries average
microbatch losses across ranks, matching the source lifecycle; validation is
sample-weighted. Checkpoint events follow successful
saves. Resource logs distinguish available RAM/cgroup, shared memory and OOM
events where the host exposes them. Page-cache usage alone is not an OOM claim.

The archive includes predictor checkpoints/RNG, config, logs, predictions and
metrics; it excludes base weights, vision cache and targets. Evaluation batch
commits are identity/checksum checked for resume; coverage precedes existing
BLEU/METEOR/ROUGE-L/CIDEr scoring. No latency/FLOPs benchmark or automatic upload.

Offline verification uses real small native models, local tokenizer/cache/teacher
exports and CPU/Gloo. It covers RoPE/KL/gradients, target joins, frozen-backbone
guards, spawned loaders, DDP gradients, resume, greedy generation, metrics and
bundling. No large pretrained weights or dataset are downloaded for these tests.

**CUDA batch 32 train / 64 test, pinned memory, VRAM, throughput, QA quality and
10-epoch session duration remain unverified until the notebook runs on Kaggle.**

Verification on 2026-11-24: **37 predictor tests passed** (86.05 s), including
real CPU/Gloo training/resume and complete tiny-model inference/metrics/bundle;
**250 related regression tests passed** (144.38 s). Environment: Torch
2.15.0+cpu, Transformers 4.46.3, Tokenizers 0.20.4, Accelerate 1.15.0.
Regression warnings were the three expected CPU-only pinned-memory warnings.
Python/notebook syntax and whitespace checks passed. Existing tracked files
were unchanged. No production extraction/training/test, commit or push ran.

## Optimized pipeline

Both notebooks now default to new `*_optimized_outputs` directories. Start a new run:
old predictor checkpoints are intentionally not migrated across this pipeline identity.
Total duration is **5 epochs**; latest/best still carry rank-specific RNG sidecars.

Set `PROJECTED_CACHE_INPUT` to a complete unpacked projected cache, or leave it
empty and keep `PROJECTED_CACHE_DIR=/kaggle/working/smolvlm_projected_visual_cache`.
The first session builds it once from the unchanged raw FP16 cache, with two GPU
connector workers and batch 32/GPU. It is FP32 `[4072,6,64,576]`, approximately
3.36 GiB. A subsequent Input reuse needs no raw cache download or scan. Do not
point PROJECTED_CACHE_INPUT at teacher targets or the original raw cache.
There is no automatic upload; retain this cache separately for future sessions.

A single persistent spawn pool per rank handles both training and validation,
with two workers, pinning and prefetch 1. Training uses native Accelerate sampler
classes without double sharding; validation/test remain exact nonduplicated shards.
Projected batch 32 is 27 MiB. Finite checking uses a reusable per-image-group
scratch buffer instead of batch-sized Torch intermediates.

Notebook output is a compact progress widget. Detailed JSON, child stdout/stderr,
diagnose results and failures stay under `logs/`, `debug/`, and `diagnose.json`.
Diagnose is a separate disposable predictor: 3 warmup and 20 measured updates,
with startup, batch wait, CPU collation/finite-copy, transfer, frozen computation
and predictor-update timings. It never produces training checkpoints. GPU events
measure intervals, not isolated kernel costs or proof of H2D overlap. Production
training receives no extra timing synchronizations.

The predictor archive excludes raw/projected caches, teacher targets and base
weights. No precision/batch/worker sweep, production training, or automatic
publishing is performed by local verification. CUDA peak memory and speed must
still be checked by running the notebook's real stress tests.
