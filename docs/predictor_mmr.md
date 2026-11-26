# Predictor MMR test-only

Two independent notebooks run the frozen existing predictor checkpoints:

- `notebooks/kaggle_smolvlm_predictor_before_mmr_test.ipynb`
- `notebooks/kaggle_smolvlm_predictor_after2_mmr_test.ipynb`

Set `PREDICTOR_CHECKPOINT` to a local/Kaggle Input `predictor_best.pt`,
`PROJECTED_CACHE_INPUT` to the complete projected cache folder, then run all.
No HF predictor resolver, teacher targets, optimizer or RNG sidecars are needed.
Both 5-epoch and 10-epoch trained predictors are accepted when method and semantic identities match.
Notebooks pull the clean `smolvlm-attention` branch with `--ff-only` before importing.

Relevance is `exp(log_pred)` followed by sample-global min-max on all 384 visual tokens.
Flat probability rows become zero relevance. Selection uses lambda 0.9 and maximum signed
cosine similarity to already chosen states, L2-normalized with epsilon 1e-12.
The first pick maximizes relevance; ties prefer the lowest original visual index.
After 32 picks, sort indices into source order and gather original states.
Before-LLM states are native connector outputs; after2 states are outputs after the entire
block index 2 (layer 3). No preserve mass or camera quotas are applied.

The implementation follows the MMR structure in Nguyet branch commit
`5061b949299504c821994591b8bc1b65530bad04`,
`smolvlm_attention/posprune_ablation/decoder.py`, with predictor-derived,
probability-minmax relevance. This differs from native attention relevance.

Each notebook runs two independent GPU workers at fixed FP32 batch 64/GPU,
greedy beam 1 and 512 new tokens. Both shards first stress their longest prompts;
production starts only after both stress gates pass. Outputs are separate from Top-K and
preserve-mass runs. Fixed original-row schedules and checksummed batch commits support
evaluation resume; enable `RESUME` in the notebook for the same output.
Changed input/checkpoint/config/source/library identity is rejected.

The ZIP contains predictor weights, runtime config, logs, predictions and metrics,
not the frozen base weights, raw/projected caches or teacher targets.
No training recipe is changed by these notebooks. CPU fixtures do not establish CUDA
batch-64 VRAM safety, throughput or quality; those remain to be measured on Kaggle.
