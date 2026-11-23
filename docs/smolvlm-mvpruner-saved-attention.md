# MVPruner: prune index2, saved score index22

Open `notebooks/kaggle_smolvlm_mvpruner_saved_attention_test.ipynb` on Kaggle. Enable Internet and two GPUs. Set `ATTENTION_INPUT` to the extracted complete base attention export folder (containing attention.npy, metadata.json, samples.jsonl and manifest.json). Attach the existing cache `huyhuy12334/cache-smolvlm`; prepared data and the normal Front-to-Back best checkpoint use the same pinned revisions as the online MVPruner notebook. No publishing credentials are needed.

The notebook runs two independent GPU processes with disjoint test rows, FP32, TF32 disabled, batch64 per GPU, greedy, maximum512 new tokens, repetition penalty1. Stage1 remains384?196. After decoder index2 (third layer), Stage2 uses scores from export index22 (23rd layer), gathers the196 scores by Stage1 original visual indices and applies the unchanged camera IRA/quota/per-camera Top-K selector to retain32. It does not renormalize the196 scores. Hidden states are outputs of the current third layer, not layer23. Native attention/Q/K/V remain model computations; no online scoring hooks or extra base pass run.

The export measures mean-head attention from the last native prompt token (`:` in `Assistant:`) on the full384-token base input. This differs from online MVPruner mean question-token attention after Stage1. This is a score-source ablation, not attention distillation or training.

Input exports are read-only memmaps and aligned by sample identity, prompt IDs and cache identity. Checkpoint SHA, revisions, camera order, axes and ordered layer IDs must match. Missing or invalid exports stop the run; there is no fallback/extraction. The existing normal producer format is accepted without rewriting it.

Each GPU first runs normal and forced512-step stress on its longest64 unique shard prompts. A parent barrier requires both stress probes to pass before either starts its test shard. Rows are partitioned by position modulo2 (8409/8408); the parent verifies per-shard and full coverage, merges by eval_id, then computes metrics once on all16817 predictions, including both final partial batches. Progress/debug/logs remain under rank_0 and rank_1. A worker failure stops the sibling; no partial metrics are produced. There is no OOM retry or precision/batch adjustment. Metrics retain the current unnormalized source caption-scoring protocol. Output is `/kaggle/working/smolvlm_mvpruner_prune2_score22_outputs`; choose a new directory if a previous run already wrote metadata/predictions. Old online configurations/notebook are unchanged.

Artifacts: metadata (checkpoint/export identities, score/query policy, precision), first-batch debug (source indices, quotas and cache/layer lengths), predictions, progress, metrics and ZIP. No weights/cache/export tensor is packaged. Timing includes inference and reading the precomputed scores, excluding creation of the attention export. It is not end-to-end online pruning latency.

CLI after the common immutable-cache validation step:

```sh
python -m smolvlm_inference.saved_attention_run --cache CACHE --prepared PREPARED --receipt RECEIPT --checkpoint CHECKPOINT --attention-input EXPORT --config configs/mvpruner_prune_index_2_score_index_22.json --output OUTPUT --gpus 2
```

Offline verification uses tiny native models and synthetic caches/exports; real checkpoint batch64 fit, GPU performance and QA quality require Kaggle validation. No large data/model download or full production evaluation is part of local verification.

## Local verification

95 focused tests passed in13.16s (saved-score tests plus collector/export/Top-K/online MVPruner regressions). Tests include synthetic384?196?32 stress/inference with a final partial batch, frozen-selection cache parity, unchanged prefix block outputs/weights/RNG, forbidden online scoring hooks, sample alignment and export identity failures. Real Java/caption metric preflight passed. Python/notebook cells and fixed JSON configuration validate. No real checkpoint GPU run or full16817-row evaluation was performed locally.
