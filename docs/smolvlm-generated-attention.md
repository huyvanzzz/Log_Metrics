# Generated-token attention on Kaggle

Upload `notebooks/kaggle_smolvlm_generated_attention.ipynb`, enable Internet and a GPU
(one GPU or T4 ×2), attach `huyhuy12334/cache-smolvlm`, then Run All. Cell 1 clones
the entire `huyvanzzz/Log_Metrics` repository, or runs `git pull --ff-only` when the
checkout already exists. Push local changes first. `SOURCE_BRANCH` defaults to
`smolvlm-attention`; change it if you push to another branch. Dependencies and
checkpoints are downloaded automatically. Only `huyvanzzz/smolvlm256M_base` is loaded,
at the checkpoint revision pinned in `smolvlm_attention/config.py`.

The notebook selects 1,000 test **questions**, not unique scenes, without replacement.
Selection sorts sample IDs before PCG64 seed 42. Each question retains its six cached
cameras in base prompt order. Prepared data comes from the pinned
`huyvanzzz/prepared_data` revision. Ground-truth answers are not used.

Greedy generation uses one beam and at most 32 new tokens. Native EOS stops generation.
EOS and subsequent padding are excluded from attention queries. Answers ending at
the token limit retain their final content token. Empty answers are recorded, not
replaced, and excluded from the averaging denominator. Counts are printed explicitly.

After generation, the exact content-token sequence is replayed with native SDPA.
Attention is reconstructed only for generated-token queries, with RoPE, grouped-query
attention, padding and future-key masks applied. Keys at the query's own position are
included. No full all-layer attention matrices are retained in production measurement.
A small native eager oracle checks parity before the run.

Image keys are the 384 prompt image positions. Prompt text includes all other prompt
tokens, including camera labels and chat delimiters. Generated keys include content
tokens up to and including the query. Softmax covers every valid causal key; there is
no image-only renormalization. The three group masses sum to one.

For every question, mean heads then mean generated content tokens. Then average
nonempty questions equally. Long answers and scenes with repeated questions are not
assigned additional weight beyond their question records. The three-line figure
shows mean ±1 SD across these per-question means, clipped to [0,1] for display. The
histogram contains **384 already-averaged values per layer**, using 100 shared linear
bins, magma colors and logarithmic counts. It does not pool 384,000 observations.

Outputs are under `/kaggle/working/generated_attention_outputs/`:

- `selected_samples.json` and `generated_run.json`: selection and run identity.
- `data/shard_*/results/*.npz`: atomic per-question results and generation metadata.
- `logs/shard_*.json` and `.log`: live stages, counts and detailed worker output.
- `summary.npz`: mean visual attention, group mean/SD and sample counts.
- `responses.jsonl`: questions, generated/content token IDs, response and stop reason.
- `figures/base/`: the two figures, each PNG 300 dpi and PDF.
- `generated_attention_figures.zip`: four figure files, metadata and captions.

Cell 3 shows cache-validation and prompt progress. A valid earlier cache receipt can
be reused after input identity checks; otherwise the full cache is scanned with a
byte progress bar. Cell 5 shows GPU shard and total progress. Two GPUs run the same
base model on disjoint 500-question shards; one GPU runs the full list. Batch starts
at 16 per GPU without automatic tuning. OOM halves the batch without committing
failed measurements. Interrupting the wait cell leaves background workers running.

Keep the output directory to resume; rerun preparation and launch after workers have
stopped. Changes to checkpoint, selection, source or measurement settings reject
old results. Keep the same GPU/shard count when resuming an existing experiment.
Samples are saved atomically after successful measurement; completed
IDs are not measured again. A new Kaggle session must retain or restore these files
and the referenced inputs. For a changed experiment, choose a new output directory.

Both notebooks use `/kaggle/working/visualize-github-source`, while outputs stay
outside that checkout. After pushing code, rerun cell 1 to update all source and
clear old package imports. Finish workers first; setup refuses active workers or
tracked local checkout edits. To redraw saved results, run **cell 1, then cells
6–9 only**, keeping the original run config and results. Do not rerun preparation
just to update plots, because a changed source fingerprint rejects old measurements.

Notebook setup is maintained in `scripts/kaggle_source_setup.py`. After changing
that setup helper or notebook templates, rebuild both notebooks with:

```powershell
.\.venv\Scripts\python.exe scripts/build_attention_notebooks.py
```

Ordinary measurement/plotting source changes need only a Git push and cell 1;
they no longer require rebuilding or reuploading the notebook. The older builder
entrypoint remains an alias for rebuilding both notebooks.

Reference: [What's in the Image?](https://arxiv.org/html/2411.17491v1).
Figure styling follows the supplied examples and the local `figures4papers` skill;
the actual curves and ranges are determined by SmolVLM measurements.

CPU fixtures verify attention against native eager and KV-cache decode, aggregation,
resume, OOM handling, exports and first-cell Git update behavior. Full checkpoint
generation and Kaggle throughput require a real GPU run and are not claimed from
local CPU tests.
