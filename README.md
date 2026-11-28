"# Log_Metrics" 

SmolVLM cached checkpoint inference: see [Kaggle instructions](docs/smolvlm-inference.md)
and [notebook](notebooks/kaggle_smolvlm_checkpoint_inference.ipynb).

Two-GPU visual attention measurement: [instructions](docs/smolvlm-attention.md)
and [standalone Kaggle notebook](notebooks/kaggle_smolvlm_attention_distribution.ipynb).

Base same-layer combined visual attention, layers 0,2,...,28 (two GPUs, fixed batch 64):
[instructions](docs/smolvlm-combined-visual-attention.md) and
[Kaggle notebook](notebooks/kaggle_smolvlm_combined_visual_attention.ipynb).

Generated-token attention on 1,000 test questions (base only):
[instructions](docs/smolvlm-generated-attention.md) and
[standalone Kaggle notebook](notebooks/kaggle_smolvlm_generated_attention.ipynb).

DINOv2–T5 unpruned base, decoder-start cross-attention on all test questions:
[instructions](docs/dinov2-attention.md) and
[standalone Kaggle notebook](notebooks/kaggle_dinov2_base_attention.ipynb).

SmolVLM base causal perturbation and exact-phrase bbox localization on frozen test data:
[instructions](docs/smolvlm-probing.md) and
[standalone Kaggle notebook](notebooks/kaggle_smolvlm_causal_localization.ipynb).

Per-question prompt attention export, all 16,817 test questions and 30 layers (base, initial batch 32/GPU):
[instructions](docs/smolvlm-prompt-attention-export.md) and
[standalone Kaggle notebook](notebooks/kaggle_smolvlm_prompt_attention_export.ipynb).

Single-layer Top-32 answer-quality benchmark (two waves of eight configs, four sequential configs/GPU, fixed batch 64):
[instructions](docs/smolvlm-topk-benchmark.md) and
[standalone Kaggle notebook](notebooks/kaggle_smolvlm_topk_benchmark.ipynb).

Combined-attention Top-32 benchmark from existing projected cache (ten layers, five sequential configs/GPU):
[instructions](docs/smolvlm_combined_topk.md) and
[single Kaggle notebook](notebooks/kaggle_smolvlm_combined_attention_topk_benchmark.ipynb).
