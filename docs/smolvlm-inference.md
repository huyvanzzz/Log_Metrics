# SmolVLM best checkpoint on Kaggle

Open `notebooks/kaggle_smolvlm_checkpoint_inference.ipynb` in Kaggle, enable a GPU and Internet, then Run All. Attach public dataset `huyhuy12334/cache-smolvlm` as Input when possible. The notebook uses kagglehub to resolve the dataset without making a second cache copy in working. `CACHE_INPUT_DIR` can point directly to the mounted directory containing the cache files.

Defaults:

- Best checkpoint: `huyvanzzz/smolvlm256M_base`, revision `f9d4e9a4d704078f9e7f39f061d13602cf251c78`, file `finetune_best.pt` only.
- Prepared data: `huyvanzzz/prepared_data`, revision `d2a6914f681dbb9148c4ac11e6fa92452010506e`.
- Camera order: Front, Front-Left, Front-Right, Back, Back-Left, Back-Right.
- FP32, one GPU, batch 1, native text SDPA; beam 3, maximum 512 new tokens, length penalty 1.0 and early stopping.

The checkpoint contains the full frozen vision encoder, connector, language model and head. The loader constructs the pinned native architecture and strictly restores all weights; it never falls back to pretrained weights. Optimizer, scheduler and `.rng` sidecars are not required. Only load checkpoints from the trusted repository above: the original PyTorch checkpoint format contains Python metadata.

The feature cache remains pre-connector FP16 `[N,6,1024,768]`; the pretrained/finetuned connector produces 64 tokens per camera, 384 total. Prepared sample keys are hashes of the canonical six camera paths, not row numbers. Startup validates processor/model identity, prepared hashes/counts, index coverage/hash, manifest fingerprint, geometry, payload size and finiteness in bounded chunks. The full cache is memory-mapped read-only; each inference reads one sample and casts it to FP32. Validation can take time on the approximately 35.8 GiB cache. It runs once per session creation, not for every prediction.

After loading, change `SPLIT` and `SAMPLE_INDEX` in the final cell and run only that cell. `session.predict()` returns sample ID, question, ground truth, prediction and camera order. No source images, extraction, training, dataset publishing or attention collection are performed. The dashboard is independent and unchanged.

## Source and verification

Selected model signature, native fusion/generation and chat/collation logic were ported from `huy-smolvlm-256m-no-pruner` commit `4d7dbd238c7960d305c882b75220fb95beb37644`. Cache validation retains the original identity and key format without importing the training package. Dependencies are isolated in `smolvlm_inference/requirements.txt`; use the Kaggle CUDA Torch installation. Global `pip check` for unrelated preinstalled applications is not a gate.

Run `python -m pytest -q tests backend/test_parser.py` in an environment with the inference requirements and pytest. Offline tests use a tiny real Idefics3 model and tokenizer, synthetic read-only cache, strict checkpoint validation and native generation. CPU fixture results do not establish Kaggle VRAM or successful loading of the full checkpoint. Confirm those by running the notebook and checking its checkpoint metadata/camera-order report and generated answer.

Implementation verification: 25 tests passed, including the dashboard regression and a real tiny-model checkpoint → cache → session prediction. A separate metadata-only inspection of the pinned HF best checkpoint confirmed canonical camera order, completed epoch 2 and complete architecture-signature equality with the pinned processor/config/generation config. No full checkpoint weights or feature dataset were downloaded for local verification; full-checkpoint Kaggle inference remains to be confirmed by running the notebook.
