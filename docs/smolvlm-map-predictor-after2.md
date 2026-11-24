# Predictor sau block index 2, teacher index 22

Biến thể `map_after2_lastprompt_direct_v1` độc lập với predictor trước LLM.
Index 2 là layer thứ 3; teacher index 22 là layer thứ 23.

## Luồng tính toán

Train/validation: cache FP16 → connector frozen FP32 → native prompt đầy đủ
384 visual tokens → toàn bộ blocks 0,1,2 frozen → visual states và state của
token hợp lệ cuối prompt → predictor → log-softmax trên 384 tokens → KL với
teacher log-target đã lưu. Không residual; không chạy suffix, final norm,
LM head, vision encoder, generation hoặc teacher online.

Predictor có hai MLP riêng `576 → 128 → 128` với GELU, Q/K riêng `128 → 64`
không bias, tổng 197.120 tham số. Q/K mới dùng native RoPE tại original prompt
positions; một query cuối prompt, không mean câu hỏi. Không xoay lại hidden
states của LLM. KL cộng chiều visual rồi mean samples; không chia cho 384.

Test: initial prefill chạy blocks 0–2 một lần, predictor một lần, stable global
Top-32 rồi gather **hidden states hiện tại sau toàn bộ block 2**. Giữ mọi
non-image token hợp lệ và original positions. Suffix tiếp tục trên chuỗi compact.
KV cache ở blocks 0–2 giữ full context, suffix giữ compact context; mask/cache
slots theo từng layer. Selection cố định suốt greedy decode, tối đa 512 tokens.

MAP là cơ sở cho hai nhánh MLP, Q/K, RoPE và KL. Boundary sau block 2, query
cuối prompt và teacher cố định là lựa chọn của thí nghiệm này, không phải tuyên
bố tái lập recipe nguyên bản của MAP.

## Inputs và cách chạy Kaggle

Mở `notebooks/kaggle_smolvlm_map_predictor_after2_lastquery_train_test.ipynb`.
Source phải chứa package `smolvlm_map_predictor_after2` trên nhánh
`smolvlm-attention`; notebook báo lỗi nếu clone đang cũ. Bật Internet và hai GPU.

Điền `TARGETS_INPUT` vào thư mục Kaggle Input chứa:

```text
train/log_probs.npy, train/samples.jsonl, train/manifest.json
val/log_probs.npy, val/samples.jsonl, val/manifest.json
```

Chỉ nhận portable export hoàn chỉnh; không nhận partial shards. Targets và cache
đọc trực tiếp, không copy vào working. Join `(split, sample_id)`, kiểm tra hashes
câu hỏi/native prompt, prepared row và cache key/index; không join mù theo row.

Giữ base `huyvanzzz/smolvlm256M_base/finetune_best.pt` tại revision
`f9d4e9a4d704078f9e7f39f061d13602cf251c78`; native model revision
`7e3e67edbbed1bf9888184d9df282b700a323964`. Prepared data:
`huyvanzzz/prepared_data`; cache: `huyhuy12334/cache-smolvlm/versions/1`.
Camera: Front → Front-Left → Front-Right → Back → Back-Left → Back-Right.
Public downloads không cần publishing credentials.

Run mới để trống `RESUME_INPUT`, `RESUME_CHECKPOINT`, dùng output mặc định
`smolvlm_map_predictor_after2_lastquery_index22_optimized_outputs`. Notebook lần lượt
validate inputs/metrics → stress train hai GPU → train 5 epoch → stress test
hai GPU → full test/metrics → bundle. Không tự giảm batch hoặc đổi precision.

CLI tương đương:

```bash
python -m smolvlm_map_predictor_after2 validate-inputs --config configs/smolvlm_map_predictor_after2_lastquery_index22.json --targets-input /kaggle/input/YOUR-TARGETS/teacher_targets --output /kaggle/working/smolvlm_map_predictor_after2_lastquery_index22_optimized_outputs
python -m smolvlm_map_predictor_after2 diagnose --run OUTPUT/run.json
python -m smolvlm_map_predictor_after2 train --run OUTPUT/run.json
python -m smolvlm_map_predictor_after2 evaluate --run OUTPUT/run.json --predictor OUTPUT/checkpoints/predictor_best.pt
python -m smolvlm_map_predictor_after2 bundle --run OUTPUT/run.json
```

Thay `OUTPUT` bằng thư mục thực. CLI production yêu cầu hai GPU, không fallback
CPU. Notebook chỉ test khi training đã đủ epoch; CLI evaluate là lệnh riêng
cho một checkpoint được chọn rõ ràng.

## Recipe, checkpoint và resume

FP32 compute/H2D from the projected cache; raw FP16 is read only during one-time connector cache construction.
Chỉ predictor vào DDP/optimizer: batch 32/GPU, accumulation 1, effective batch 64,
5 epochs, AdamW LR `1e-4`, weight decay `0.05`, gamma `0.9` mỗi epoch hoàn tất,
clip 1.0, seed 42. Loader spawn, 2 workers/GPU, persistent/pin memory, prefetch 1.
Giữ seedable shuffle/set_epoch/even batches của Accelerate, gồm bổ sung mẫu ở
batch train cuối. Validation không duplicate, sample-mean KL; train epoch loss
là mean microbatch loss. Best chọn bằng validation KL, không Top-32 overlap.

Giữ `predictor_latest.pt`, `predictor_best.pt` **kèm các thư mục `.pt.rng`**.
Chỉ predictor và training state được lưu, không sao chép backbone. Checkpoint
ghi RNG từng rank trước `.pt`, improved best trước latest. Resume kiểm tra
method/architecture, base/data/targets, code/library hashes, recipe/world size,
sidecar counters; từ chối checkpoint trước LLM dù shapes giống nhau. RNG được
khôi phục sau khởi tạo/prepare/iterator seek. Production checkpoint ở ranh giới
epoch; ngắt giữa epoch thì chạy lại phần chưa lưu từ checkpoint hoàn chỉnh.

Qua session khác: giải nén artifact vào Kaggle Input và đặt `RESUME_INPUT` là
root có `run.json` và `checkpoints/`. Notebook copy các checkpoint nhỏ cùng
`.rng`, rebuild operational paths sau kiểm tra identity. Cùng session có thể
đặt `RESUME_CHECKPOINT` tới latest trong output. Không sửa receipt cũ, targets
hoặc checkpoint base. Inference không cần optimizer/RNG hay teacher targets.

## Log và artifacts

`debug/train_progress.jsonl` có timestamp/event phẳng, loss microbatch, LR,
counters, elapsed/throughput/ETA và CUDA memory; epoch summary ghi train/val KL.
`debug/debug_events.jsonl` đếm thực tế từng prefix block và cấm suffix trong
train/validation. Diagnose ghi query ID/decoded token/position, gradient và RAM,
shared memory, CUDA allocated/reserved. Best/latest event chỉ ghi sau save thành công.

Test chia 16.817 mẫu thành 8.409/8.408, batch 64/GPU, tails 25/24, hai process
độc lập. Batch commits có checksum/identity để resume; coverage đủ trước metrics.
`map_predictor_after2_artifacts.zip` và `.sha256.json` chứa predictor checkpoints,
RNG, config, logs và predictions/metrics; không chứa base, cache hoặc targets.
Timing QA không phải benchmark latency/FLOPs.

## Phạm vi kiểm chứng

Kiểm thử dùng native model nhỏ thực trên CPU, không tải checkpoint/dataset lớn.
Bao gồm boundary/cache parity, last-query mapping, KL/gradients, freeze guards,
two-rank CPU/Gloo, epoch resume/RNG, strict identity, greedy/reference,
target-free test, metrics/bundle và regression các luồng cũ.

Chưa chạy CUDA/checkpoint thật: chưa xác minh batch train 32/test 64 mỗi GPU,
VRAM, RAM/shared-memory peak, tốc độ, chất lượng hoặc thời gian 5 epoch trên
Kaggle. Stress thật trong notebook phải pass trước production.

Kết quả xác minh local ngày 2026-11-24: 28 tests riêng của biến thể mới và
287 tests hồi quy liên quan đều pass. Ba warnings của test loader cũ thông báo
pin memory không hoạt động trên máy không có accelerator. Môi trường kiểm tra:
Torch 2.15.0+cpu, Transformers 4.46.3, Tokenizers 0.20.4, Accelerate 1.15.0.
Notebook JSON/Python syntax, CLI help và whitespace checks pass; SHA-256 của
167 file cũ được bảo vệ giữ nguyên. Không commit/push hoặc chạy production.


## Optimized shared I/O (5 epochs)

The optimized notebook uses the same projected FP32 cache and compact input
pipeline documented in [the before-LLM guide](smolvlm-map-predictor.md#optimized-pipeline).
`PROJECTED_CACHE_INPUT` reuses a complete mounted cache without loading the raw
vision cache. Otherwise `PROJECTED_CACHE_DIR` builds it once from raw FP16 input;
the raw cache and teacher targets are unchanged. Connector calls are absent
from train/validation/test after cache construction. The frozen blocks 0?2 still
run normally; no hidden-state cache or teacher online is introduced.

Total training is **5 epochs**, with a new output and pipeline identity. Old
predictor runs/checkpoints are preserved and rejected as training resumes into
this version. One persistent pool per rank is shared across train/validation,
with 2 workers, prefetch 1 and unchanged Accelerate sample order semantics.
Notebook progress is compact; detailed JSON and timing results are file-only.
