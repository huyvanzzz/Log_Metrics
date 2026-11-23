# Teacher log-targets SmolVLM: index 22 là layer thứ 23

Notebook riêng: [`kaggle_smolvlm_teacher_index22_train_val.ipynb`](../notebooks/kaggle_smolvlm_teacher_index22_train_val.ipynb). Chạy trên nhánh `smolvlm-attention`. Công cụ chỉ xuất teacher targets; chưa có MLP, KL training, pruning, Top-32 hoặc evaluation.

## Luồng và công thức

Cache vision hiện có → connector frozen → đủ 384 visual tokens → native decoder blocks index 0–21 → input layernorm của block index **22** → Q cho token hợp lệ cuối prompt và K cho context → native RoPE/GQA → teacher log-score.

Index **22** tương ứng layer thứ **23**. Không chạy V, attention output hoặc FFN của block này, các block phía sau hoặc LM head. Không sinh đáp án, tính CE, backward hay cập nhật weights. Ground-truth answer không tham gia prompt.

Với logits `z[h,k]` sau RoPE/scale, attention được chuẩn hóa trên **toàn bộ valid keys**, gồm ảnh và text:

```text
log_a[h,k] = z[h,k] - logsumexp_valid_keys(z[h,:])
v[j] = logsumexp_heads(log_a[:, visual_position[j]]) - log(9)
log_p[j] = v[j] - logsumexp_384_visual_tokens(v)
```

Dùng đủ 9 query heads sau GQA từ 3 KV heads. Không mean logits trước softmax, không normalize riêng camera/head trên visual keys, không smoothing/clamp/temperature. `logsumexp(log_p)` gần 0. Query là token cuối native generation prompt, không mặc định là token cuối câu hỏi; mapping ghi ID, decoded token và vị trí thực tế.

## Inputs được pin và tài nguyên

- Best checkpoint: `huyvanzzz/smolvlm256M_base/finetune_best.pt`, revision `f9d4e9a4d704078f9e7f39f061d13602cf251c78`.
- Native model: `HuggingFaceTB/SmolVLM-256M-Instruct`, revision `7e3e67edbbed1bf9888184d9df282b700a323964`.
- Prepared data: `huyvanzzz/prepared_data`, revision `d2a6914f681dbb9148c4ac11e6fa92452010506e`.
- Cache: `huyhuy12334/cache-smolvlm/versions/1`, fingerprint `0632954bab4d7a8e5af5934f3476610a0547db8f3083495c1207c3da87b15585`.
- Camera: Front → Front-Left → Front-Right → Back → Back-Left → Back-Right.
- Hai GPU, batch **64/GPU**, FP32, autocast/TF32 tắt, model frozen và `eval()`/`inference_mode()`.

Parent validate cache identity/coverage một lần, rồi kiểm tra finiteness và SHA256 của payload trong cùng một lượt đọc. Vì vậy đổi đường dẫn giữa sessions vẫn đối chiếu được đúng nội dung cache. Workers kiểm tra receipt, files và từng feature batch; không quét lại toàn cache. Đọc cache trực tiếp từ Input; không tạo cache mới hoặc copy sang working. Prepared data và checkpoint chỉ tải từ nguồn public khi cần, không có publishing credentials hoặc upload tự động.

Train `[341381,384]` và validation `[19785,384]` chứa tổng **554.750.976 bytes**, khoảng **0,517 GiB** numeric payload FP32. Cần thêm dung lượng cho compact prompt IDs/mapping, shards, file tạm khi finalize và archive; xem storage log thực tế. Không xuất targets test.

## Chạy mới

Trong notebook, điền `CACHE_INPUT` nếu muốn chỉ định cache Input; có thể để trống để tự resolve. Điền `PREPARED_DIR` nếu đã mount prepared data; trống để dùng HF đã pin. Giữ `MODE='new'`, không điền resume/complete input. Chạy lần lượt các cell.

CLI tương đương:

```bash
python -m smolvlm_attention.teacher_targets prepare --output /kaggle/working/smolvlm_teacher_index22_train_val_loader --mode new --cache-input /kaggle/input/cache-smolvlm
python -m smolvlm_attention.teacher_targets run --output /kaggle/working/smolvlm_teacher_index22_train_val_loader
python -m smolvlm_attention.teacher_targets validate --output /kaggle/working/smolvlm_teacher_index22_train_val_loader
python -m smolvlm_attention.teacher_targets bundle --output /kaggle/working/smolvlm_teacher_index22_train_val_loader
```

Mỗi GPU stress 64 prompt dài nhất của shard train/val. Cả hai phải pass trước khi commit extraction. OOM/lỗi thì dừng, không tự giảm batch/precision, bỏ mẫu hoặc fallback CPU. Logs chứa phase/GPU/split, processed/committed/total, elapsed, throughput, ETA và peak CUDA memory; không in từng sample.

## Resume và complete reuse

Partial artifacts gồm `run.json`, `internal/prompts/` và `internal/shards/`, chứa compact token IDs phục vụ extraction/resume. Sau khi workers đã dừng, đóng gói:

```bash
python -m smolvlm_attention.teacher_targets bundle --output /kaggle/working/smolvlm_teacher_index22_train_val_loader --partial
```

Phiên sau giải nén partial ZIP, điền thư mục đó vào `RESUME_INPUT`, chọn OUTPUT mới và `MODE='resume'`. Notebook chỉ copy các artifacts resume, không copy cache/checkpoint. Nếu partial đã ở OUTPUT hiện tại thì để `RESUME_INPUT` trống. Prepare xác minh content identity rồi cập nhật operational paths; thay checkpoint/data/prompt/code/format/geometry không được tự chấp nhận. Các batch đã commit được kiểm tra checksum và không tính lại; phần chưa commit được tính lại. Không tự xóa shards.

Portable export hoàn chỉnh chỉ gồm `train/` và `val/`, mỗi split có `log_probs.npy` FP32 little-endian, `samples.jsonl` và `manifest.json`. Mapping giữ row/sample ID/cache key/index/query và hashes; không có answer hoặc full prompt IDs. Identity sample là `(split, sample_id)`.

Để dùng lại export hoàn chỉnh, điền `COMPLETE_EXPORT_INPUT` trỏ đến thư mục chứa `train/` và `val/`. Notebook validate trực tiếp, bỏ qua prepare/checkpoint/cache/GPU và có thể đóng gói lại. CLI độc lập sau khi chuyển thư mục:

```bash
python -m smolvlm_attention.teacher_targets validate --export /path/to/export
```

Không cần `run.json` hoặc inputs gốc cho portable validation. Manifest `complete` chỉ được công bố sau đủ coverage/checksum/normalization. Không xem partial ZIP là dataset hoàn chỉnh.

## Phạm vi kiểm chứng

Các kết quả kiểm thử dưới đây thuộc bản exporter serial trước; kết quả loader mới được ghi ở cuối tài liệu.

Tests dùng native model nhỏ và dữ liệu giả offline để kiểm tra eager attention tại index 22, stable log-normalization, visual mapping, padding, answer independence, weights/RNG và storage/resume. Không tải weights/dataset lớn để test local.

Kết quả local: **215 tests liên quan pass** (78,72 giây), gồm nhóm teacher mới và regression attention exporter, inference, saved-score MVPruner, Top-K, source setup. Đã kiểm tra crash trước receipt và trong khởi tạo shard, checksum hỏng, thứ tự hàng, partial archive khi worker đang chạy, và dừng worker còn lại khi một worker lỗi. Hai test fault injection được cô lập đúng module sau regression source reload; không nới tolerance hoặc bỏ assertion.

Kiểm tra cú pháp: 11 file Python mới, 6 cell Python trong notebook và config JSON hợp lệ; notebook không chứa saved outputs. Kiểm tra whitespace/diff pass; các file tracked cũ không thay đổi. Có thể chạy riêng nhóm teacher bằng:

```bash
python -m pytest tests/test_teacher_attention.py tests/test_teacher_prompts.py tests/test_teacher_storage.py tests/test_teacher_runtime.py tests/test_teacher_lifecycle.py -q
```

**Chưa xác minh batch 64/GPU, VRAM hoặc thời gian với checkpoint thật trên CUDA.** CPU tests không phải bằng chứng đủ bộ nhớ GPU hoặc tốc độ production. Notebook dừng nếu stress không pass. Các exporter/inference cũ và vision cache giữ nguyên.

## Loader version (2026-11)

Output mặc định mới: `smolvlm_teacher_index22_train_val_loader`. Run version 2 từ chối resume run serial cũ; không xóa hoặc sửa targets cũ. Portable log-score format vẫn giữ nguyên.

Mỗi GPU dùng một DataLoader pool `spawn`, 2 workers, persistent workers, pin memory và prefetch factor 1. Dataset/collator chỉ giữ paths, lazy-open memmaps read-only; một batch features được preallocate FP32 rồi copy từ cache FP16. Parent validate đầy đủ cache/prompt một lần; workers dùng receipt/stat và kiểm tra hash prompt, finite từng batch. H2D dùng `non_blocking=True`; không khẳng định transfer đã overlap compute.

Train dùng một CPU generator riêng seed 42 để tạo randperm, validation giữ row order. Hai shard lấy `order[rank::2]`; đây không phải replay chính xác batch order của Accelerate cũ. `internal/schedule` lưu lịch, batch boundaries và checksum; completion receipts phải trọn batch, resume không regroup các row còn thiếu. Loader worker seeding dùng generator riêng.

Cùng một pool xử lý stress train/validation trước, sau đó tiếp tục extraction; hai GPU stress pass mới mở gate commit. Chỉ stress chọn 64 prompt dài nhất, extraction không sort dài-ngắn. Pool đóng/join trong finally, kể cả signal dừng từ parent. Không tự giảm batch/workers/prefetch/precision khi lỗi.

Hai GPU có khoảng 4,5 GiB features FP32 trong các tác vụ prefetch, chưa tính pinned copies, batch hiện tại và bộ nhớ khác. Đây là dung lượng lý thuyết, không phải peak RAM đo được. Preflight log RAM/cgroup/shared memory, kiểm tra headroom cho một batch và chạy stress với prefetch hoạt động. Log extraction có prompt length min/max, valid tokens, wait-next, collector wall time (gồm H2D/compute/D2H), commit time, throughput. Không thêm CUDA sync để đo các wall timers; chúng không phải GPU kernel timing.

Partial archive có thêm `internal/schedule/`. Receipt operational được tạo lại sau full validation khi prepare resume và bind checksum trong run.json. Không đưa cache/checkpoint vào archive.

Chưa đo CUDA/Kaggle cho phiên bản loader này: chưa xác minh pin-memory thực tế, peak RAM/shared memory, VRAM hoặc tốc độ. Không cam kết ETA hay nhanh hơn train.

Kiểm chứng loader mới: **253 tests liên quan pass**, 126,33 giây trên CPU. Ba cảnh báo pin-memory là do máy local không có accelerator. Coverage gồm spawn hai workers, batch/padding/features khớp reference cũ, native log-score ở tolerance cũ, shuffle/RNG, cleanup khi parent/worker lỗi, lịch và receipt resume, finalize/portable validation, cùng regression attention/Top-K/MVPruner. Không nới tolerance hoặc thay attention math.
