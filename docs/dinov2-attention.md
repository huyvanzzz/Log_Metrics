# DINOv2–T5 base attention trên Kaggle

Notebook: [kaggle_dinov2_base_attention.ipynb](../notebooks/kaggle_dinov2_base_attention.ipynb).
Toàn bộ implementation nằm trong `dinov2_attention/` của Visualize. Không chạy training,
pruning, DINO backbone hoặc truy cập repository Windows khi chạy Kaggle.

## Cách chạy

1. Push các thay đổi Visualize lên nhánh `smolvlm-attention` của
   `https://github.com/huyvanzzz/Log_Metrics.git` trước khi chạy notebook trên Kaggle.
2. Tạo notebook Kaggle từ file trên, bật Internet và chọn một GPU hoặc T4 x2.
3. Chạy cell 1. Cell này clone/pull **toàn bộ repository** vào
   `/kaggle/working/visualize-github-source`, cài dependencies inference, giữ Torch của
   Kaggle và kích hoạt source mới. Có thể sửa repository/nhánh ở đầu cell.
4. Để `RUN_MODE='auto'` và chạy lần lượt các cell. Run mới được chuẩn bị; run dở được
   resume; run hoàn chỉnh chuyển sang vẽ. Batch khởi đầu **32/GPU**, chỉ giảm khi CUDA OOM.
5. Cell cuối tạo ZIP và link tải sáu file hình PNG/PDF, metadata và caption.

Output nằm riêng ở `/kaggle/working/dinov2_attention_outputs/`; pull không xóa output.
Không pull khi worker còn chạy: cell setup sẽ từ chối. Ngắt cell chờ chỉ ngắt việc
theo dõi; chạy lại cell chờ để xem tiến độ. Lỗi worker hiển thị nguyên nhân cùng
`logs/shard_N.log`; đọc traceback trong log trước khi resume.

Khi sửa code: push → chờ worker kết thúc → chạy lại cell 1 → chạy các cell tiếp theo.
Nếu pip thay phiên bản thư viện đã import trong kernel, restart kernel rồi chạy lại.
Không reset local edits tự động; checkout bẩn cần được người dùng xử lý trước pull.

## Measure, resume và plot

- `measure`: chỉ tạo run mới; từ chối run đã tồn tại.
- `resume`: giữ cấu hình, danh sách câu hỏi và topology shard cũ, chỉ đo IDs chưa hoàn thành.
- `plot`: chạy cells **1–2, 6–10**; không tải model, không quét cache, không cần GPU.
- `auto`: tự chọn một trong ba chế độ theo coverage snapshot.

Resume yêu cầu cùng số GPU (một hoặc hai), phiên bản thư viện và measurement fingerprint.
Nếu phép đo thay đổi, chọn `OUTPUT` mới. Thay đổi `plotting.py` hoặc runtime hiển thị
tiến độ không buộc đo lại. Hash model/collector/data/config và accumulator phát hiện
thay đổi phép đo; thay đổi `data.py`, kể cả sửa helper bên trong, được kiểm tra bảo thủ.
Không sửa `run.json` hoặc `prepared_test.json` bằng tay. Metadata hình giữ riêng commit
tạo measurement và commit dùng để vẽ.

Snapshot FP64 sum và processed IDs được ghi atomic mỗi 100 batch thành công và cuối
shard; khi worker báo lỗi Python, các batch đã commit cũng được lưu. Nếu tiến trình bị
kill đột ngột, tối đa 99 batch cần đo lại. Batch thất bại không được cộng vào sum.
Hai GPU chia 8.409 và 8.408 câu hỏi; merge bằng tổng sum/tổng count, không mean hai shard.

## Phép đo

```text
cached DINO features [B,6,64,384], FP16 → FP32
 → row + column embeddings → camera embeddings
 → flatten camera, row-major → linear projector → visual modal embedding
 → nối text embeddings + text modal embedding
 → native T5 encoder (không trả encoder attention)
 → native T5 decoder: query <pad> ID 0, decoder mask = 1
 → cross_attentions [B,8,1,384+T] × 4 layers
 → query 0, mean heads, lấy encoder keys 0:384
 → FP64 sum / số câu hỏi → mean [4,384]
```

Prompt nguyên bản là `Question: {question} Answer:`; không dùng answer trong prepared
data. Tokenization giữ special tokens và không truncation, không áp giới hạn 512 cho
chuỗi visual+text. Độ dài token min/max được lưu trong prepared-input file. T5 dùng
relative positions, `d_model=384`, tám heads với `d_kv=64`.

Attention đã softmax trên **mọi encoder key hợp lệ**. Tổng 384 giá trị ảnh có thể nhỏ
hơn 1; không renormalize theo ảnh. Query là decoder-start dùng dự đoán token trả lời
đầu tiên, không phải token cuối câu hỏi hoặc token đã sinh. Đây là cross-attention đến
**encoder visual positions đã được trộn bidirectional**, không phải raw image patches
hay encoder self-attention. Phép đo quan sát này chưa chứng minh nguyên nhân causal
attention tạo positional bias và chưa thực hiện attention intervention.

## Artifacts được ghim

| Artifact | HF repository | Revision |
|---|---|---|
| Checkpoint `finetune_best.pt` | `huyvanzzz/dinov2_base` | `84f863d54c0f1526b10b6c51d6496ecb29aea5cf` |
| Prepared data | `huyvanzzz/prepared_data` | `d2a6914f681dbb9148c4ac11e6fa92452010506e` |
| DINO feature cache | `huyvanzzz/cache_dinov2` | `051552392528900ce55bbaef1707b74fd5dd4419` |
| T5 config/tokenizer | `google/t5-efficient-mini` | `af3816c2dbd386f7786f7757c536f0314084ff8b` |

Checkpoint SHA-256: `c8d640939cf9c1bb106814eeffbd8fe33c7b16e716bc6a5c8f5a23f811cc16b8`.
Cache fingerprint: `39546b7a46379be3ac39e9b7b907d80a7291d0869adbdfc413649c83f47c79bc`.
Reference baseline commit: `11d1d431176d744b53ddeab61c7f669cc390f63a`, config
`dinov2_t5_efficient_mini_kaggle_2gpu_unpruned.yaml`, class `UnprunedVLMForAD`.

Tải duy nhất `finetune_best.pt`; không tải T5 pretrained weights hoặc RNG sidecars.
Loader khởi tạo native T5 từ config, kiểm tra SHA, architecture, tokenizer `<` ID32100,
embedding rows và tied weights rồi strict-load checkpoint. Không resize hoặc sửa weights
để làm checkpoint không tương thích chạy được. Không restore optimizer/training RNG.

Cache FP16 `[4072,6,64,384]` khoảng 1,12 GiB. Lần đầu kiểm tra manifest v2, prepared/index
SHA, geometry, dense index, mọi test ID và finite feature values. Receipt cho phép resume
không quét lại toàn payload; mở lại vẫn xác thực identity/hash và payload size/mtime.
Cache key là camera paths theo **tên camera đã sort**, nối bằng `|`, khác cache SmolVLM.
Train/val/test phải có đúng 341.381 / 19.785 / 16.817 câu hỏi.

## Hình và dữ liệu xuất

- `token_atlas`: 2×2 layers 0–3; mỗi panel 384 cột, chia sáu đoạn 64 tokens.
- `spatial_atlas`: 2×2 layers; mỗi panel sáu heatmap 8×8 trên **một hàng**, không flip/rotate.
- `layer_histogram`: bốn layers, 100 attention bins tuyến tính chung; mỗi layer chứa đúng
  384 giá trị đã mean heads và samples. Màu count dùng log magma.

Camera order: Front → Front-Left → Front-Right → Back → Back-Left → Back-Right.
Token Y và màu attention viridis dùng log chung bốn layers. Zero không có cột,
heatmap hiển thị xám. PNG 300 dpi và PDF; không cần LaTeX. Style typography/spines được
tham khảo từ folder `figures4papers/`, không phụ thuộc package đó khi chạy Kaggle.

Ngoài `figures/`, output có `run.json`, validation receipt, prepared mapping,
`data/shard_N/attention.npz`, logs/status, `summary.npz` và `figure_metadata.json`.
ZIP chỉ chứa hình/metadata/caption, không chứa checkpoint hoặc cache.

## Kiểm chứng và giới hạn

```powershell
.venv/Scripts/python -m pytest tests -q
.venv/Scripts/python scripts/build_attention_notebooks.py
.venv/Scripts/python scripts/build_dinov2_attention_notebook.py
```

Tests CPU dùng native T5 nhỏ và cache fixture để kiểm tra strict-loader policy,
mask/query, native attention và generation parity, batch padding, OOM retry,
coverage/resume, weighted merge, provenance, figures/ZIP và notebook/pull regressions.
Đối chiếu riêng với class baseline trong repository tham chiếu xác nhận state-dict
tương thích, `_native_inputs` bit-exact và greedy generation khớp. Tests thông thường
không phụ thuộc đường dẫn repository tham chiếu.

Mỗi worker Kaggle phải strict-load **checkpoint thật** rồi vượt numerical preflight:
đủ `[B,4,384]`, first-token logits/greedy parity và batch-vs-single parity. Preflight
thất bại thì không đo full test. GPU VRAM, throughput, checkpoint thật và khả năng giữ
batch 32 chưa được xác minh trong môi trường CPU local; xem log thực tế trên Kaggle.
