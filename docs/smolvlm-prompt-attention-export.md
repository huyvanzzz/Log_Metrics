# Lưu attention riêng từng câu hỏi — SmolVLM base

Notebook: [kaggle_smolvlm_prompt_attention_export.ipynb](../notebooks/kaggle_smolvlm_prompt_attention_export.ipynb).

## Chạy trên Kaggle

1. Push code Visualize lên nhánh `smolvlm-attention` của `huyvanzzz/Log_Metrics`.
2. Mở notebook mới, bật Internet và GPU (một GPU hoặc T4 ×2).
3. Attach cache SmolVLM đang dùng (`huyhuy12334/cache-smolvlm`). Nếu đã mount đúng một cache thì không phải sửa đường dẫn.
4. Chạy cells 1–7. Cell đầu clone/pull toàn bộ source và cài dependencies, giữ CUDA Torch của Kaggle.

Mặc định `BATCH_SIZE=32` **mỗi GPU**, FP32, không TF32. Chỉ khi CUDA OOM mới giảm
32 → 16 → 8 → … → 1; không tăng lại. Batch đã giảm được giữ khi resume.
Không đổi batch ở notebook cũ. `PREPARED_INPUT=None` tự tải prepared metadata từ HF
revision đã ghim; đặt đường dẫn nếu bạn đã có đủ manifest/train/val/test.
`CACHE_INPUT=''` tự tìm cache đã attach; đặt cụ thể nếu có nhiều cache trùng fingerprint.

## Đo gì?

Model `huyvanzzz/smolvlm256M_base`, revision
`f9d4e9a4d704078f9e7f39f061d13602cf251c78`, checkpoint `finetune_best.pt`.
Prepared `huyvanzzz/prepared_data`, revision
`d2a6914f681dbb9148c4ac11e6fa92452010506e`.
Cache fingerprint `0632954bab4d7a8e5af5934f3476610a0547db8f3083495c1207c3da87b15585`.

Đủ **16.817 câu hỏi test**, mỗi câu giữ **30 layers × 384 visual tokens**.
Query là token hợp lệ cuối native prompt, kể cả assistant suffix, dùng dự đoán
token trả lời đầu tiên. Softmax trên mọi key hợp lệ của prompt, gồm chính query;
sau đó lấy image keys và mean heads. Không mean layers/samples, không chuẩn hóa
lại riêng ảnh, không generation. Đáp án tham chiếu chỉ là metadata, không đưa vào model.
Kết quả trung bình của notebook cũ không thay thế được lần đo riêng từng mẫu này.

Một GPU xử lý tất cả mẫu. Hai GPU có hai shard cố định 8.409/8.408 mẫu, cùng model base.
Worker strict-load checkpoint, kiểm tra native eager parity trên câu ngắn/dài
trong các prompt hoàn chỉnh tối đa 1.024 tokens và batch/single parity trước full
run. Prompt dài nhất vẫn chạy collector chính đầy đủ. Giới hạn chỉ áp dụng cho
oracle tốn bộ nhớ, không cắt prompt hoặc bỏ mẫu đo. Sai kiểm tra thì dừng.

## Tiến độ, ngắt và resume

Output: `/kaggle/working/smolvlm_prompt_attention_outputs/`, nằm ngoài source checkout.

- `auto`: chạy mới nếu chưa có run; resume nếu partial; export nếu đủ shard/kết quả.
- `measure`: chỉ chạy mới, từ chối output đã có run/state.
- `resume`: yêu cầu run cũ tương thích, giữ đúng số shard GPU.
- `export`: ghép/kiểm tra dữ liệu đã đo, không tải model hoặc đọc feature cache.

Tiến độ có số **đã tính** và **đã commit**. Worker flush/fsync và công bố progress
atomic sau mỗi 100 batch thành công, cuối shard và khi bắt lỗi measurement.
Kill process hoặc tắt session có thể làm phải đo lại phần chưa commit. Checksum
các dòng đã commit được xác minh khi resume; dữ liệu hỏng bị từ chối.

Ngắt cell 5 chỉ ngắt chờ; workers vẫn chạy. Chạy lại cell 5 để reconnect.
Muốn pull code mới, phải chờ workers kết thúc. Cell 1 chặn pull khi thấy workers sống.
Nếu mất toàn bộ Kaggle working directory thì không thể resume: cần giữ **toàn bộ
OUTPUT**, không chỉ ZIP export. Không có persistence qua session nếu bạn chưa lưu output.

Đổi model/data/tokenizer/code ảnh hưởng phép đo hoặc số shard cần OUTPUT mới.
Sửa packaging/notebook không đổi measurement fingerprint; code worker có trong
fingerprint để bắt thay đổi phép đo. Run config và
mapping cũ không bị ghi đè khi phát hiện mismatch.

## Export và dùng cho Top-K sau này

Khi đủ shard: chạy **cells 1, 2, 6, 7** để export, không cần GPU/checkpoint/cache.
Nếu ngắt lúc merge, chạy lại cell 6; dữ liệu shard không cần đo lại.

Export có đúng năm file:

- `attention.npy`: FP32 `[16817,30,384]`, khoảng 739 MiB.
- `samples.jsonl`: thứ tự dòng prepared gốc; IDs, Q/A, paths, feature key/index,
  unpadded input IDs, image-token ID và query position/ID.
- `metadata.json`: identity, layer/camera/grid policy, checkpoint hash, library
  versions và source provenance. Giữ riêng code hash ảnh hưởng phép đo và commit tổng.
- `manifest.json`: trạng thái complete, shape và SHA-256 các file khác.
- `README.md`: hướng dẫn ngắn để sử dụng độc lập.

```python
from smolvlm_attention.sample_store import open_prompt_attention
import numpy as np

result = open_prompt_attention('/path/to/extracted/export')
row = result.sample_index['test-123']
scores = result.attention[row, [21, 22, 23], :].mean(axis=0)
top_k = np.argsort(-scores, kind='stable')[:32]
sample = result.samples[row]
image_positions = np.flatnonzero(np.asarray(sample['input_ids']) == sample['image_token_id'])
absolute_prompt_positions = image_positions[top_k]
```

Reader kiểm tra hashes rồi mở memmap chỉ đọc, không import Torch. Phân biệt
visual index 0–383 với absolute prompt position. Stable sort xử lý đồng điểm theo
visual index. Đây chỉ là cách đọc dữ liệu; chưa chạy pruning hoặc benchmark.

Camera order: Front, Front-Left, Front-Right, Back, Back-Left, Back-Right.
Mỗi camera là 64 tokens, lưới 8×8, giữ orientation/cache policy hiện tại:
`camera=j//64`, `row,col=divmod(j%64,8)`.
Suggested groups: `[0,1,2]`, `[9,10,11]`, `[15,16,17]`, `[21,22,23]`, `[25,26,27]`.

ZIP chỉ chứa export, không có checkpoint/cache/shards. Giải nén trước khi dùng
memmap. Shards + export + ZIP không nén cần khoảng 2,2 GiB cộng metadata và các
artifact khác. Không tự xóa dữ liệu cũ.

## Kiểm chứng

Tests CPU bao phủ collector native parity, mapping, OOM retry, commit/resume,
checksum, merge, export reader, source guard và notebook. Full checkpoint strict
loading, VRAM, thời gian thực tế và batch 32 cần xác minh bằng preflight trên Kaggle.
