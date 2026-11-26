# SmolVLM base: điểm attention tổng hợp trong cùng layer

Notebook: [kaggle_smolvlm_combined_visual_attention.ipynb](../notebooks/kaggle_smolvlm_combined_visual_attention.ipynb).

## Chạy trên Kaggle

1. Push toàn bộ code mới lên `huyvanzzz/Log_Metrics`, branch `smolvlm-attention`.
2. Upload notebook mới lên Kaggle, bật Internet và **GPU T4 ×2**.
3. Attach feature cache SmolVLM đang dùng: `huyhuy12334/cache-smolvlm`.
4. Giữ `CACHE_INPUT=''`, `PREPARED_INPUT=None` nếu chỉ có một cache phù hợp;
   prepared data và checkpoint base sẽ được tải từ HF tại revision đã ghim.
5. Chạy cells **1–10**. Cell 1 clone/pull và làm mới imports; cell 4 khởi động
   workers; cell 5 chờ và hiển thị tqdm; cells 7–9 vẽ hình; cell 10 tạo ZIP.

Output mặc định: `/kaggle/working/smolvlm_combined_visual_attention_outputs/`.
Không cần file attention export cũ: file đó thiếu các hàng attention trung gian,
không khôi phục được phép đo mới. Không tải raw images hoặc model reverse.

## Phép tính

Với mỗi sample, chỉ thu layer index `0,2,4,...,28`. Query `q` là token hợp lệ
cuối native prompt, bao gồm assistant suffix. Mean heads trước, rồi tính:

```text
S_l(q,v) = A_mean_l[q,v] + sum_t A_mean_l[q,t] * A_mean_l[t,v]
```

`t` gồm mọi non-visual token hợp lệ, kể cả tên camera, dấu phân cách và special
tokens; loại padding và chính `q` theo vị trí. Cả hai trọng số trong tích lấy
từ cùng layer. Mỗi hàng softmax trên toàn bộ key hợp lệ theo causal mask,
bao gồm key của chính hàng đó. Token chưa nhìn thấy ảnh đóng góp 0 đến ảnh ấy.

Đây là điểm tổng hợp do thí nghiệm định nghĩa, không phải rollout, phép scale
theo V hoặc đo đóng góp nhân quả. Không chuẩn hóa lại riêng 384 vị trí ảnh.
Với tập trung gian này tổng điểm ảnh không vượt 1, ngoài sai số số học.
Tính từng sample trước rồi lấy trung bình các sample với trọng số bằng nhau;
không lấy tích của các attention đã mean samples.

## Thực thi và bảo toàn kết quả

- Base checkpoint `f9d4e9a4d704078f9e7f39f061d13602cf251c78`, prepared revision
  `d2a6914f681dbb9148c4ac11e6fa92452010506e`; giữ native processor/cache policy.
- Đủ **16.817** câu test; hai GPU chia **8.409/8.408** mẫu. **Batch 64 cố định**,
  mẻ cuối **25/24**. Nếu OOM thì dừng, không giảm batch hoặc cắt prompt.
- FP32, `eval()`/`inference_mode()`, tắt TF32/autocast; không generation,
  pruning, training hoặc chạy vision encoder. Native forward vẫn đi qua các tầng
  cần thiết, nhưng chỉ hook 15 tầng đã chọn để thu điểm.
- Các hàng query trung gian tính theo chunk 8; không giữ ma trận toàn bộ
  queries × keys qua tất cả layer. Chunk không làm thay đổi tập key softmax.
- Trước full measurement, worker kiểm tra native eager oracle trên mẫu ngắn
  và batch/single trên mẫu ngắn/dài. Native tolerance `atol=1e-6, rtol=1e-4`;
  batch/single `atol=1e-5, rtol=1e-4`.
- Snapshot FP64 sum/count/processed IDs có checksum và ghi atomic sau mỗi batch.
  Progress chỉ tính batch đã lưu. Merge bằng tổng sum/tổng count, không mean
  hai shard với trọng số bằng nhau. Không lưu tensor riêng từng sample.
- `auto` tạo mới/resume hoặc chuyển plot khi có kết quả hoàn chỉnh. `measure`
  từ chối run có sẵn; `resume` cần identity khớp; `plot` chỉ đọc kết quả đã merge.
- Sửa model, phép tính hoặc code đo cần OUTPUT mới; sửa plotting không buộc đo lại.
  Metadata hình lưu cả commit tạo measurement và commit dùng để vẽ.

Ngắt cell 5 không dừng workers; chạy lại cell 5 để reconnect. Không pull source
khi workers còn chạy. Lỗi hiển thị nguyên nhân và đường dẫn log; GPU còn lại có
thể vẫn tiếp tục. Muốn resume qua session, phải lưu **toàn bộ OUTPUT**, gồm
`combined_run.json`, `inputs.jsonl`, snapshots trong `shards/` và các receipt.
Các đường dẫn cache/checkpoint có thể được cập nhật khi prepare lại sau khi
xác minh identity; không ghi đè kết quả cũ.

## Vẽ lại và tải hình

Khi kết quả đã merge, chạy cells **1,2,6–10**; có thể đặt `RUN_MODE='plot'` ở
cell đầu. Không cần model, cache hoặc GPU để vẽ lại.

- Token atlas: 5×3 panels, 384 cột/layer; sáu đoạn camera theo native prompt.
- Spatial atlas: 5×3 panels; mỗi panel sáu camera một hàng, từng camera lưới
  8×8 nguyên orientation. Thứ tự Front, Front-Left, Front-Right, Back,
  Back-Left, Back-Right.
- Histogram: đúng 15 layer chẵn; mỗi layer có 384 giá trị đã mean samples,
  100 bins tuyến tính chung. Đây không phải histogram của tất cả sample thô.
- Token Y và màu viridis dùng log chung; zero không có cột và hiện xám trên
  heatmap. Histogram magma, màu count theo log.

Mỗi loại xuất PNG 300 dpi và PDF trong `figures/base/`. ZIP chỉ gồm sáu hình,
metadata và caption, không chứa model/cache/snapshots. Kết quả số tổng hợp
`combined_attention.npz` có sum/mean shape `[15,384]`, layer IDs và count;
`result.json` lưu hash và identity. Không tự xóa output cũ.

VRAM/tốc độ batch 64 và full test trên T4 cần xác minh bằng lượt chạy Kaggle;
tests CPU và fixtures không thay thế bằng chứng GPU thực tế.
