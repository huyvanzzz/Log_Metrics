# SmolVLM base: causal perturbation và object localization

Notebook: [kaggle_smolvlm_causal_localization.ipynb](../notebooks/kaggle_smolvlm_causal_localization.ipynb).

## Chạy trên Kaggle

1. Push package `smolvlm_probing`, builder/notebook, helper và **toàn bộ**
   `data/drivelm_attention_v1` lên nhánh `smolvlm-attention` của `huyvanzzz/Log_Metrics`.
   Không push `.cache`, checkpoint hoặc ảnh nguồn. Không có bước push tự động.
   `.gitattributes` giữ nguyên bytes của dataset và hai script lọc để Git trên
   Windows không đổi line endings làm sai SHA-256 khi chạy trên Kaggle/Linux.
2. Upload notebook mới; bật Internet và GPU (một GPU hoặc T4 ×2).
   Attach cache SmolVLM hiện có `huyhuy12334/cache-smolvlm`.
3. Chạy các cell theo thứ tự. Cell 1 clone/pull toàn bộ source; cell 3 validate và
   chuẩn bị dữ liệu, cell 4 kiểm tra native parity trước khi chạy đầy đủ.

Không cần sửa cấu hình mặc định khi dùng các artifact đã chốt. Nếu có nhiều cache
phù hợp, đặt `CACHE_INPUT` tại cell 2. `PREPARED_INPUT` có thể trỏ tới folder prepared
đã có đủ manifest/train/val/test; nếu không, chỉ các metadata files được tải từ HF.
Không chạy vision encoder và không tải ảnh gốc cho phần đo attention.

Cell 1 kiểm tra workers rồi `git pull --ff-only`, không reset local edits và không
xóa output. Nếu pip thay đổi phiên bản đang import trong kernel, restart kernel rồi
chạy lại cell 1. Giữ CUDA Torch của Kaggle; inference dùng Transformers 4.46.3 và
tokenizers 0.20.4. Các notebook cũ chỉ được tái sinh helper để cùng chặn worker mới,
không thay phép đo của chúng.

Output mặc định: `/kaggle/working/smolvlm_probing_outputs/`.
Một GPU xử lý 2.581 câu; hai GPU chia 1.291/1.290 câu với cùng base checkpoint.
Batch bắt đầu **16/GPU**; CUDA OOM giảm một nửa. Không tự tăng batch hoặc đổi precision.
Tqdm hiển thị stage/shard/tổng/batch; causal có 26 điều kiện mỗi câu nên một batch
có thể lâu hơn generation. Lỗi hiển thị nguyên nhân và đường dẫn log.

## Resume và cập nhật source

- `RUN_MODE='auto'`: mới thì measure, partial tương thích thì resume, đầy đủ thì plot.
- `measure` từ chối run đã tồn tại; `resume` yêu cầu run cũ; `plot` yêu cầu đủ kết quả.
- Giữ toàn bộ output và cùng số GPU/shards khi resume. Nếu artifact nằm ở session
  khác, cần giữ hoặc khôi phục các input paths mà run config tham chiếu.
- Ngắt cell chờ không giết workers. Chạy lại cell đó để nối progress. Không pull
  khi worker còn chạy, kể cả worker notebook khác.
- Generation ghi atomic trước baseline replay. Nếu replay/causal lỗi hoặc OOM,
  câu đã sinh không bị sinh lại. Mỗi stage giữ kết quả theo sample ID và checksum.
- Sửa model/tokenization/scoring/matching hoặc code phép đo sẽ bị phát hiện;
  cần OUTPUT mới. Không ghi đè config, prepared-input file hoặc validation receipt
  của run không tương thích. Sửa plotting không làm mất khả năng dùng kết quả cũ.
- **Vẽ lại sau pull: chạy cell 1, sau đó cells 8–11.** Không chạy model/quét cache.

## Phép đo cố định

Dataset đã chốt theo metadata: 2.581 câu causal, 749 câu localization có tọa độ
(749 targets), 210 câu không tọa độ chính xác (652 targets). Hai tập localization
là tập con của causal. Giữ Q/A và sáu camera paths nguyên trạng; không dùng đáp án
tham chiếu hoặc bbox làm nội dung prompt bổ sung.

Model `huyvanzzz/smolvlm256M_base`, revision
`f9d4e9a4d704078f9e7f39f061d13602cf251c78`, `finetune_best.pt`.
Greedy một beam, tối đa 64 tokens mới. FP32, eval/inference mode, không TF32/autocast.
EOS native dừng generation; câu chạm giới hạn vẫn giữ lại và được ghi `length`.
Special tokens nằm giữa output được giữ trong sequence/context nhưng mask khỏi
queries và điểm số. Reserved image generation được ghi invalid, không tự sửa.

Attention là xác suất sau softmax trên mọi causal key hợp lệ, gồm chính query;
không renormalize riêng ảnh. Query là token đã sinh khi replay, khác với query
trước token dùng để dự đoán nó. Original generated IDs luôn được giữ nguyên.

### Causal perturbation

Tại từng layer 1,7,13,19,25 (zero-based): mean heads rồi mean content-token queries
của đáp án gốc để xếp hạng 384 visual positions. Chọn Top-32/Bottom-32, hòa theo
index tăng dần. Random-32 ba lần với seed SHA-256/PCG64 cố định theo sample/repeat,
dùng chung giữa năm layers. Không mean samples trước khi tính chỉ số.

Mỗi cách chọn đo remove (zero 32 vị trí) và keep (zero 352 vị trí còn lại), tại
fused visual embeddings sau connector và trước decoder layer đầu. Giữ nguyên
text, sequence length, mask, positions; không tái sử dụng KV cache baseline.
Không sinh đáp án mới sau can thiệp: teacher-force đúng chuỗi gốc, chấm token từ
logits tại vị trí trước nó, không tính special tokens/EOS/padding.

`s = mean(log p(token))`; `Comp = -expm1(s_remove - s_base)`;
`Sgap = -expm1(s_keep - s_base)`. Đây là tỷ lệ trung bình nhân xác suất token,
không phải xác suất toàn chuỗi và khác mean probability của paper tham chiếu.
Random mean ba chỉ số đã tính riêng từng lần. Giữ số âm/outliers, không Otsu.
Scoring/attention lỗi số học dừng stage; không âm thầm loại câu.

Năm scatter có cùng linear limits: Top xanh dương, Random cam, Bottom xanh lá.
Đáp án rỗng/invalid được audit và không vào mẫu số causal; không chọn lại sample.
Đáp án model có thể sai; phép đo đánh giá sự phụ thuộc của chính đáp án tự sinh.
Zero contextualized embedding không tương đương xóa vùng ảnh hoặc xóa hết thông tin
về vật, vì visual encoder đã trộn thông tin giữa các vị trí.

### Object localization

Chỉ nhận khi literal `reference_span.text` xuất hiện một lần trong output, đúng
case/whitespace và word boundary. Không synonyms/fuzzy/LLM judge. Cụm lặp, chồng lấn
bindings, thiếu cụm, invalid/empty hoặc không căn được token đều có audit reason.
Ghép token trực tiếp bằng prefix decode của original IDs; không tokenize lại rồi
dùng các IDs mới. Prefix decoding không ổn định được ghi unalignable.

Mean heads → tokens trong cụm → layers trong các nhóm 0–4/5–9/10–14/15–19/20–24/25–29.
Sau đó một argmax trên **toàn bộ 384 positions**, không peak riêng camera/layer.
Mỗi ảnh 64 tokens grid 8×8 row-major. Peak đúng khi đúng camera và tâm ô nằm trong
bbox, tính cả boundary, không thêm dung sai. Tâm ô `(col+.5)*W/8, (row+.5)*H/8`.

Accuracy đếm theo target, mẫu số là targets đủ điều kiện, cố định cho sáu nhóm.
Coverage = eligible / toàn bộ targets. Denominator bằng 0 ghi N/A. Bbox không
chứa tâm ô nào vẫn giữ lại: 167/749 có tọa độ và 148/652 không tọa độ chính xác.
Exact textual match không chứng minh model nhận diện đúng về ngữ nghĩa. Đây là
metric bbox center hit, không phải segmentation accuracy hoặc object detection.

Bar chart báo cáo riêng hai nhóm dữ liệu. Mỗi nhóm chọn layer group accuracy cao
nhất (hòa chọn sớm hơn) để minh họa hậu nghiệm; không gọi đây là validation độc lập.
Montage có ảnh gốc, heatmap/bbox và dấu + ở global peak, tiêu đề ghi khoảng layer,
caption ghi exact phrase. Sai camera có panel camera chứa peak riêng; không đặt
dấu + giả lên ảnh target. Không có segmentation masks.

Chỉ tải ảnh cho các panel được chọn, tối đa ba hàng mỗi partition; cache local,
serial retry giới hạn và tôn trọng HTTP 429. Lỗi ảnh không thay đổi measurement;
rerun cell 10 để thử tải ảnh thiếu. Overlay dùng log colors chung mỗi montage,
zero mask; nội suy chỉ để hiển thị, không thay peak trên grid gốc.

## Files và kiểm chứng

`probing_run.json`, `inputs.json`, validation receipts, `state/{generation,baseline,
causal,localization}/*.npz`, logs/status/preflight receipts, responses, summaries,
localization audit, figures/captions/provenance và ZIP được giữ dưới OUTPUT.
ZIP chỉ chứa hình, numeric summaries, audit và metadata; không có checkpoint,
feature cache hoặc toàn bộ ảnh nguồn. PNG 300 dpi + PDF.

Preflight kiểm tra strict load/geometry/processor, native eager attention,
predecessor scoring, zero-ablation và cached decode parity. Nếu generation của
probe rỗng/invalid, receipt ghi rõ fallback dùng cho numeric oracle; fallback
không được lưu thành kết quả thật. Kiểm chứng local dùng fixture nhỏ, không thay
thế việc chạy preflight checkpoint thật trên GPU.

Các lệnh local:

```powershell
.venv/Scripts/python -m pytest tests -q
.venv/Scripts/python scripts/build_probing_notebook.py
```

Khi sửa helper pull, tái sinh cả hai notebook SmolVLM cũ và notebook DINOv2 bằng
builders tương ứng, rồi chạy regression. Measurement source thay đổi bình thường
chỉ cần push và cell pull; không phải upload notebook lại.

Tham khảo: [causal perturbation](https://arxiv.org/html/2609.00830v1),
[object attention/localization](https://openaccess.thecvf.com/content/CVPR2025/papers/Kaduri_Whats_in_the_Image_A_Deep-Dive_into_the_Vision_of_CVPR_2025_paper.pdf),
[Transformers 4.46.3](https://github.com/huggingface/transformers/tree/v4.46.3),
[figures4papers](https://github.com/ChenLiu-1996/figures4papers).
