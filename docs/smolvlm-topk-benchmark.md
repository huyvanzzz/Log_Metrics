# Benchmark Top-32 theo từng layer — SmolVLM base

Notebook: [kaggle_smolvlm_topk_benchmark.ipynb](../notebooks/kaggle_smolvlm_topk_benchmark.ipynb).

## Chạy lượt đầu

1. Push toàn bộ code mới lên `huyvanzzz/Log_Metrics`, nhánh `smolvlm-attention`.
2. Upload notebook lên Kaggle, bật **Internet** và **GPU T4 ×2**.
3. Attach cache SmolVLM đang dùng (`huyhuy12334/cache-smolvlm`).
4. Attach **bộ export attention đầy đủ đã giải nén**, gồm `attention.npy`,
   `samples.jsonl`, `metadata.json`, `manifest.json`, `README.md`. Chỉ tensor riêng
   không đủ vì phải xác định đúng sample, prompt, checkpoint và camera order.
   Nếu bộ export còn ở working directory của session hiện tại thì không cần attach lại.
5. Giữ `CONFIG_SET = 'first'` ở **cell 1**, chạy cells **1–8**.

Nếu chỉ có một cache và một export phù hợp thì không cần sửa đường dẫn.
`ATTENTION_INPUT=None` tự tìm export trong Input và thư mục output attention cũ.
Nếu nhiều export thì đặt đường dẫn cụ thể tới thư mục chứa năm file.
`PREPARED_INPUT=None` tải prepared metadata từ HF revision đã ghim; có sẵn thì đặt
đường dẫn chứa manifest/train/val/test. Không tải raw images.

Cell 1 clone/pull source, cài dependencies (giữ CUDA Torch Kaggle) và kiểm tra Java.
Nếu Java chưa có trên Kaggle, cài OpenJDK 17 để chạy METEOR. Cell 4 chuẩn bị dữ liệu,
Top-K và smoke-test đủ bảy scorers trước khi sinh toàn bộ test. GPU numerical
preflight chạy khi workers khởi động ở cell 5, mỗi GPU chỉ nạp checkpoint một lần.

## Hai lượt, 16 cấu hình

| CONFIG_SET | GPU 0 luân phiên | GPU 1 luân phiên |
|---|---|---|
| `first` | 0, 1, 2, 9 | 10, 11, 15, 16 |
| `second` | 17, 21, 22, 23 | 25, 26, 27, base |

Mỗi cấu hình chạy đủ **16.817 câu test**, không phải một phần tư dữ liệu.
Hai workers xử lý song song; trong từng worker, mỗi batch luân phiên bốn config
và dùng chung connector/fused prompt đã tính. Không giữ bốn KV caches đồng thời.

Tám cấu hình đầu không có một lượt baseline đầy đủ. Preflight có kiểm tra allkeep
trên vài mẫu, tối đa bốn generated tokens; đó không phải kết quả benchmark base.

Lượt sau chỉ đổi **`CONFIG_SET = 'second'` ở cell 1**, rồi chạy lại notebook.
Giữ OUTPUT cũ hoặc khôi phục nó. CONFIG_SET không thuộc generation identity của
từng config, nên đổi lượt không làm mất tương thích kết quả trước.

Nếu dùng session mới: tải ZIP kết quả lượt đầu lên Kaggle Input và giải nén,
đặt `PREVIOUS_RESULTS` ở cell 2 vào thư mục gốc đã giải nén để tổng hợp 16 cấu hình.
Cũng cần attach lại attention export và cache để chạy generation lượt hai.
Thư mục PREVIOUS_RESULTS chỉ được đọc, không bị sửa. Hai lượt phải có cùng
experiment identity; không ghép metric/log beam-search cũ vào baseline mới.

## Phép đo

- Model base/checkpoint/revisions giống lần xuất attention, strict loading và SHA-256.
- Chọn riêng từng sample và đúng một layer từ vector 384 điểm FP32 đã mean heads.
- **Không mean nhóm layers**. Stable Top-32 trên toàn bộ sáu camera, ties ưu tiên
  visual index nhỏ hơn, sau đó giữ thứ tự xuất hiện gốc.
- Không quota camera; camera có thể không còn visual token nào.
- Native connector và merger vẫn nhận đủ feature cache, sau đó thực sự gather/xóa
  352 visual embeddings trước LLM layer 0. Mọi text/camera labels/special tokens còn nguyên.
- Đầu vào LLM gồm **32 visual tokens cộng mọi non-visual tokens**, không phải tổng chỉ 32.
- RoPE giữ vị trí gốc; slot KV cache chạy liên tiếp theo chuỗi vật lý đã nén.
  Generated token đầu tiên khi được encode mang vị trí bằng độ dài prompt gốc.
- Base dùng cùng adapter và giữ đủ 384 visual tokens.

Tất cả dùng **greedy, max_new_tokens=512, FP32, không TF32/autocast, batch 32**.
Mẻ cuối 17 mẫu, không drop_last. EOS có thể dừng sớm. Giới hạn 512 không bảo đảm
câu trả lời đúng hơn; các câu chạm giới hạn vẫn được chấm và thống kê riêng.

**OOM là lỗi, không giảm batch.** Worker bị lỗi dừng, kết quả batch đã commit được
giữ. Worker GPU kia có thể còn chạy; cell chờ báo rõ tình trạng và đường dẫn log.

Đây là kiểm tra chất lượng **tập token do layer đó chọn** khi prune từ đầu, không
phải prune trực tiếp tại layer đó. Attention được lấy từ lần chạy full-384 trước;
không tính lại attention hay chọn lại Top-K sau pruning. Chưa huấn luyện pruner,
không có random/bottom-K, không đưa ra kết luận latency/FLOPs từ lượt GPU chia sẻ.

## Tiến độ, resume và source pull

Output mặc định: `/kaggle/working/smolvlm_topk_benchmark_outputs/`.

- `auto`: tạo run mới, resume phần thiếu của lượt đang chọn, hoặc score nếu generation hoàn tất.
- `measure`: yêu cầu OUTPUT mới.
- `resume`: tiếp tục run cùng identity.
- `score`: yêu cầu đủ predictions của lượt đang chọn; không tải model/attention/cache.

Cell 6 có tqdm cho tám config, hai GPU và tổng. Chỉ tính một batch là hoàn thành
sau khi records được kiểm tra, ghi atomic và fsync. Resume xác minh checksum rồi
giữ nguyên ranh giới batch 32; không trộn các mẫu còn thiếu thành batch khác.

Ngắt cell 6 chỉ ngắt chờ. Chạy lại cell 6 để reconnect với workers còn sống.
Không pull khi workers đang chạy. Helper ở notebook mới và các notebook cũ đã
được mở rộng để nhận diện workers benchmark; không đổi phép đo của notebook cũ.

Đổi model, attention, dữ liệu, code generation hoặc phiên bản thư viện ảnh hưởng
inference cần OUTPUT mới. Sửa scorer không buộc sinh lại: scoring identity/cache
tách riêng. Source commit trong run metadata là commit tạo experiment; code hashes
ghi lại các module liên quan, kết quả không mất tương thích chỉ vì đổi CONFIG_SET.

Giữ **toàn bộ OUTPUT** để resume generation dở dang qua session. ZIP ở cell 8 chứa
kết quả hoàn chỉnh, inputs và provenance; không chứa partial batch journals,
checkpoint, feature cache hoặc attention tensor lớn. ZIP có thể giải nén để score
lại kết quả hoàn chỉnh, nhưng không thay thế việc lưu OUTPUT đang chạy dở.

## Metrics và export

Khi generation đã xong: chạy **cells 1, 2, 7, 8** để score/export lại.

Chấm đủ corpus mỗi config bằng BLEU-1/2/3/4, METEOR, ROUGE-L và CIDEr theo pipeline
training gốc commit `4d7dbd238c7960d305c882b75220fb95beb37644`:

- Decode bỏ special tokens rồi `.strip()`, giữ reference gốc.
- Không PTB tokenization hoặc lowercase bổ sung; không mean điểm từng batch.
- Chỉ ở giao thức METEOR: line breaks đổi thành spaces và bỏ delimiter `|||`,
  để không phá stdin Java. Text gốc vẫn được lưu và dùng cho scorers còn lại.
- Không bỏ prediction rỗng hoặc chạm 512; scorer lỗi phải báo lỗi, không chuyển
  ngầm sang exact-match hay tự bỏ METEOR.
- Metrics raw và display (*100 như pipeline gốc) tách riêng; CIDEr không được
  diễn giải như tỷ lệ accuracy/phần trăm.

`configs/` có selection và batch journals. `results/<config>/` có predictions,
selection indices, absolute prompt positions và metadata/hashes. `scores/` có
metrics, diagnostics và logs CPU. `summary.json`/`summary.csv` tổng hợp các cấu hình
đã chấm; delta với base chỉ có sau khi baseline mới hoàn tất.

Cell 8 tạo ZIP kết quả kèm manifest hashes; không tự xóa dữ liệu cũ.

## Kiểm chứng

Tests bao phủ native/allkeep parity, pruned cached/full-prefix parity, mixed
padding, Top-32 thật, vị trí gốc, scheduler, batch/OOM, resume, metadata imports,
CPU scoring và notebook. Numerical preflight bắt buộc trước full generation trên
Kaggle. Checkpoint thật, VRAM batch 32 và chạy đủ test cần xác minh trên GPU;
tests CPU/fixtures không thay thế bằng chứng đó.
