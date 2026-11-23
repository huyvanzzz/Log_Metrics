# Top-32 bằng score index 22: prune trước LLM hoặc sau index 2

Thí nghiệm mới trong `smolvlm_topk_deferred` so hai vị trí pruning trên cùng checkpoint best SmolVLM, thứ tự camera Front → Front-Left → Front-Right → Back → Back-Left → Back-Right.

| GPU | Cấu hình | Luồng visual tokens |
|---|---|---|
| 0 | `score22_prune_input` | Chọn Top-32 trước block 0, chạy 30 blocks |
| 1 | `score22_prune_after2` | 384 tokens qua blocks 0–2, gather Top-32, chạy blocks 3–29 |

Index 22 là **layer thứ 23**. Index 2 là **layer thứ 3**; gather xảy ra sau toàn bộ block, gồm attention, residual và FFN. Cả hai dùng cùng selection artifact: global Top-32 theo probability attention đã lưu của base index 22, hòa điểm ưu tiên visual index nhỏ, sau đó sắp theo thứ tự nguồn.

Không Stage 1, quota camera, scorer online hoặc training. Score sâu chỉ chọn token; hidden states được giữ ở nhánh mới là output hiện tại của block 2. Prompt và original RoPE positions được giữ. Layers 0–2 có KV cache đầy đủ; layers 3–29 có KV cache compact. Vì vậy các token sinh vẫn nhìn context đầy đủ ở ba layers đầu.

## Chạy trên Kaggle

Notebook: `notebooks/kaggle_smolvlm_top32_score22_prune_input_vs_after2.ipynb`. Push code lên nhánh `smolvlm-attention` trước khi clone notebook. Bật Internet, hai GPU và attach cache cùng export attention test đã giải nén.

- `ATTENTION_INPUT`: bắt buộc trỏ tới export probability `[16817,30,384]`; không dùng teacher log-target train/validation.
- `CACHE_INPUT`: để trống cho resolver cache hiện có, hoặc điền thư mục cache Input. Không copy cache sang working.
- `PREPARED_DIR`: để trống để tải prepared data public đã pin, hoặc điền thư mục local hợp lệ.
- `RUN_MODE='auto'`: tạo mới hoặc tiếp tục/reuse output; `measure` từ chối output đã tồn tại; `resume` yêu cầu output cũ.

Run All: validate/alignment/selection/metric preflight → numerical preflight và stress từng GPU → mở gate khi cả hai pass → generation → metrics → ZIP. Stress ép đủ 512 token chỉ trong cấu hình probe; production giữ EOS native. OOM dừng, không retry bằng batch/precision khác. Một worker lỗi thì worker còn lại bị dừng, các batch đã commit được giữ.

Protocol giữ batch 64/GPU, đủ 16.817 mẫu test (batch cuối 49), FP32, autocast và TF32 tắt, frozen/eval, greedy `num_beams=1`, `do_sample=False`, repetition penalty 1.0, tối đa 512 token mới. Đây là benchmark **chất lượng QA**, chưa có protocol latency/FLOPs.

Có thể chạy CLI từ root repository:

```bash
python -m smolvlm_topk_deferred prepare --output /path/output --mode auto --attention-input /path/export --cache-input /path/cache --prepared-dir /path/prepared
python -m smolvlm_topk_deferred run --output /path/output
python -m smolvlm_topk_deferred score --output /path/output
python -m smolvlm_topk_deferred bundle --output /path/output
```

## Output và resume

Output mặc định `smolvlm_top32_score22_input_vs_after2_outputs` độc lập với benchmark cũ. Hai cấu hình có journal, predictions và metadata riêng, dùng chung selection được checksum. `summary.json` và `summary.csv` chứa BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr và delta **after2 − input**. Metadata dùng `prune_after_layer_index: null` cho input và `2` cho after2. Đối chứng input là Top-32 trước block 0, không phải base không prune.

ZIP `top32_score22_input_vs_after2_results.zip` chứa kết quả và hashes, không chứa model weights, vision cache hoặc attention export. ZIP kết quả không phải bộ resume hoàn chỉnh. Muốn tiếp tục qua session cần giữ toàn bộ output chưa hoàn tất, rồi resolve lại inputs cùng nội dung. Không đổi source/dependencies của run đang resume; mismatch bị từ chối. Không có upload tự động hoặc yêu cầu publishing credentials.

## Kiểm chứng

Tests model native nhỏ bảo vệ all-keep/native parity, cached/slow-reference parity, gather sau output block 2, original positions, cache lengths từng layer, padding, EOS, max-token budget, RNG/weights và việc không gọi Stage 1/quota. Có tests state/selection, preflight và lifecycle offline; chạy nhóm `tests/test_deferred_*.py` cùng các regression Top-K hiện có.

Kết quả local: **245 tests liên quan pass** trong 100,36 giây, gồm regression benchmark/exporter/MVPruner/teacher và tests mới. Sau đó bổ sung kiểm tra native decoder chạy đủ 512 token, xác nhận cache tăng đúng 511 slots tại từng layer; cả **10 tests decoder pass** trong 9,06 giây. Tổng cộng 246 test cases khác nhau đã pass. Smoke lifecycle dùng model native nhỏ với 384 visual tokens, cache FP16 giả, hai generator, resume độc lập, metrics COCO thật qua Java/METEOR và archive; không mock phép tính model hoặc metrics.

Đã kiểm tra JSON/config, 12 file Python mới và 7 cell Python notebook, không có saved outputs; whitespace và `git diff --check` pass. Các file tracked cũ không thay đổi. Không chạy extraction/generation production, commit hoặc push.

Chưa chạy checkpoint thật trên CUDA trong đợt triển khai local. Chưa xác minh batch 64/GPU, VRAM hoặc thời gian trên Kaggle; stress thực tế là bước bắt buộc trước generation toàn test. Không suy chất lượng QA hoặc tốc độ GPU từ CPU tests.
