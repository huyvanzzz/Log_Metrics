# Predictor preserve mass rồi hard-prune — test riêng

Nguồn ablation: `origin/smolvlm-attention-nguyet`, commit
`c3157787fe63640c56326040cb499d0929bcd795`, `visipruner_ablation/decoder.py`.
Nguồn gốc chỉ mask/rescale attention, không xóa token. Biến thể này thêm gather
sau toàn bộ block can thiệp, không đổi predictor hoặc training loss.

Hai notebook `kaggle_smolvlm_predictor_before_preserve_mass_test.ipynb` và
`kaggle_smolvlm_predictor_after2_preserve_mass_test.ipynb` dùng predictor đã train;
không train lại, không đọc teacher targets. Checkpoint 5 hoặc 10 epoch đều dùng
được nếu architecture, method, base, camera và data/cache identity khớp.

1. Đưa `predictor_best.pt` lên Kaggle Input. Predictor trước LLM và sau index 2
   phải dùng đúng checkpoint tương ứng; không dùng lẫn dù tensor shapes giống nhau.
2. Đặt `PREDICTOR_CHECKPOINT` trong notebook. Đặt `PROJECTED_CACHE_INPUT` tới
   thư mục chứa `manifest.json`, `index.json`, `projected.npy` đã hoàn chỉnh.
   Nếu chưa có projected cache, để trống và cung cấp `CACHE_INPUT` raw cache để
   xây cache mới. Base/prepared có thể dùng đường dẫn Input hoặc tải public đã pin.
3. Run All trên hai GPU. Notebook clone/pull fast-forward trước import, kiểm tra
   checkpoint và dữ liệu, chuẩn bị prompt test, stress longest 64/GPU rồi mới chạy test.
4. Download `predictor_preserve_mass_artifacts.zip` và file SHA256 đi kèm. ZIP
   chứa predictor best, config, logs, predictions và metrics; không chứa base,
   raw/projected cache hoặc targets.

Nếu session bị ngắt, đưa toàn bộ output evaluation đã lưu trở lại working,
giữ `evaluation/` cùng `internal/` và `run.json`, đặt `RESUME=True` và dùng cùng
inputs/code/libraries. Các batch có commit hợp lệ được tái sử dụng, hai GPU
vẫn stress lại. Không tự giảm batch hoặc precision khi OOM. Di chuyển Input
hay internal files làm file receipt không khớp thì dừng; không nới validator.

CLI tương đương:

```bash
python -m smolvlm_predictor_mass prepare --variant before --predictor /path/predictor_best.pt --projected-cache-input /path/cache --output /path/new-output
python -m smolvlm_predictor_mass evaluate --run /path/new-output/run.json
python -m smolvlm_predictor_mass bundle --run /path/new-output/run.json
```

`after2` thay cho `before` khi dùng predictor sau block 2. Cả hai process độc lập,
64 mẫu/GPU, FP32, TF32/autocast tắt, greedy và tối đa 512 token mới.

Đường trước LLM can thiệp tại index 0 rồi gather sau toàn bộ block 0. Đường sau
index 2 chạy prefix 0–2, predictor, can thiệp index 3 rồi gather sau toàn bộ block 3.
Intervention chỉ sửa phần attention nhìn vào visual keys của query từ đầu câu
hỏi tới hết assistant prefix, riêng từng head/query, giữ attention text và tổng
visual mass. 352 visual tokens không được chọn bị xóa sau block can thiệp.
Cache các block prefix vẫn đầy đủ; suffix compact, giữ original RoPE positions.
Không intervention hoặc chọn lại trong decode. Đây là kết hợp ablation Nguyệt
với hard-pruning; không khẳng định tương đương ablation gốc hoặc nhanh hơn.

CPU tests không chứng minh an toàn VRAM/tốc độ/chất lượng cho batch 64 thật.
Kiểm tra CUDA thực tế vẫn cần notebook stress; không chạy production trong đợt code.
