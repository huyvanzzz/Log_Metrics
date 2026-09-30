# Kế hoạch Triển khai Web Dashboard Trực quan hóa & So sánh Mô hình AD (Đã Verify Toàn Diện)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng Web Dashboard hoàn chỉnh (FastAPI backend + React/Vite + Apache ECharts frontend) trực quan hóa và so sánh các mô hình xe tự hành (EM_VLM4AD) dựa trên 3 loại file: `train_progress*.jsonl`, `metrics_display.json`, và `benchmark.json`. Hệ thống đáp ứng đầy đủ 100% các yêu cầu bắt buộc của người dùng.

> [!IMPORTANT]
> **Ràng buộc ổ đĩa nghiêm ngặt:** Toàn bộ môi trường ảo Python (`.venv`), thư viện pip cache, thư mục `node_modules` và npm cache **PHẢI nằm 100% trên ổ D:** (`d:\AD\Visualize\.venv`, `d:\AD\Visualize\.cache\pip`, `d:\AD\Visualize\.cache\npm`). Tuyệt đối không lưu bất kỳ file cài đặt hay cache nào vào ổ C:.

---

## Bảng Đối Chiếu Kiểm Định Yêu Cầu (Verification Checklist)

| Yêu cầu của Người dùng | Trạng thái thiết kế & Giải pháp trong Plan |
| :--- | :--- |
| **1. Trực quan hóa 3 loại file** (`train_progress*.jsonl`, `metrics_display.json`, `benchmark.json`) | ✅ Backend quét đệ quy thư mục `Log/`, parse và nạp chuẩn xác cả 3 loại file. |
| **2. Tên phương pháp lấy theo folder** (ví dụ `EM_VLM4AD_Adapter`) | ✅ Tự động format tên dạng `<Folder_Cha>_<Folder_Con>` (`EM_VLM4AD_Adapter`, `EM_VLM4AD_Freeze`), mở rộng tự động cho folder mới. |
| **3. Tự động gộp `train_progress` thông minh, không gộp dồn trùng lặp** | ✅ Thuật toán deduplicate theo khóa `(stage, global_step)`, bản ghi resume sau ghi đè bản ghi cũ (đã kiểm tra thực tế: bước 64015-64214 ở Adapter và 42677-44176 ở Freeze được làm sạch 100%). |
| **4. Ảnh visualize loss cả 2 stage trong cùng 1 ảnh (đơn lẻ & tập hợp max 5)** | ✅ Vẽ cả Stage 1 (Align) và Stage 2 (Finetune) trên cùng 1 biểu đồ/ảnh; có vạch phân tách và phân vùng màu stage; hỗ trợ xem đơn lẻ từng model hoặc tick chọn tối đa 5 model đồng thời. |
| **5. Bảng benchmark so sánh tất cả** | ✅ Bảng so sánh toàn diện mọi chỉ số (E2E mean/p50/p95, Gen mean/p50/p95, avg ms/token, tokens/s, peak VRAM, GPU). Xử lý an toàn khi Freeze chưa có file benchmark. |
| **6. Bảng metrics so sánh tất cả** | ✅ Bảng đối đầu 7 metrics (BLEU-1..4, METEOR, ROUGE-L, CIDEr) + Cột điểm **Overall**, tự động highlight Best Score. |
| **7. Ảnh metrics + benchmark (Scatter plot chấm màu: X = avg ms/token, Y = Overall)** | ✅ Biểu đồ phân tán ECharts: Trục X = `average_generation_ms_per_output_token`, Trục Y = $\text{Overall} = \frac{\text{BLEU4} + \text{METEOR} + \text{ROUGE\_L} + \text{CIDEr}/10}{4}$. Mỗi model là 1 chấm tròn màu sắc riêng biệt. |
| **8. Cơ chế bấm vào line hoặc chấm để hiển thị thông tin** | ✅ Click/Hover Inspector: Bấm vào đường line loss hoặc chấm scatter sẽ mở modal/tooltip chi tiết hiển thị toàn bộ thông số, metrics, loss, step, latency của model đó. |
| **9. Hiển thị nhiều cái cùng lúc trong 1 ảnh để dễ so sánh** | ✅ Khung nhìn tổng hợp **Composite Comparison Board** + Nút bấm **"📸 Xuất 1 Ảnh So Sánh (HD PNG 2x/3x)"** chụp trọn vẹn toàn bộ biểu đồ và bảng vào 1 file ảnh duy nhất. |
| **10. Môi trường ảo và thư viện 100% trên ổ D:** | ✅ Python `.venv` tại `d:\AD\Visualize\.venv`, pip cache `d:\AD\Visualize\.cache\pip`, npm cache `d:\AD\Visualize\.cache\npm`, node_modules tại `d:\AD\Visualize\frontend\node_modules`. |

---

## Chi tiết các Task Triển khai

### Task 1: Khởi tạo Môi trường Ảo và Cấu hình Cache trên Ổ D:
**Files:**
- Create: `d:\AD\Visualize\.gitignore`
- Config: `d:\AD\Visualize\.cache\pip`
- Config: `d:\AD\Visualize\.cache\npm`
- Path: `d:\AD\Visualize\.venv`

- [ ] **Step 1: Tạo thư mục cache trên ổ D**
  ```powershell
  New-Item -ItemType Directory -Force -Path "d:\AD\Visualize\.cache\pip"
  New-Item -ItemType Directory -Force -Path "d:\AD\Visualize\.cache\npm"
  ```
- [ ] **Step 2: Tạo virtual environment Python tại `d:\AD\Visualize\.venv`**
  ```powershell
  python -m venv "d:\AD\Visualize\.venv"
  ```
- [ ] **Step 3: Cài đặt các gói phụ thuộc backend với PIP_CACHE_DIR trỏ vào ổ D**
  ```powershell
  $env:PIP_CACHE_DIR = "d:\AD\Visualize\.cache\pip"
  & "d:\AD\Visualize\.venv\Scripts\python.exe" -m pip install --upgrade pip
  & "d:\AD\Visualize\.venv\Scripts\python.exe" -m pip install fastapi uvicorn pydantic python-multipart pytest
  ```
- [ ] **Step 4: Kiểm tra xác nhận venv và các gói đã cài đặt thành công trên ổ D**
  ```powershell
  & "d:\AD\Visualize\.venv\Scripts\python.exe" -c "import fastapi, uvicorn; print('Backend environment OK on D: drive')"
  ```

---

### Task 2: Xây dựng Module Backend Parser & Smart Merge Engine
**Files:**
- Create: `d:\AD\Visualize\backend\__init__.py`
- Create: `d:\AD\Visualize\backend\parser.py`
- Create: `d:\AD\Visualize\backend\main.py`
- Test: `d:\AD\Visualize\backend\test_parser.py`

- [ ] **Step 1: Viết test cho logic parser và gộp log thông minh (`test_parser.py`)**
  - Test phát hiện model: `discover_models` tìm thấy `EM_VLM4AD_Adapter` và `EM_VLM4AD_Freeze`.
  - Test gộp thông minh: xác nhận số bước sau gộp của `Adapter` là 106,690 steps, không trùng lặp các bước 64015-64214; `Freeze` có đủ 2 stage (`align` 42,676 steps và `finetune` tới 64,014 steps).
  - Test tính `Overall`: đúng theo công thức $\frac{\text{BLEU4} + \text{METEOR} + \text{ROUGE\_L} + \text{CIDEr}/10}{4}$.
- [ ] **Step 2: Hiện thực `backend/parser.py`**
  - Tự động quét thư mục `Log/` đệ quy.
  - Đọc tuần tự các file `train_progress*.jsonl`, gộp theo key `(stage, global_step)` ghi đè khi resume.
  - Trích xuất các sự kiện `validation_end` (val_loss, best_val_loss).
  - Parse `metrics_display.json` và `benchmark.json` (xử lý an toàn khi thiếu file).
- [ ] **Step 3: Hiện thực REST API `backend/main.py`**
  - `GET /api/models`: Danh sách các model và metadata trạng thái.
  - `GET /api/metrics`: Bảng metrics và điểm Overall.
  - `GET /api/benchmark`: Bảng benchmark chi tiết.
  - `GET /api/tradeoff`: Danh sách tọa độ Scatter plot (X: ms/token, Y: Overall).
  - `GET /api/loss`: Dữ liệu chuỗi loss đa giai đoạn (hỗ trợ lọc theo `models=...`).
- [ ] **Step 4: Chạy test xác nhận backend hoạt động chính xác**
  ```powershell
  & "d:\AD\Visualize\.venv\Scripts\python.exe" -m pytest backend/test_parser.py -v
  ```

---

### Task 3: Khởi tạo Dự án Frontend (React + Vite + ECharts trên Ổ D:)
**Files:**
- Create: `d:\AD\Visualize\frontend\package.json`
- Config: npm cache trỏ vào `d:\AD\Visualize\.cache\npm`

- [ ] **Step 1: Khởi tạo Vite React trong `d:\AD\Visualize\frontend`**
  ```powershell
  $env:npm_config_cache = "d:\AD\Visualize\.cache\npm"
  npm create vite@latest frontend -- --template react
  ```
- [ ] **Step 2: Cài đặt các thư viện Frontend vào `d:\AD\Visualize\frontend\node_modules`**
  ```powershell
  npm --prefix "d:\AD\Visualize\frontend" --cache "d:\AD\Visualize\.cache\npm" install echarts echarts-for-react lucide-react html-to-image clsx tailwind-merge
  npm --prefix "d:\AD\Visualize\frontend" --cache "d:\AD\Visualize\.cache\npm" install -D tailwindcss postcss autoprefixer
  ```
- [ ] **Step 3: Cấu hình TailwindCSS và Vite Proxy**
  - Cấu hình proxy `/api` trỏ về `http://127.0.0.1:8000`.

---

### Task 4: Xây dựng Component Visualize Loss Đa Giai Đoạn (Dual-Stage Loss Visualizer)
**Files:**
- Create: `d:\AD\Visualize\frontend\src\components\LossVisualizer.jsx`

- [ ] **Step 1: Giao diện điều khiển (Controls)**
  - Checkbox chọn tối đa 5 model đồng thời (tự động khóa khi đã chọn đủ 5).
  - Chuyển đổi hiển thị: "Từng model (Single View)" hoặc "Tập hợp (Comparison View)".
  - Chuyển đổi trục X: `Global Step` hoặc `Epoch`.
  - Chuyển đổi trục Y: `Linear` hoặc `Logarithmic`.
  - Toggle hiển thị đường `Validation Loss`.
- [ ] **Step 2: Cấu hình biểu đồ ECharts 2 Stage trong cùng 1 ảnh**
  - Stage 1 (Align): vẽ nét đứt (dashed).
  - Stage 2 (Finetune): vẽ nét liền (solid).
  - Đánh dấu vạch phân cách `Stage Transition: Align → Finetune`.
  - Điểm Validation loss hiển thị dạng biểu tượng kim cương (diamonds).
- [ ] **Step 3: Tương tác Click & Hover**
  - Hover hiển thị Tooltip: Model, Stage, Step, Epoch, Train Loss, Val Loss, Tốc độ (step/s).
  - Click vào đường line kích hoạt sự kiện chọn model hiển thị chi tiết (Model Inspector).

---

### Task 5: Xây dựng Bảng So Sánh Metrics & Bảng Benchmark
**Files:**
- Create: `d:\AD\Visualize\frontend\src\components\MetricsTable.jsx`
- Create: `d:\AD\Visualize\frontend\src\components\BenchmarkTable.jsx`

- [ ] **Step 1: Xây dựng `MetricsTable.jsx`**
  - Hiển thị tất cả model tìm thấy.
  - Cột: Phương pháp | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr | **Overall**.
  - Hiển thị badge công thức: $(BLEU4 + METEOR + ROUGE\_L + CIDEr/10)/4$.
  - Tự động highlight giá trị Best Score màu xanh.
- [ ] **Step 2: Xây dựng `BenchmarkTable.jsx`**
  - Hiển thị tất cả model tìm thấy.
  - Cột: Phương pháp | E2E Mean | E2E P50 | E2E P95 | Gen Mean | Gen P50 | Gen P95 | Avg ms/token | Tokens/s | Peak VRAM | GPU.
  - Xử lý trạng thái an toàn: Với model chưa có benchmark (như Freeze), hiển thị badge `Chưa đo benchmark`.

---

### Task 6: Xây dựng Biểu đồ Scatter Plot Trade-off (Tốc độ vs Điểm Overall)
**Files:**
- Create: `d:\AD\Visualize\frontend\src\components\TradeoffScatter.jsx`

- [ ] **Step 1: Cấu hình biểu đồ phân tán ECharts**
  - Trục X: `average_generation_ms_per_output_token` (ms/token, càng nhỏ càng tốt).
  - Trục Y: `Overall Score` (càng lớn càng tốt).
  - Mỗi model là một chấm tròn to với màu sắc nhận diện riêng, kèm nhãn tên model.
- [ ] **Step 2: Tương tác Click & Hover**
  - Hover hiển thị Card: Chi tiết độ trễ, tokens/s, điểm Overall, breakdown 4 metrics thành phần.
  - Click vào chấm tròn để chọn model và mở Inspector chi tiết.

---

### Task 7: Xây dựng Khung So Sánh Tổng Hợp & Cơ chế Chụp 1 Ảnh HD (1-Click Snapshot)
**Files:**
- Create: `d:\AD\Visualize\frontend\src\components\CompositeComparisonBoard.jsx`
- Create: `d:\AD\Visualize\frontend\src\components\ModelInspectorModal.jsx`
- Modify: `d:\AD\Visualize\frontend\src\App.jsx`

- [ ] **Step 1: Bố cục Composite Comparison Board**
  - Bọc tất cả các khối (Loss Chart, Scatter Plot, Metrics Table, Benchmark Table) trong một container duy nhất có ID định danh.
- [ ] **Step 2: Tích hợp công cụ xuất ảnh `html-to-image`**
  - Nút bấm: **"📸 Xuất 1 Ảnh So Sánh (HD PNG)"**.
  - Tự động chụp toàn bộ vùng container với độ phân giải cao (`pixelRatio: 2` hoặc `3`) nền tối chuẩn dark mode, lưu về file PNG sắc nét.
- [ ] **Step 3: Xây dựng `ModelInspectorModal.jsx`**
  - Hiển thị modal chi tiết khi người dùng click vào bất kỳ đường line hay chấm tròn nào.

---

### Task 8: Tích hợp Toàn diện, Tạo Script 1-Click Khởi Chạy & Kiểm Thử Trình Duyệt
**Files:**
- Create: `d:\AD\Visualize\run_dashboard.bat`

- [ ] **Step 1: Viết script `run_dashboard.bat`**
  - Tự kích hoạt môi trường ảo `d:\AD\Visualize\.venv`, chạy backend FastAPI và frontend Vite song song.
- [ ] **Step 2: Kiểm thử trực tiếp bằng `browser_subagent`**
  - Mở `http://localhost:5173`.
  - Kiểm tra biểu đồ loss hiển thị cả 2 stage.
  - Kiểm tra chọn tối đa 5 model.
  - Kiểm tra bảng metrics và bảng benchmark.
  - Kiểm tra scatter plot trade-off và công thức Overall.
  - Kiểm tra bấm vào line/chấm tròn mở thông tin chi tiết.
  - Kiểm tra bấm nút xuất 1 ảnh so sánh.
  - Kiểm tra dung lượng ổ C: không bị thay đổi.
