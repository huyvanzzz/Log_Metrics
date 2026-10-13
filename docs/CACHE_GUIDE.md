# Hướng Dẫn Quản Lý Cache & Môi Trường Làm Việc Trên Ổ D

Tài liệu này ghi chú lại cấu hình hệ thống đã thiết lập nhằm đảm bảo **100% dữ liệu cache, model weights và thư viện không bao giờ ghi đè làm đầy ổ C**.

---

## 1. Các thư mục Cache trên ổ D

Hệ thống đã được thiết lập các biến môi trường người dùng (User Environment Variables) trong Registry Windows (`HKCU\Environment`):

| Biến Môi Trường | Giá Trị Đường Dẫn | Mục Đích |
|---|---|---|
| `HF_HOME` | `D:\huggingface_cache` | Chứa cache model weights, datasets, hub checkpoints của Hugging Face |
| `TORCH_HOME` | `D:\torch_cache` | Chứa pre-trained weights và checkpoints của PyTorch / Torch Hub |
| `PIP_CACHE_DIR` | `D:\pip_cache` | Chứa các file `.whl` và HTTP cache khi tải thư viện qua `pip` |
| `OLLAMA_MODELS` | `D:\ollama` | (Đã có từ trước) Lưu trữ weights của Ollama |

---

## 2. Quy trình thiết lập môi trường mới an toàn (Ví dụ: `understanding_lp`)

Khi bạn hoặc trợ lý AI cài đặt bất kỳ repository mới nào trên ổ `D:\AD\...`:

### Bước 1: Luôn tạo Virtual Environment (`.venv`) trên ổ D
Tuyệt đối không chạy `pip install` ở môi trường global của Python.
```bash
# Di chuyển vào thư mục project
cd D:\AD\understanding_lp

# Tạo virtual environment ngay tại project (nằm trên ổ D)
python -m venv .venv
```

### Bước 2: Kích hoạt `.venv`
- **Trên PowerShell**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **Trên Command Prompt / Batch**:
  ```cmd
  call .venv\Scripts\activate.bat
  ```

### Bước 3: Cài đặt packages
Khi đã ở trong `.venv`, `pip` sẽ:
1. Đọc cache tải về từ `D:\pip_cache` (không tốn dung lượng ổ C).
2. Cài toàn bộ thư viện vào `D:\AD\understanding_lp\.venv\Lib\site-packages` (nằm trên ổ D).

```bash
pip install -r requirements.txt
```

---

## 3. Kiểm tra biến môi trường bất cứ lúc nào

Nếu muốn kiểm tra lại biến môi trường hiện tại:
- **PowerShell**:
  ```powershell
  [Environment]::GetEnvironmentVariable('HF_HOME', 'User')
  [Environment]::GetEnvironmentVariable('TORCH_HOME', 'User')
  [Environment]::GetEnvironmentVariable('PIP_CACHE_DIR', 'User')
  ```
- **Command Prompt**:
  ```cmd
  reg query HKCU\Environment /v HF_HOME
  reg query HKCU\Environment /v TORCH_HOME
  reg query HKCU\Environment /v PIP_CACHE_DIR
  ```
