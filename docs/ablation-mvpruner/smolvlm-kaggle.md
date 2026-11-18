# Thí nghiệm attention theo MVPruner với SmolVLM

## Chạy trên Kaggle

1. Đưa mã lên nhánh Git mà Kaggle có thể đọc. Mở
   `notebooks/kaggle_smolvlm_mvpruner_ablation.ipynb`, bật Internet và chọn một GPU.
   Notebook dùng PyTorch CUDA có sẵn của Kaggle.
2. Notebook mặc định lấy dữ liệu từ
   [DriveLM nuScenes trên Hugging Face](https://huggingface.co/datasets/minhdang0901/drivelm-nuscenes-v1-0).
   Cell chuẩn bị tải `multi_frame/multi_frame_test.json` và đúng 1.134 ảnh
   mà file test tham chiếu, vào `/kaggle/working/drivelm-nuscenes-v1-0`.
   Phiên bản dataset được ghim trong `smolvlm_attention/mvpruner_ablation/data.py`.
   Nếu dùng Kaggle Input, đặt `DATA_SOURCE=kaggle-input` và để trống
   `DATA_DIR`, `TEST_JSON` để tự tìm dưới `/kaggle/input`. Nếu có nhiều đường
   dẫn trùng tên, đặt các biến đường dẫn theo thông báo của CLI.
   Chỉnh `SOURCE_BRANCH` nếu mã nằm trên nhánh khác.
3. Chạy lần lượt các cell cấu hình, thiết lập, `prepare`, `measure` và `plot`.
   Mặc định dùng `finetune_best.pt` của checkpoint base được ghim trong
   `smolvlm_attention/config.py`. Nếu đã gắn bản sao checkpoint đáng tin cậy,
   có thể thêm `--checkpoint /kaggle/input/.../finetune_best.pt` vào lệnh đo.

`prepare` kiểm tra 16.817 câu và 1.134 ảnh. Với bản test trong repo, kết quả
là 8.858 câu có tag một camera, 4.739 câu chỉ hướng, 1.155 câu có nhiều camera
và 2.065 câu không có nhãn camera. Mặc định chọn câu có tag hợp lệ đầu tiên làm
minh họa. Thêm `--example-id` vào lệnh `prepare` để chọn một dòng test theo
chỉ số từ 0 trước khi đo.

`measure` đối chiếu attention trích xuất với attention native trên một câu
trước khi chạy. Các câu dùng chung bộ sáu ảnh dùng lại đặc trưng của bộ đó.
Worker chạy FP32, lưu kết quả sau mỗi bộ ảnh và giảm batch khi hết VRAM.
Chạy lại cell đo với cùng đường dẫn kết quả để tiếp tục phần còn thiếu.
Nếu Kaggle xóa thư mục làm việc khi kết thúc phiên, cần lưu cả thư mục kết quả
từ phiên trước rồi khôi phục vào `OUTPUT_DIR`. Giữ cùng mã nguồn, checkpoint,
manifest, phiên bản thư viện và loại GPU để tiếp tục. Có thể chạy riêng cell
`plot` từ kết quả đo đã hoàn tất, kể cả trên máy không có GPU.

## Phép đo và đầu ra

Các query là token thuộc nội dung câu hỏi gốc. Với mỗi query và layer, áp dụng
causal softmax theo các key hợp lệ, lấy trung bình đều các head, rồi giữ các
giá trị tương ứng với 384 token ảnh.

- Độ chính xác camera dùng trung bình qua query của câu hỏi và qua 64 token
  ảnh mỗi camera, rồi lấy camera có điểm cao nhất. Hai nhóm nhãn có đường chuẩn
  camera phổ biến nhất riêng. Khi điểm bằng nhau, chọn camera xuất hiện trước
  trong thứ tự camera của prompt. Bảng CSV gồm cả accuracy theo từng camera.
- Token-level entropy chuẩn hóa phân bố ảnh và tính Shannon entropy riêng cho
  từng query, sau đó lấy trung bình các query trong câu.
- Question-level entropy lấy trung bình attention trên ảnh qua các query trước.
  Sau đó chuẩn hóa vector kết quả thành phân bố ảnh và tính Shannon entropy.
  Query có tổng attention lên ảnh lớn hơn đóng góp nhiều hơn vào phân bố này.
- Cả hai cách chia entropy cho `ln(384)`, rồi lấy trung bình đều các câu hỏi
  trên toàn bộ test. Câu có nhiều query và câu có ít query có cùng trọng số.

Kết quả mặc định nằm trong `/kaggle/working/mvpruner_ablation_outputs/`.
Bốn hình độc lập trong `figures/` đều có bản PNG và PDF.

- `view_recognition_accuracy` có kết quả theo layer cho hai nhóm nhãn.
- `token_level_attention_entropy` và `question_level_attention_entropy`
  có chung trục entropy từ 0 đến 1.
- `text_to_vision_attention_maps` có sáu camera tại layer 2, 15 và 30, đánh số
  layer từ 1. Heatmap dùng giá trị attention trung bình theo câu hỏi, chung
  thang màu log khi có nhiều giá trị dương khác nhau. Vòng tròn đánh dấu tọa độ
  trong object tag trên ảnh tương ứng.

`metrics_by_layer.csv` chứa dữ liệu vẽ các đường theo layer.
`summary.json` chứa số mẫu, các baseline và ID minh họa.
`manifest.json`, `measurement.json` và `frames/` chứa trạng thái đo.
