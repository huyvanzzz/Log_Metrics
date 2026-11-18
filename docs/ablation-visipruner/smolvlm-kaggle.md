# Ablation VisiPruner trên SmolVLM

## Chạy trên Kaggle

Đưa mã lên nhánh Git mà Kaggle đọc được, rồi dùng hai notebook theo thứ tự.
Cell cấu hình mặc định dùng nhánh `smolvlm-attention-nguyet`.

1. `notebooks/kaggle_smolvlm_visipruner_ablation.ipynb` chạy lượt 01–12,
   gồm baseline, toàn bộ shallow và nhóm middle 9–15. Bật GPU và Internet.
   Giữ cùng `DATA_DIR` để dùng lại ảnh từ MVPruner nếu đã tải trong phiên.
   Có thể khai báo tên Kaggle Secret quyền Read ở `HF_TOKEN_SECRET`.
   Sau cell xác minh cuối, lưu output của notebook, bao gồm cả thư mục
   `visipruner_ablation_outputs/`.
2. `notebooks/kaggle_smolvlm_visipruner_ablation_part2.ipynb` chạy lượt 13–23,
   gồm middle 18–24 và deep. Notebook chấm điểm và vẽ hình cho cả 23 lượt.
   Trong phiên mới, gắn output phần 1 làm Kaggle Input. Nếu tìm thấy nhiều
   bản, đặt `PART1_OUTPUT` thành thư mục chứa `manifest.json` của bản cần dùng.
   Bước khôi phục sao chép dự đoán, thứ hạng và cache đặc trưng của phần 1.
   Dùng cùng loại GPU, phiên bản PyTorch, mã và cấu hình sinh như phần 1.

Nếu chạy liền cả 23 cấu hình trong một phiên, dùng
`notebooks/kaggle_smolvlm_visipruner_ablation_full.ipynb`. Bản full gồm chuẩn bị
dữ liệu, đo, chấm điểm và xuất hình.

Baseline chạy trên GPU 0 và lưu thứ hạng token ảnh cho các lượt mask. Sau
baseline, mỗi cell mô tả tối đa hai biến thể. Trên T4 ×2, một biến thể chạy trên
GPU 0 và biến thể kia chạy trên GPU 1. Trên một GPU, hai biến thể chạy tuần tự.
Có thể chạy lại cell để tiếp tục từng biến thể từ bộ ảnh cuối đã lưu. Kết quả
từ runner một GPU trước đây có thể dùng lại nếu checkpoint, dữ liệu, mã decoder,
cấu hình sinh, phiên bản PyTorch và loại GPU vẫn khớp. Đợi cell đang chạy dừng
trước khi cập nhật repo trong cùng phiên Kaggle.
Mỗi lượt in số thứ tự trên 23, số câu đã xong và thanh tiến độ. Log được lưu
trong `logs/<variant>.log`. Dự đoán được lưu từng bộ ảnh trong
`predictions/<variant>/`. Cache đặc trưng đã qua connector nằm trong
`projected/`, dùng chung cho 23 biến thể. Phần 2 dùng cache này trực tiếp.
Các bước chấm điểm và vẽ có thể chạy khi tắt GPU.

## Quy ước can thiệp

- Cùng 4.204 câu cho tất cả 23 cấu hình. Lấy seed 42, phân tầng theo loại câu
  hỏi và dạng đáp án Yes/No. Greedy decoding FP32, mặc định tối đa 128 token
  mới. Bảng điểm ghi số câu chạm giới hạn.
- Xếp hạng 384 token ảnh theo attention trung bình trên head và token câu hỏi
  ở baseline. Mask sau softmax trong prefill, bắt đầu từ token đầu tiên thuộc
  câu hỏi. Biến thể bảo toàn mass chỉ phân bổ lại trọng số dành cho ảnh.
  Nếu trọng số ảnh còn lại không vượt quá `1e-8` nhưng mass gốc dương, dùng
  softmax có điều kiện trên logits của các token ảnh còn lại. Hàng có mass
  gốc bằng 0 giữ nguyên 0. Lưu tỷ lệ hàng cần dùng nhánh xử lý số học.
- Merge chạy ba seed. Mỗi seed ứng với một camera khác nhau và một vùng cao,
  giữa hoặc thấp của lưới 8 × 8. Giữ patch cố định trong cả lượt. Manifest
  lưu camera và chỉ số patch theo hệ 0-based. Log chuẩn bị in hệ 1-based.
- Skip tại layer L cho visual hidden state đi qua nguyên trạng và đưa lại vào
  chuỗi tại L+1. Layer L chỉ tính attention, FFN và KV cho text. Discard từ L
  loại visual hidden states và visual KV ở L và các layer sau. Giữ vị trí
  RoPE gốc của text và giữ visual KV ở các layer trước L.

## Đầu ra

`scores.csv` và `scores.json` có BLEU-4, METEOR, ROUGE-L, CIDEr, Overall, tỷ lệ
exact-match Yes/No, số câu chạm giới hạn, lượng visual mass bị lấy đi trước
khi phân bổ lại và tỷ lệ dùng nhánh ổn định số học. Công thức Overall dùng
cùng cách tổng hợp của dashboard repo. Bộ chấm điểm COCO cần Java cho PTBTokenizer
và METEOR.

Thư mục `figures/` có bốn hình độc lập, mỗi hình xuất PNG và PDF.

1. **Shallow interventions** hiển thị ΔOverall của các phép mask và ba điểm
   merge. Cột merge là trung bình, thanh sai số là độ lệch chuẩn mẫu qua seed.
2. **Middle masking controls** đặt hai dải 9–15 và 18–24 cạnh nhau cho năm chế
   độ mask, gồm các cặp top/bottom có và không bảo toàn visual mass.
3. **Deep skip versus discard** hiển thị hai biến thể cạnh nhau tại layer 20,
   23 và 26, theo ΔOverall.
4. **Middle removed visual mass** hiển thị tỷ trọng visual mass ban đầu bị
   lấy đi ở mỗi nhóm middle trước khi phân bổ lại.
