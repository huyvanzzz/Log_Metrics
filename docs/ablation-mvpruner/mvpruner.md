### iả thuyết: Model dần nhận diện được thông tin liên quan đến tác vụ qua các layer (MVPruner)

**Đo những gì:**

- **Task-related view recognition:** Attention từ instruction tokens đến từng camera tại các decoder layers; kiểm tra camera có attention cao nhất có đúng camera liên quan không.
- **Attention entropy:** Độ phân tán của text-to-vision attention qua các layer.

**Đo như nào:** Đo toàn bộ layer, visualize ở các layer

- **Task-related view recognition:** Trung bình trên toàn bộ instruction tokens, dùng mean attention heads theo Q. Chọn các câu hỏi có chỉ rõ camera liên quan, từ DriveLM. Camera có attention score cao nhất được xem là camera model nhận diện. Nếu camera đó trùng với camera được câu hỏi chỉ định thì tính là đúng. ⇒ Tính accuracy.
- **Attention entropy:** Đo bằng loss entropy trên tất cả token (có chuẩn hóa softmax của llm)
- **Dữ liệu: Lấy sub data và đo trung bình**