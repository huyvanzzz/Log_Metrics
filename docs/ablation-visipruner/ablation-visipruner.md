### **14. Giả thuyết: Quá trình xử lý thị giác có ba giai đoạn — shallow attention sinks, middle cross-modal fusion và deep language completion (VisiPruner)**

**Dữ liệu: Một phần dữ liệu (1/4 là hợp lý nhất) nhưng nên lấy phân phối**

**Giả thuyết:** Quá trình xử lý thông tin thị giác trong LLM gồm ba giai đoạn:

- **Shallow layers:** Nhận diện tác vụ từ instruction. Visual tokens chủ yếu đóng vai trò attention sinks.
- **Middle layers:** Tích hợp thông tin ảnh vào text, tập trung vào một số ít visual tokens quan trọng.
- **Deep layers:** Thông tin ảnh đã được tích hợp; model chủ yếu hoàn thiện câu trả lời bằng ngôn ngữ.

**2. Phương pháp phân tích**

**2.1. Shallow layers – Attention có thực sự quan trọng?**

**Cách đo:** Mask attention từ text đến visual tokens tại các shallow layers, sau đó đánh giá performance.

| Can thiệp | Kết quả |
| --- | --- |
| Mask Top-10% attention (layers 1–2) | Performance gần như không đổi |
| Mask 90% visual tokens còn lại | 72.6 → 71.5 |
| Mask toàn bộ visual tokens tại layer 1 | 72.6 → 65.2 |
| Gom toàn bộ visual attention vào một token | Performance gần như không đổi |

**Kết luận:** Visual tokens tại shallow layers chủ yếu giúp ổn định phân phối attention, không đóng góp đáng kể vào quá trình truyền thông tin ảnh sang text.

**2.2. Middle layers – Visual tokens nào quan trọng?**

**Cách đo:** Mask Top-10% và Bottom-10% visual tokens theo attention tại layers 9–15.

| Can thiệp | GQA |
| --- | --- |
| Baseline | 61.95 |
| Mask Top-10% | 54.09 |
| Mask Bottom-10% | 61.93 |

Chỉ giữ Top-5% visual tokens theo attention vẫn duy trì performance trung bình 72.6 → 71.3.

**Kết luận:** Middle layers thực hiện cross-modal fusion, nhưng chỉ cần một tập visual tokens tương đối nhỏ.

**2.3. Deep layers – Khi nào có thể loại bỏ visual tokens?**

**Cách đo:** So sánh hai phương pháp: (Loại tất)

- **Skipping:** Bỏ xử lý visual tokens ở một layer nhưng vẫn giữ chúng cho các layer sau.
- **Discarding:** Loại bỏ hoàn toàn visual tokens từ một layer trở đi.

Trên LLaVA-v1.5 7B:

| Can thiệp | GQA |
| --- | --- |
| Baseline | 61.95 |
| Discard từ layer 20 | 59.13 |
| Skip layer 20 | 61.66 |
| Discard từ layer 26 | 61.91 |

**Kết luận:** Visual tokens không cần được xử lý liên tục ở mọi layer. Từ các deep layers, chúng có thể được loại bỏ khi thông tin đã được tích hợp vào text.