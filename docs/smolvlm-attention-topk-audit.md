# Audit attention từng layer và Top-K SmolVLM

Ngày kiểm tra: 2026-11-23. Snapshot: `c29d5afbb8f86055b08719e7aebcd88f026b32f7`, nhánh `smolvlm-attention`.

## Kết luận

Không phát hiện lỗi tính attention, lệch layer hoặc ánh xạ token trong các phép kiểm chứng CPU bằng model nhỏ. Score khớp native eager reference; indices của cả hai selector khớp reference độc lập. Đây chưa phải xác nhận toàn bộ artifacts của một lượt Kaggle hoặc checkpoint production thật.

Có một khe hở validation: reader chưa xác nhận `layer_ids` và `axes` trong metadata export. Producer hiện tại ghi đúng hai trường này; chưa có bằng chứng export bình thường đang bị đảo layer. Audit không áp dụng sửa đổi.

Chỉ thêm báo cáo này; không sửa source, tests, config, notebook, checkpoint hoặc kết quả cũ.

## Hai score thực sự đo gì?

```text
Export: full 384 visual tokens → native LM + collector từng layer
        → last valid prompt query attention → lưu score
        → chọn global32 từ một layer → gather trước block0 → greedy

Online: 384 → Stage1 giữ196 → blocks0…i
        → question attention tại block i → IRA camera quota → giữ32
        → blocks i+1…cuối → cached greedy; i=2 hoặc15
```

| Thuộc tính | Export → Top-K | Online MVPruner |
|---|---|---|
| Query | Vị trí hợp lệ cuối native prompt | Các vị trí thuộc chính câu hỏi |
| Reduction | Mean heads, riêng từng layer | Mean heads và question tokens |
| Visual keys | 384 | 196 sau Stage1 |
| Chọn | Global stable Top32 | Quota từng camera rồi stable Top-K |
| Thời điểm gather | Trước block0 của lượt sinh mới | Sau block index2/15, tức block thứ3/16 |
| Decode | Không đo/chọn lại | Không đo/chọn lại |

Với tokenizer native đã cache của revision `7e3e67edbbed1bf9888184d9df282b700a323964`, hai câu hỏi thử có prompt dài440/456 tokens. Query cuối đều là **token42, decoded `:` trong `Assistant:`**. Nó không phải token cuối câu hỏi, cũng chưa phải token đáp án được sinh. Question mask online chọn đúng4/20 tokens và decode lại đúng câu hỏi, kể cả newline/object tag; không chứa dấu `:` này. Thay ground truth không đổi input IDs hoặc question mask. Mỗi mẫu có đúng384 image positions.

## Theo dấu implementation

- `smolvlm_attention/data.py:143`: cache features được remap theo camera; `sample_inputs.py:83` và `sample_worker.py:44` nối prepared row/sample ID với global export row. Sorting batch theo độ dài không đổi row identity.
- Visual index là vị trí **sau connector**, camera=`index // 64`, local token=`index % 64`; không phải index1024-token của cache trước connector.
- `collector.py:40` tìm last valid physical position, image positions và RoPE positions từ mask. `collector.py:69` lấy Q đúng query; `:76` áp RoPE/GQA; `:79` scale; `:80` mask padding; `:81` softmax trên toàn bộ keys hợp lệ rồi mean heads; `:82` mới gather visual columns.
- `collector.py:85` đăng ký theo thứ tự decoder blocks, stack riêng từng layer và gỡ hooks bằng cleanup kể cả exception. Collector tính thêm Q cho query cuối và K cho toàn chuỗi; native block vẫn tính projection của nó. Không chạy lại toàn decoder hoặc V projection để đo.
- `smolvlm_topk_benchmark/data.py:79` đọc đúng một layer, không mean nhiều layers. `:36` align bằng sample ID, không dùng thứ tự row. `:11` stable Top-K ưu tiên source index nhỏ khi hòa, rồi sắp indices theo source order.
- `smolvlm_topk_benchmark/generation.py:28` giữ mọi non-image tokens và original positions khi compact. `:57` giữ native weights/position handling; `:79` đặt greedy (`num_beams=1`, `do_sample=False`); `:91` xử lý EOS và continuation.
- `smolvlm_inference/decoder.py:78` dùng Q/K từ projection của target attention thực tế, RoPE/GQA/scale/mask rồi all-key softmax. `:174` score tại đúng target block; `:188` gather hidden states **sau** block đó. `:199` decode không chọn lại.
- `smolvlm_inference/pruning.py:131` tính mean camera theo camera IDs thực, min–max, softmax shares, IRA và bounded budget; không đổi thành direct allocation. `pruning_text.py:72` giữ source IDs/positions.

Các đường dẫn ở đoạn trên nằm trong package tương ứng tại repository này. Softmax không chuẩn hóa lại riêng visual keys: tổng visual mass có thể nhỏ hơn1.

## Kết quả kiểm chứng động độc lập

Môi trường: Windows, Python trong `.venv`, Torch `2.15.0+cpu`, Transformers `4.46.3`; không CUDA. Model nhỏ dùng GQA4 query heads/2 KV heads, sáu camera, tổng384 projected tokens. Geometry vision giả là256 tokens/camera ×8 chiều, không phải vision encoder production1024×768. Không tải weights/cache lớn.

| Phép đo | Max absolute error | Relative L2 | Kết quả selector |
|---|---:|---:|---|
| Export30 layers, left padding, native eager | 4.657e-10 | 1.760e-8 | Exact Top32 mọi sample/layer |
| Export30 layers, right padding, native eager | 4.657e-10 | 1.894e-8 | Exact Top32 mọi sample/layer |
| Online index2, input thực sau Stage1 | 9.313e-10 | 6.560e-8 | 32/32 cả hai mẫu |
| Online index15, input thực sau Stage1 | 9.313e-10 | 6.292e-8 | 32/32 cả hai mẫu |

Cosine score xấp xỉ1. Native eager/SDPA target-block output lệch tối đa2.980e-8. Online visual mass khoảng0.7888/0.7825. Quota reference độc lập khớp lần lượt `[8,5,4,6,6,3]`, `[6,6,8,3,4,5]` ở index2; `[7,6,5,5,6,3]`, `[6,6,8,3,4,5]` ở index15. Tổng32, indices duy nhất, source/camera mapping khớp qua hai stages.

Reference export dựng native fused inputs độc lập và lấy eager attention; reference Top-K dùng Python sort `(-score,index)`. Reference online dùng eager của chính block với input sau Stage1, rồi quota repair tuần tự và Python ranking độc lập. Không gọi lại selector đang kiểm tra để làm oracle.

Các kiểm tra bổ sung:

- Export shape `[2,30,384]`; dấu nhận diện layer phát hiện được lệch thứ tự. Shuffled writes qua hai shards đọc lại chính xác theo sample IDs; max error0.
- Collector không đổi weights, CPU RNG hoặc native SDPA output (output max error0). Exception được tiêm tại layer13 vẫn gỡ sạch30 hooks.
- Ties fixture chọn đúng ưu tiên index nhỏ; source sorting và camera markers đều khớp. Export min rank32/33 gap chỉ4.657e-10; online min per-camera quota cutoff gap1.770e-8. Precision/backend khác có thể đổi indices gần hòa; không suy từ đó thành lỗi mapping.
- Compact384 visual+26 nonvisual từ410 xuống58 positions, giữ đủ nonvisual IDs/embeddings/positions. Identity/all-keep greedy khớp native.
- Cached pruned greedy khớp reference chậm full-history với selection cố định. Probe position riêng xác nhận token decode tiếp tục ở original positions84/85/86, không phải compact64/65/66.
- Online layer lengths tại index2 là `[196]×3+[32]×14`; tại index15 là `[196]×16+[32]`. Giữ đúng output tokens sau target block. Instrumentation đo scorer chạy đúng1 lần qua4decode steps; không Gumbel/STE/sampling/chọn lại. RNG/weights không đổi.

## Khe hở validation có thể tái hiện

`smolvlm_attention/sample_store.py:353` và `smolvlm_topk_benchmark/data.py:51` chưa kiểm tra semantics của `layer_ids`/`axes`. Trên một export hợp lệ, đổi metadata thành `layer_ids=[1,0]`, `axes=['sample','visual_token','layer']`, cập nhật hash metadata trong manifest để mọi checksum vẫn nhất quán: `open_prompt_attention` vẫn chấp nhận.

Phân loại: **có lỗi kiểm tra metadata**, không phải bằng chứng thuật toán attention/producer đang sai. Producer tại `sample_store.py:329` ghi canonical axes và layer order. Checksum xác nhận bytes, không xác nhận ý nghĩa các trục.

Đề xuất sửa riêng nếu được yêu cầu: fail closed khi axes khác `[sample,layer,visual_token]` hoặc layer IDs không khớp thứ tự dự kiến; thêm regression cho metadata tự nhất quán nhưng sai nghĩa. Audit này không sửa.

## Coverage, identity và giới hạn

Focused suite: **76 passed,12.94s**. Không cộng các lượt chạy subset trùng nhau thành tổng test mới. Lệnh:

```powershell
$env:PYTHONPATH='D:\AD\Visualize;D:\AD\Visualize\tests'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:USE_TF='0'
.\.venv\Scripts\python.exe -m pytest tests/test_attention_collector.py tests/test_attention_data.py tests/test_sample_attention.py tests/test_topk_data.py tests/test_topk_generation.py tests/test_topk_preflight.py tests/test_topk_state.py tests/test_smolvlm_pruning.py tests/test_smolvlm_pruned_decoder.py -q
```

Đã đọc assertion: attention oracle cũ ở `test_smolvlm_pruned_decoder.py:56` chỉ đo block0 chưa Stage1; fixture cũ nhỏ hơn production; `selection_calls` trong debug là giá trị khai báo (`decoder.py:273`), không phải counter đo. Vì vậy audit dùng probes độc lập ở target2/15,384→196→32 và instrumentation thật. Cached reference có dùng chung prefill/selection nên chỉ đủ kiểm tra history/cache; correctness của selection được kiểm tra riêng bằng oracle độc lập bên trên.

Source/config kiểm tra normal best checkpoint `huyvanzzz/smolvlm256M_base`, revision `f9d4e9a4d704078f9e7f39f061d13602cf251c78`, camera Front→Front-Left→Front-Right→Back→Back-Left→Back-Right. Export hỗ trợ thêm reverse profile; benchmark Top-K hiện kiểm tra base normal. Prepared revision `d2a6914f681dbb9148c4ac11e6fa92452010506e`; cache fingerprint `0632954bab4d7a8e5af5934f3476610a0547db8f3083495c1207c3da87b15585`. Đây là identity khai báo/được validator yêu cầu, chưa phải xác minh file production của một run chưa cung cấp.

Chưa kiểm chứng checkpoint thật trên GPU, FP16/BF16, toàn bộ16817 samples, full decode-logit parity mọi bước hoặc thời gian/VRAM. Không chạy full suite. Không yêu cầu hai luồng sinh cùng đáp án vì query, candidate pool và thời điểm prune khác nhau.

## Artifacts audit tạm và lệnh tái hiện

Trong `%TEMP%` của phiên audit:

- `audit_export_attention_c29.py` / `_results.json`: native eager30layers, hooks/RNG, shards/mapping.
- `audit_topk_mapping.py` / `_results.json`: independent ranking, compaction và greedy.
- `audit_mv_native.py` / `_results.json`: actual target2/15, independent quota/ranking và cached decode.
- `audit_export_metadata.py`: metadata validation reproduction.
- `smolvlm_attention_audit_vkxfzyot/token_probe.py`, `native_question_span_probe.py` và JSON tương ứng: tokenizer thật local-only, answer isolation; `source_hashes.json`: hashes source trước audit.

Chạy các script với interpreter/env của lệnh pytest trên; script là phép đo tạm, không đưa vào production. Hashes source được đối chiếu lại cuối audit để xác nhận source không đổi. Không commit/push.
