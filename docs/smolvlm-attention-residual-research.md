# Research residual attention predictor cho MVPruner

Ngày: 2026-11-23. Repository được đọc: Log_Metrics, `smolvlm-attention`, snapshot `104d80bb65c69304c07e047a68af17c374339332`.

## 1. Kết luận và quyết định đã chốt

**Giữ hai hướng làm đối chứng trước khi chọn**, theo trả lời của người dùng: (A) residual log-score tổng hợp và loss phân bố; (B) residual raw logits từng head và MSE. Chưa chọn phương pháp thắng, chưa lập plan code và chưa train.

Đề xuất mạng chính cho vòng đầu là **MLP residual dùng chung theo visual token, hidden64**, có tương tác visual–query. Linear residual là baseline tối thiểu; Q/K predictor chiều thấp là phương án thứ ba khi MLP chưa đủ hoặc cần giảm phép nhân. Đây là đề xuất tổng hợp, không phải kiến trúc đã được paper chứng minh trên SmolVLM/DriveLM.

Lý do: selector cần một score mỗi token; học toàn attention matrix không cần thiết. Mạng phải thấy thông tin câu hỏi, không chỉ nhận scalar attention thấp. MSE raw logits có bằng chứng từ TinyBERT; KL attention có bằng chứng độc lập từ MiniLM và MAP. Không có bằng chứng các nguồn này xác nhận một hướng luôn thắng hướng kia trong setup hiện tại.

Chỉ research và thêm báo cáo; không sửa implementation/config/notebook, tạo dataset/cache, tải weights hay chạy training. Các phép tính tham số/dung lượng là công thức, không phải benchmark.

## 2. Sàng lọc 15 paper

Venue dưới đây phân loại theo proceedings/paper chính thức; Findings không được gọi là ACL/EMNLP main. Không gán CORE rank hoặc acceptance cho preprint.

| # | Paper / nguồn chính | Venue, năm | Quyết định và giá trị |
|---|---|---|---|
| 1 | [TinyBERT](https://aclanthology.org/2020.findings-emnlp.372.pdf) | Findings EMNLP2020 | Đọc sâu: MSE attention chưa softmax, code mask/reduction |
| 2 | [MiniLM](https://proceedings.neurips.cc/paper/2020/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) | NeurIPS2020 | Đọc sâu: KL phân bố, teacher-layer transfer |
| 3 | [MiniLMv2](https://aclanthology.org/2021.findings-acl.188.pdf) | Findings ACL2021 | Đọc sâu: relation heads và teacher layer; không ép head correspondence |
| 4 | [MobileBERT](https://aclanthology.org/2020.acl-main.195/) | ACL2020 main | Bổ trợ: bottleneck và knowledge transfer; không dùng cả student LM cho mục tiêu nhỏ này |
| 5 | [MAP](https://arxiv.org/html/2608.06411v1) | arXiv2026, chưa xác minh acceptance | Đọc sâu: frozen VLM + predictor attention; sát nhất nhưng teacher/query/context khác |
| 6 | [DynamicViT](https://proceedings.neurips.cc/paper/2021/file/747d3443e319a22747fbb873e8b2f9f2-Paper.pdf) | NeurIPS2021 | Đọc sâu: predictor local/global, cách huấn luyện selection; không phải attention residual |
| 7 | [EViT](https://openreview.net/pdf/feb0c5a2e1c1fc63509c2e528ca07aa95aea2d5e.pdf) | ICLR2022 | Bổ trợ: CLS attention và fusion; không có predictor học score sâu |
| 8 | [TwigVLM](https://openaccess.thecvf.com/content/ICCV2025/html/Shao_Growing_a_Twig_to_Accelerate_Large_Vision-Language_Models_ICCV_2025_paper.html) | ICCV2025 main | Đọc sâu: shallow trainable branch trên frozen base; khác attention-regression |
| 9 | [LearnPruner](https://openreview.net/pdf?id=Dxb6gBJHby), [author manuscript](https://arxiv.org/html/2604.23950v1) | ICLR2026, published header được search index xác nhận | Đọc sâu: MLP nhỏ, STE/task loss; không phải distillation logits |
| 10 | [FastV](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/10478_ECCV_2024_paper.php) | ECCV2024 | Bổ trợ: baseline early attention pruning, không train predictor |
| 11 | [SparseVLM](https://proceedings.mlr.press/v267/zhang25s.html) | ICML2025 | Đọc sâu paper/code: query/rater selection; không tự đổi query của mình theo họ |
| 12 | [FEATHER](https://openaccess.thecvf.com/content/ICCV2025/papers/Endo_Feather_the_Throttle_Revisiting_Visual_Token_Pruning_for_Vision-Language_Model_ICCV_2025_paper.pdf) | ICCV2025 main | Đọc sâu phần phân tích: bias vị trí và localization; không coi QA trung bình đủ chứng minh teacher tốt |
| 13 | [VisPruner](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Beyond_Text-Visual_Attention_Exploiting_Visual_Cues_for_Effective_Token_Pruning_ICCV_2025_paper.pdf) | ICCV2025 main | Bổ trợ đối chứng: attention không luôn là importance; không thay IRA bằng diversity |
| 14 | [VisionZip](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_VisionZip_Longer_is_Better_but_Not_Necessary_in_Vision_Language_CVPR_2025_paper.pdf) | CVPR2025 main | Bổ trợ: compression ngoài LM; không trực tiếp giải quyết residual attention |
| 15 | [DART](https://aclanthology.org/2025.emnlp-main.505/) | EMNLP2025 main | Bổ trợ: duplication có thể quan trọng hơn scalar importance; không tự thêm selector mới |

Chọn **9 bài đọc sâu** (1,2,3,5,6,8,9,11,12); sáu bài còn lại sàng lọc abstract/method overview để phân loại và kiểm tra phản chứng. Không tuyên bố đã audit toàn bộ code của cả15 bài.

## 3. Bằng chứng liên quan nhất

| Paper / vị trí | Target, loss và kiến trúc thực tế | Điều có thể dùng / điều không được suy ra |
|---|---|---|
| TinyBERT §3.1 Eq7 | Fit attention chưa softmax từng head bằng MSE; đồng thời có hidden-state/embedding distillation | Tác giả báo pre-softmax hội tụ tốt hơn trong BERT. Không chứng minh MSE residual23←3 tốt hơn KL trên multi-camera |
| MiniLM §3 Eq3–6 | KL teacher→student của attention distributions và value relations; last teacher layer | Post-softmax vẫn học được. Có train student Transformer, không phải chỉ thêm một scalar predictor |
| MiniLMv2 §3.1–3.2 | Q-Q/K-K/V-V relations, chia lại relation heads, KL; teacher upper-middle có thể tốt hơn last với model lớn | Không phải cứ layer cao hơn luôn tốt; không cần buộc native head h thấp có cùng ý nghĩa với head h cao |
| MAP §Cross-Modal Attention Distillation Eq1–3 | Teacher mean heads/question queries, L1-normalize visual scores; predictor MLP theo modality + Q/K, KL; không V/O | Frozen backbone và small predictor có tiền lệ sát bài toán. Họ dự đoán trực tiếp ở pre-LLM, chọn teacher theo sample, không residual cố định22←2. Chưa tìm thấy repo tác giả để xác minh hidden width/optimizer |
| DynamicViT §3.2–3.3 và source | Predictor local/global, binary keep/drop; Gumbel masking; classification/ratio/distillation objectives | Tham khảo ngữ cảnh global rẻ. Không cần bê STE/Gumbel vào supervised regression của mình; paper cho phép train backbone |
| TwigVLM §2.2, §3.1–3.2 và source | Train nhánh Transformer/head riêng bằng AR loss trên frozen trunk; attention của twig dùng chọn token | Có thí nghiệm chọn bằng layer sâu rồi prune sớm rất sát oracle hiện tại. Nhánh decoder/head nặng hơn MLP; không phải residual attention MSE |
| LearnPruner §3.3, AppA.1 Eq7–8 | MLP binary pruning, STE/masked attention; native next-token loss + pruning-ratio penalty | Chỉ train ít parameter không đồng nghĩa không backward qua phần LM phía sau. Không dùng task-loss route cho v1 distillation nếu muốn train nhẹ |
| SparseVLM method và source | Rater text tokens, lựa chọn heads và visual scoring; training-free | Query policy có ảnh hưởng. Không dùng kết quả đó để tự đổi last-prompt-query thành mean-question |
| FEATHER §analysis/method | Phân tích RoPE/position bias và localization; sửa scoring strategy và multistage sampling | Cần xem phân bố camera/vị trí và bài hỏi không gian. Đây không phải bằng chứng phải bỏ RoPE trong model hiện tại |

Không có bài trong nhóm này xác nhận trực tiếp **SmolVLM256M, sáu camera, DRA/CCTS196, teacher22 và IRA32**. Số speedup/retention của LLaVA/BERT không được chuyển thành kỳ vọng định lượng cho DriveLM.

### Code tác giả đã đọc, pin commit

- TinyBERT: commit `0598f02d7fc4eaa7b4cbc1e9d7ab18bc875c24f9`. [Transformer forward L401–420](https://github.com/huawei-noah/Pretrained-Language-Model/blob/0598f02d7fc4eaa7b4cbc1e9d7ab18bc875c24f9/TinyBERT/transformer/modeling.py#L401): scaled score cộng mask, softmax riêng, trả attention_scores cho distillation. [Training L935–964](https://github.com/huawei-noah/Pretrained-Language-Model/blob/0598f02d7fc4eaa7b4cbc1e9d7ab18bc875c24f9/TinyBERT/task_distill.py#L935): teacher no_grad, layer mapping, thay masked entries bằng0 rồi MSE. Không nên sao chép mean gồm padding; v1 cần mean trên valid entries.
- DynamicViT: commit `1322e626d1e9eca18cc150569908fb4af29e15f7`. [PredictorLG L287–315](https://github.com/raoyongming/DynamicViT/blob/1322e626d1e9eca18cc150569908fb4af29e15f7/models/dyvit.py#L287): LayerNorm/MLP, local half và policy-weighted global mean, output2 classes/log-softmax. [Loss L136–185](https://github.com/raoyongming/DynamicViT/blob/1322e626d1e9eca18cc150569908fb4af29e15f7/losses.py#L136): ratio và task/distillation losses; không hồi quy attention sâu.
- TwigVLM: commit `65f94c4ab5cfb5f7b9d628f41f397da1b6516f23`. [Train L957–967](https://github.com/MILVLG/twigvlm/blob/65f94c4ab5cfb5f7b9d628f41f397da1b6516f23/twigvlm/train/train.py#L957): truncate shallow training model, freeze rồi mở gradient twig layers, norm/head. [Inference patch L159](https://github.com/MILVLG/twigvlm/blob/65f94c4ab5cfb5f7b9d628f41f397da1b6516f23/twigvlm/inference/twigvlm_patches_for_llama.py#L159): tạo twig layers/head. Không kết luận họ chỉ train MLP nhỏ.
- SparseVLM: commit `a9e71427220c4cacb6afb5a96d23cfb43142e0c3`. [Model L195–259](https://github.com/Gumpest/SparseVLMs/blob/a9e71427220c4cacb6afb5a96d23cfb43142e0c3/llava/model/language_model/modelling_sparse_llama.py#L195): chọn text raters từ visual/text states, rồi select heads/postprocess attention. Code có assumptions batch/index của họ; không copy trực tiếp sang batch64 của mình.
- MiniLM official [README theo commit](https://github.com/microsoft/unilm/blob/31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd/minilm/README.md): đối chiếu release/model identity; chưa xác minh đầy đủ training-loss implementation, nên công thức được dẫn từ paper.

MAP chưa tìm được code tác giả qua paper/link/search; LearnPruner manuscript nói sẽ release, chưa xác minh repo chính thức. OpenReview trực tiếp bị browser challenge dù search index có published header; FEATHER code request gặp lỗi chứng chỉ. Không suy ra các repo này không tồn tại. Không gán hidden sizes/reduction/optimizer không được công bố.

## 4. Đối chiếu repository và đại lượng cần học

Source hiện tại:

- `smolvlm_attention/collector.py:69–82`: query cuối prompt, Q/K native, RoPE/GQA, all-key softmax rồi mean heads và gather384 visual columns. `:95` trả `[B,L,384]`; export không chứa logits hoặc từng head.
- `smolvlm_attention/sample_inputs.py:71–92`: workflow export đang cố định full test16817; không phải dataset train predictor.
- `smolvlm_inference/decoder.py:173–189`: online score query-question ở target block; saved scores gather theo source IDs của196 tokens rồi gọi selector. Không tái sử dụng score question-mean làm baseline last-query mà không đổi nhãn.
- `smolvlm_inference/pruning.py:131–139`: camera means → minmax/clamp epsilon → camera softmax → IRA weights → bounded budget và raw-score ranking trong camera.

Đọc config native đã có trong local cache, không tải weights: D=576,30layers,9 query heads,3 KV heads,head_dim64. GQA không có nghĩa chỉ có3 attention distributions; reduction đang mean9 query heads.

Teacher vẫn là **full384 base index22**, query cuối native prompt (`:` trong Assistant:). Student chạy đường **Stage1→196→index2**, cùng last-query policy khi định nghĩa residual. Giữ prompt/camera/source position IDs. Query state của dấu `:` sau các layer có thể chứa thông tin câu hỏi, không chỉ là embedding cố định của dấu câu.

Chênh lệch teacher–student gồm độ sâu **và** thay context384→196. Không gọi nó thuần túy là layer residual. Không thay teacher bằng teacher chạy196 hoặc đổi thành mean-question trong lượt research này.

### Vì sao số0.01 không tự làm học thất bại?

Với MSE trực tiếp trên xác suất, gradient qua softmax có thêm Jacobian và mean reduction có thể làm scale nhỏ. Có thể scale loss, nhưng scale không sửa được target không phù hợp với ranking. Với KL(p||softmax(u)), gradient theo u là `p_hat−p` khi sum trên token; không nhân thêm toàn bộ Jacobian như probability-MSE. Không có ngưỡng “xác suất dưới0.01 thì không học được”.

Raw logits cũng có nhược điểm: cộng cùng offset không đổi softmax; MSE vẫn phạt offset. Dùng centered logits có thể loại nuisance offset nhưng **không chuẩn hóa variance/temperature tùy tiện**, vì điều đó đổi phân bố. Tất cả mask/padding bị loại khỏi loss, không hồi quy `−inf` hoặc sentinel rất âm.

Và `mean_h softmax(z_h)` khác `softmax(mean_h z_h)`. Chênh lệch mean raw logits của hai layers không tự cho đúng chênh lệch score selector.

## 5. Hai hướng đối chứng được đề xuất

Ký hiệu J là196 original visual indices sau Stage1, H=9. Teacher attention tính trên full native prompt; student attention tính trên prompt đã giữ196.

### A. Residual log-score tổng hợp + KL

Từ mean-head native attention s trên J, tạo phân bố visual:

`p_T = s_22[J] / sum(s_22[J])`, `p_S = s_2 / sum(s_2)`.

Định nghĩa `u = log(p)` và center theo196 tokens; thực hiện ổn định bằng logsumexp khi có logits, thay vì lấy log của xác suất đã underflow. Predictor học:

`u_hat = u_S + delta_theta(X_2, q_2, u_S)`, `p_hat = softmax(u_hat)`.

Loss chính đề xuất: `mean_batch sum_j p_T[j] * (log p_T[j] − log p_hat[j])`. Không dùng mean trên cả batch×196 mà vô tình thêm hệ số1/196. Temperature v1 đề xuất1; temperature khác là ablation, không tự flatten teacher. Centered log-score MSE là đối chứng loss có kiểm soát nếu cần, không mặc định cộng mọi loss.

Đây **không phải raw QK logits của bất kỳ head nào**. Nó là biểu diễn log của score đã tổng hợp; softmax cuối chỉ phục hồi một phân bố score cho selector. Native attention của LLM không đổi. Đây là residual learning đúng yêu cầu, không phải direct-prediction MAP.

Với IRA, score nhân một số dương chung a cho cả mẫu cho cùng minmax nếu cả hai ranges đều vượt epsilon; Top-K trong camera cũng giữ nguyên. Vì vậy dùng phân bố trên J về mặt toán học không mất tín hiệu selector trong miền này. Nếu range chạm epsilon, tính bất biến không còn chính xác; phải log tần suất và so quota exact với score teacher gốc. Float rounding/ties vẫn có thể đổi integer allocation. Không thể tuyên bố mọi mẫu luôn bất biến.

### B. Residual raw logits từng head + MSE

Teacher target visual: scaled post-RoPE logits `z_T[h,j]` trên J; student `z_S[h,j]` từ input của attention block index2. Center từng head trên cùng J để loại offset. Mạng xuất9 residuals/token:

`z_hat[h,j] = centered(z_S[h,j]) + delta_theta[h,j]`.

`L_z = mean_batch,head,valid_visual (centered(z_hat) − centered(z_T))²`.

Không dùng mean raw logits rồi softmax để thay mean attention. Có hai cách phục hồi score đúng nghĩa: dự đoán cả logits của toàn bộ valid keys, hoặc giữ visual logits và học thêm **retained-visual mass từng head**. Cách thứ hai gọn hơn:

`m_T[h] = sum_(j in J) A_T[h,j]`, `s_hat[j] = mean_h m_hat[h] * softmax_J(z_hat[h])[j]`.

Lý do cần mass: softmax riêng visual trong từng head làm mọi head có cùng total mass1, trong khi teacher có thể dành attention cho text hoặc188 visual tokens đã bỏ ở Stage1. Khác biệt mass giữa heads ảnh hưởng mean-head visual scores và quota camera. Một hệ số chung sau mean không sửa được sai lệch này.

Mass có thể học residual trên `logit(m_S[h])` với sigmoid và BCE soft targets; hoặc MSE log-mass. Đây là lựa chọn bổ sung cần chốt, không được âm thầm bỏ mass rồi coi là khớp teacher. Nếu m gần0/1 cần ổn định bằng logsumexp và clipping có log thống kê. Chưa chốt hệ số loss phụ trước khi kiểm tra scales.

Head h là slot target của predictor, không khẳng định native head h ở hai layers cùng chức năng. Residual theo slot vẫn học được về mặt biểu diễn, nhưng correspondence tùy ý có thể tăng độ khó; cần so loss/selection theo từng head và không dùng MSE nhỏ làm bằng chứng semantic alignment.

**Khuyến nghị research:** A dễ khớp đúng selector hơn, B sát yêu cầu raw pre-softmax hơn nhưng cần mass/normalization rõ ràng. Người dùng đã chọn giữ cả hai. Không đổi query, teacher context, IRA hoặc bỏ residual để khiến một hướng thuận lợi hơn.

## 6. Mạng đề xuất, shapes và chi phí

Tận dụng output hidden states của block index2, sau khi block đã chạy bình thường: X `[B,196,576]`, last-query state q `[B,576]`; low score/logits lấy từ attention của chính block đó. Không đọc states index22 khi inference.

Với fixed per-token RMS normalization không affine, feature tham khảo:

`F_j = [RMS(x_j); RMS(q); RMS(x_j) * RMS(q); u_S[j]]` →1729 chiều.

Interaction product giúp phụ thuộc câu hỏi. Chỉ linear trên `[x_j;q]` thì phần q là offset chung theo token và bị centering/softmax triệt tiêu; baseline đó không đủ để kiểm tra question conditioning.

| Ứng viên | Công thức/shape | Tham số và chi phí đại số mỗi mẫu |
|---|---|---|
| Linear residual tối thiểu | linear trên `[x_j; x_j*q;u_j]`, output1 | 1154 params; khoảng0.452M FLOPs cho196tokens, chưa tính normalization/product |
| MLP64 ưu tiên | `Linear(1729,64)→GELU→Linear(64,1)` | **110785 params**, khoảng43.40M dense FLOPs |
| MLP128 ablation | cùng input, hidden128 | **221569 params**, khoảng86.80M dense FLOPs |
| Q/K low-rank | `delta_j=(Wq q)^T(Wk x_j)/sqrt(r)`; r32/64, không bias/V/O | 36864/73728 params; khoảng7.27M/14.55M FLOPs |

Quy ước2 FLOPs/MAC; không gồm tính low attention, memory traffic, RMS/GELU/softmax, tensor packing hoặc backward. Không quy FLOPs thành latency4090. Có thể tính phần `Wq*q` của MLP một lần/sample thay vì repeat query/concat để giảm compute; giữ nguyên toán học.

Đối với B, dùng cùng shared trunk và output9 logits/token, cộng mass head9 từ pooled trunk. Trên feature1729/hidden64 minh họa: shared trunk110720 params, mỗi output head585 params → tổng **111890 params**, chưa gồm thay đổi feature nếu bổ sung9 low logits. Không nhầm con số này với scalar MLP110785.

Khởi tạo output residual bằng0 để bắt đầu từ score thấp; hidden layer vẫn khởi tạo ngẫu nhiên. Backward đầu tiên hidden-layer gradient có thể bằng0 hợp lệ. Với Q/K bilinear, chỉ zero một projection; zero cả hai làm cả hai gradient bằng0. Không thêm camera embedding mới v1: states đã có context/camera positions và selector dùng camera IDs hiện tại. Không thêm Transformer hoặc cross-token attention mới trước khi các baseline này được kiểm tra.

## 7. Dữ liệu, training cost và protocol đề xuất

### Không dùng test để học

Export hiện tại chỉ dùng làm oracle evaluation và kiểm tra format, không làm train targets. Tạo targets từ train; validation chọn checkpoint/representation; test chỉ báo cuối. Nếu đã dùng test để chọn teacher index22 thì phải công bố đó là lựa chọn exploratory trên test, không gọi toàn pipeline là test-blind. Một predictor không thấy answers vẫn có thể leakage nếu fit attention targets từ chính test.

Kiểm tra split overlap theo scene trước, tối thiểu theo bộ6camera paths/cache key. Không tự đổi official split. Nếu thiếu scene ID thì ghi giới hạn image-group không loại hết near-duplicate temporal frames; khi cần chia nội bộ train phải hỏi lại protocol.

### Cache tối thiểu, không mặc định lưu hidden states

Teacher full384/index22 chạy offline no_grad; student prefix196/index2 chạy no_grad trong train predictor. Chỉ graph predictor được giữ. Không cần backward qua LLM/STE/Top-K hoặc chạy decoder đáp án để học target attention. Frozen không miễn chi phí teacher forward hoặc prefix inference.

Với341381 train QA, FP32, chưa gồm metadata/indices/filesystem:

| Tensor cache | Dung lượng đại số |
|---|---:|
| Teacher196 scalar scores | khoảng0.249GiB |
| Teacher384 scalar scores | khoảng0.488GiB |
| Teacher9×196 raw visual logits | khoảng2.243GiB, chưa gồm9mass/sample |
| StudentX196×576 + query576 + low196 + teacher196 | khoảng144.806GiB; FP16 khoảng72.403GiB |

Đề xuất teacher targets nhỏ trên disk, prefix states tính online/no_grad hoặc cache subset phục vụ ablation. Không thay cache vision36GiB hiện có. Vì query làm states khác nhau, không deduplicate student hidden states theo ảnh như vision cache. Chưa tạo bất kỳ cache nào.

### Các đối chứng và thứ tự đánh giá

1. Low index2 **last-query** chưa hiệu chỉnh; ghi tách khỏi online MVPruner question-mean cũ.
2. Teacher index22 saved-score với cùng Stage1/IRA32; chỉ là oracle source, không bảo đảm upper bound chất lượng.
3. Linear residual; MLP64 với A/B; MLP128 hoặc Q/K chỉ mở rộng khi validation cho thấy cần.

Giữ tập mẫu/seeds, teacher/context/query/selection và precision như nhau. Đo target error, KL, Spearman/rank32 boundary; **quota vector exact match và overlap32 sau IRA**; camera/site histogram, near-tie cases, epsilon hit rate và mass theo head. Chất lượng cuối đo QA greedy; không thay IRA bằng globalTop32 khi báo agreement.

QA outcome có thể tốt dù không match teacher hoàn toàn, hoặc match teacher nhưng không tốt hơn low baseline. Không chọn model chỉ bằng train loss. Nên chọn validation selection metrics làm screening, xác nhận QA trên validation trước test; không chốt hệ số loss/epoch/LR từ recipeBERT lớn.

Đo riêng teacher-target creation, prefix forward, low-score calculation, predictor, selector và generation. Cho cả chi phí chuẩn bị lẫn inference steady-state; không tính FLOPs predictor là toàn bộ overhead. Thay đổi batch64/grouping có thể tạo numeric near ties; không nới tolerance để ép agreement.

## 8. Các quyết định còn cần chốt trước plan code

- Hai hướng A/B đã được chốt làm đối chứng; query vẫn last-prompt-query và teacher vẫn full384/index22. Student low baseline cần last-query mới, không âm thầm dùng question-mean.
- Với B: có đồng ý centered raw logits + thêm residual head-mass để khôi phục đúng mean-head score, hay muốn dự đoán toàn valid-key logits? Chưa chốt; không tự bỏ phần này.
- Có đồng ý MLP64 shared-token làm mạng chính và linear interaction làm baseline? Đây là đề xuất, chưa phải kết luận thực nghiệm.
- Budget dữ liệu: full train hay subset scene-disjoint ban đầu; lưu teacher-only hay thêm prefix-state cache. Cần chọn theo thời gian/disk, chưa chọn thay người dùng.
- Nếu oracle index22 không cải thiện pruning so với baseline, cần xem lại mục tiêu học trước khi đầu tư tạo targets; không tự đổi teacher layer.

Không cần đổi architecture của SmolVLM, native attention, generation hoặc existing checkpoint. Chưa có kết quả thực nghiệm để nói A/B tốt hơn, predictor nhanh bao nhiêu, hay cần bao nhiêu epoch.

## 9. Phạm vi kiểm chứng của research

Đã search/sàng lọc15 bài, đọc sâu9 bài, đối chiếu source pin commit ở TinyBERT/DynamicViT/TwigVLM/SparseVLM và source hiện tại. Một số PDF/website bị403/challenge; dùng author HTML/official indexed source thay thế và ghi giới hạn. Không có tải weights/dataset, model forward, training hoặc production benchmark. Source snippets tải vào thư mục tạm `attention_distillation_research_7b9w2jyd`; không đưa repo tác giả vào repository này.

Không áp dụng các đề xuất vào code. Đây là bàn giao research để chốt lựa chọn rồi lập plan triển khai riêng.
