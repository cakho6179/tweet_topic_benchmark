# Future Work & Deferred Experiments

Tài liệu này lưu lại các kỹ thuật / phương pháp được tạm gác lại trong đợt thực nghiệm hiện tại để giữ cho hệ thống benchmark dễ hiểu, kiểm soát chặt chẽ từng yếu tố và ưu tiên các hàm mất mát nền tảng (`BCE`, `Weighted BCE`, `Focal Loss`).

---

## 1. Asymmetric Loss (ASL) cho Phân loại Đa nhãn Mất cân bằng

* **Tham khảo gốc:** Ben-Baruch et al. (2021), *"Asymmetric Loss For Multi-Label Classification"*.
* **Lý do tạm loại khỏi đợt benchmark hiện tại:**
  * Đợt thí nghiệm hiện tại tập trung làm rõ từng thành phần nền tảng trên kiến trúc **BERT truyền thống** (`BCE`, `Weighted BCE`, `Focal Loss`), hiệu ứng tối ưu ngưỡng theo lớp (`per-class thresholding`), cơ chế pooling (`CLS` / `Mean` / `Attention`), Multi-Sample Dropout và hiệu chuẩn xác suất.
  * ASL kết hợp đồng thời 2 cơ chế bất đối xứng ($\gamma^+ \neq \gamma^-$ và *probability margin shifting* $m$), cần hiểu và phân tách kỹ trước khi đưa vào báo cáo chính thức.

### 1.0. Điều kiện bắt buộc trước khi chạy lại ASL (rút ra từ Vòng 3)

1. **Kiểm soát head.** Ở Vòng 3, cùng hàm BCE đạt Macro-F1 0.257 với head mặc định (pooler tiền huấn luyện, dropout 0.1, lr 2e-5) và 0.568 với head tùy chỉnh (đọc `[CLS]` trực tiếp, dropout 0.2, lr 1e-3) trên `test_2021`; chưa tách riêng được yếu tố nào trong head gây ra khác biệt (cách rẻ nhất: head tùy chỉnh với `head_lr: 2.0e-5`). Run RoBERTa + BCE của Vòng 2 (0.289) dùng head mặc định, nên khoảng cách với ASL (0.573) có thể một phần là do head chưa học đủ. Mọi arm ASL phải chạy cả hai kiểu head (`force_custom_head: true` và mặc định), hoặc ít nhất dùng head tùy chỉnh để so với P0.
2. **Cấu hình ASL của Vòng 2 không phải cấu hình chuẩn.** `AsymmetricLoss` trong commit hiện tại có mặc định $\gamma^+ = 1$, $\gamma^- = 4$, $m = 0.05$ và các config `configs/v2/*asl*.yaml` không ghi đè. Khi so sánh với Vòng 2, phải dùng $\gamma^+ = 1$; khi so với bài báo gốc, dùng $\gamma^+ = 0$. Ma trận ở mục 1.6 nên có thêm một arm $\gamma^+ = 1$.
3. **Khôi phục loss.** `tweet_topic/models/bert/losses.py` ở Vòng 3 chỉ có `bce`, `weighted_bce`, `focal`. Muốn chạy lại `configs/v2/*asl*.yaml` phải thêm lại lớp `AsymmetricLoss` đăng ký dưới khóa `"asymmetric"` (mã tham khảo ở mục 1.7, đặt $\gamma^+$ mặc định theo điểm 2).

### 1.1. Vì sao BCE thuần gặp khó khăn ở đây

Với 91.01% ô nhãn là 0 và tỉ lệ mất cân bằng 20.18 : 1, tổng gradient của BCE bị chi phối bởi các mẫu âm. Với một nhãn đơn, logit $z$, $p = \sigma(z)$:

$$
\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial z} = p - y
$$

* **Mẫu dương hiếm** ($y = 1$): gradient $= p - 1 = -(1-p)$. Một mẫu dương khó ($p$ thấp) vẫn tạo gradient lớn, nhưng số lượng rất ít.
* **Mẫu âm phổ biến** ($y = 0$): gradient $= p$. Mỗi mẫu âm dễ chỉ đóng góp rất nhỏ, nhưng có khoảng 19 × N mẫu âm, nên tổng vẫn áp đảo.

Hệ quả: mô hình có xu hướng đẩy mọi xác suất về gần 0. Nó đạt Micro-F1 tạm ổn nhưng **thu hẹp hoặc triệt tiêu các lớp hiếm** (class collapse), làm Macro-F1 thấp. Đây chính là hiện tượng quan sát được ở run RoBERTa + BCE của Vòng 2: F1 = 0 trên 7 trong 19 lớp (`test_2021`). Vòng 3 cho thấy một phần hiện tượng này đến từ head chưa học đủ chứ không chỉ do loss (xem mục 1.0).

### 1.2. Hai cơ chế của ASL

ASL sửa cả hai phía của bài toán, nhưng theo hai cách khác nhau:

1. **Asymmetric Focusing** ($\gamma^+ \neq \gamma^-$): dùng hệ số $(1-p)^{\gamma}$ để giảm trọng số các mẫu đã được dự đoán đúng và tự tin.
   * $\gamma^+$ áp cho mẫu dương. Cấu hình chuẩn đặt $\gamma^+ = 0$: **không** làm yếu tín hiệu từ mẫu dương hiếm.
   * $\gamma^-$ áp cho mẫu âm. Cấu hình chuẩn đặt $\gamma^- = 4$: làm yếu mạnh các mẫu âm dễ.
2. **Probability Margin Shifting** ($m$): thay vì dùng $p$ cho mẫu âm, dùng
   $$
   p_{m} = \max(p - m,\ 0)
   $$
   Khi $p \le m$, mẫu âm được coi là "quá dễ" và bị **cắt hoàn toàn** khỏi loss. Trong implementation chuẩn, điều này tương đương dùng xác suất âm $1 - p_{m} = \min(1 - p + m,\ 1)$.

### 1.3. Công thức

Với $p_c = \sigma(z_c)$ và $y_c \in \{0, 1\}$:

$$
\mathcal{L}_{\text{ASL}} = -\frac{1}{C}\sum_{c=1}^{C} \Big[ y_c (1 - p_c)^{\gamma^+} \log(p_c) + (1 - y_c) (p_{m,c})^{\gamma^-} \log(1 - p_{m,c}) \Big]
$$

Cấu hình chuẩn thường dùng: $\gamma^+ = 0.0$, $\gamma^- = 4.0$, $m = 0.05$.

### 1.4. Gradient nhìn từng phía

**Phía dương** ($y = 1$), với $p_{m}$ không áp dụng:

$$
\frac{\partial \mathcal{L}^+}{\partial z}
= \gamma^+ p (1-p)^{\gamma^+} \log p - (1-p)^{\gamma^+ + 1}
$$

Khi $\gamma^+ = 0$, biểu thức rút về $-(1-p)$, giống hệt BCE. Đây là lý do cấu hình chuẩn **không** làm yếu mẫu dương.

**Phía âm** ($y = 0$), đặt $p_m = \max(p-m, 0)$:

$$
\frac{\partial \mathcal{L}^-}{\partial z} =
\begin{cases}
0, & p \le m \\[4pt]
p(1-p)\left[-\gamma^- p_m^{\gamma^- - 1}\log(1-p_m) + \dfrac{p_m^{\gamma^-}}{1-p_m}\right], & p > m
\end{cases}
$$

Hai quan sát:

* Khi $p \le m$, gradient bằng **0**. Các mẫu âm cực dễ không còn kéo logit xuống nữa.
* Khi $p > m$, thừa số $p_m^{\gamma^-}$ làm giảm mạnh đóng góp của các mẫu âm còn lại. Với $\gamma^- = 4$ và $p = 0.1$ (khi đó $p_m = 0.05$), $p_m^4 = 6.25 \times 10^{-6}$.

**Ví dụ số** (một nhãn, $m = 0.05$, $\gamma^+ = 0$, $\gamma^- = 4$):

| Trường hợp | BCE gradient | ASL gradient (xấp xỉ) |
|:---|:---:|:---:|
| Dương khó, $p = 0.10$ | $-0.90$ | $-0.90$ (không đổi) |
| Dương dễ, $p = 0.95$ | $-0.05$ | $-0.05$ (không đổi) |
| Âm dễ, $p = 0.01$ | $+0.01$ | $0$ (bị cắt bởi $m$) |
| Âm trung bình, $p = 0.20$ | $+0.20$ | $\approx 4.5 \times 10^{-4}$ (bị focusing) |

Điểm cần nhớ: ASL **không** khuếch đại mẫu dương; nó chỉ giữ nguyên tín hiệu dương và **giảm tín hiệu âm**. Vì vậy ASL phù hợp khi lớp hiếm bị "chìm" bởi lớp phổ biến, nhưng không phải là thuốc chữa mọi trường hợp mất cân bằng.

### 1.5. Cảnh báo khi đánh giá ASL

* **Ngưỡng quyết định:** probability margin shifting làm thay đổi phân phối xác suất. So sánh ASL ở ngưỡng 0.5 với BCE ở ngưỡng 0.5 là không công bằng; phải tìm ngưỡng trên validation cho cả hai.
* **Per-class threshold:** vì ASL đã dịch xác suất, ngưỡng tối ưu theo lớp có thể rất khác 0.5. Nếu chỉ báo cáo kết quả ở 0.5, ASL có thể trông tệ hơn thực tế.
* **Calibration:** ASL thường làm xác suất kém hiệu chuẩn hơn BCE; ECE phải được báo cáo riêng, không trộn với F1.
* **Tương tác với loss khác:** ASL và Weighted BCE đều tăng trọng số phía dương/theo lớp; không nên kết hợp tùy tiện mà không có ablation.

### 1.6. Các thí nghiệm đề xuất cho tương lai

Ma trận tối thiểu, tất cả trên cùng backbone `bert-base-uncased`, cùng recipe P6 (cleaner + attention pooling + MSD) và cùng seed:

| Mã | Loss | $\gamma^+$ | $\gamma^-$ | $m$ | Mục đích |
|:---:|:---|:---:|:---:|:---:|:---|
| A0 | BCE | — | — | — | Đối chứng |
| A1 | ASL focusing-only | 0 | 4 | 0.0 | Tách riêng tác dụng của focusing |
| A2 | ASL full | 0 | 4 | 0.05 | Tách riêng tác dụng của margin shifting |
| A3 | ASL symmetric-focus | 4 | 4 | 0.05 | Kiểm tra vai trò của bất đối xứng |
| A4 | Weighted BCE | — | — | — | Đối chứng cân bằng theo lớp |

Quy tắc đọc kết quả:

1. So sánh **A1 vs A0** để đo focusing thuần.
2. So sánh **A2 vs A1** để đo margin shifting thuần.
3. So sánh **A3 vs A2** để kiểm tra bất đối xứng có thực sự cần thiết.
4. So sánh **A2 vs A4** để biết ASL có hơn Weighted BCE hay không.
5. Chỉ kết luận khi mức tăng vượt quá biến thiên giữa các seed; chạy tối thiểu 3 seed cho cấu hình thắng.

Các hướng mở rộng khác:

* Áp dụng hàm loss cân bằng lên đầu dò `LiquidD1Probe` (`tweet_topic/models/decision/d1_probe.py`).
* Khảo sát tương quan đồng xuất hiện 19 nhãn bằng Label Graph / Classifier Chains trên biểu diễn BERT.
* Kiểm tra ASL trên `cardiffnlp/twitter-roberta-base-dec2020` để tách ảnh hưởng của tokenizer/pre-training khỏi loss.

### 1.7. Đoạn mã tham khảo để khôi phục `AsymmetricLoss`

```python
class AsymmetricLoss(nn.Module):
    """ASL with optional probability margin shifting and asymmetric focusing."""

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_pos = probs.clamp(min=self.eps, max=1.0 - self.eps)
        # Shift the negative probability upward: 1 - max(p - m, 0) = min(1 - p + m, 1).
        probs_neg = 1.0 - probs_pos
        if self.clip is not None and self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(probs_pos)
        loss_neg = (1.0 - targets) * torch.log(probs_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = probs_pos * targets + probs_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            loss = loss * torch.pow((1.0 - pt).clamp(min=0.0), gamma)

        return -loss.mean()
```

Lưu ý: đoạn mã trên dùng `clip` như một **khoảng dịch xác suất âm** (official ASL implementation). Nó tương đương công thức $p_m = \max(p - m, 0)$ ở trên, nhưng tên biến dễ gây nhầm; khi khôi phục cần viết test nhỏ kiểm tra rằng $p \le m$ cho gradient bằng 0.

### 1.8. Checklist trước khi đưa ASL trở lại báo cáo chính

- [ ] Có đối chứng BCE và Weighted BCE chạy cùng backbone, cùng recipe, cùng seed.
- [ ] Có ablation tách focusing và margin shifting (A1, A2).
- [ ] Có kiểm tra bất đối xứng (A3).
- [ ] Ngưỡng được tìm trên validation cho tất cả các arm.
- [ ] Báo cáo ECE riêng, không dùng ECE của arm này để mô tả arm khác.
- [ ] Có ít nhất 3 seed cho cấu hình thắng trước khi kết luận.
