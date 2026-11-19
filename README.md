# CardiffNLP Tweet Topic Multi-Label Classification Benchmark

Hệ thống benchmark phân loại chủ đề đa nhãn (Multi-Label Topic Classification) trên tập dữ liệu [cardiffnlp/tweet_topic_multi](https://huggingface.co/datasets/cardiffnlp/tweet_topic_multi) chạy tối ưu trên môi trường **Kaggle (GPU T4 / P100)**.

Dự án bao gồm đầy đủ **4 baseline đại diện cho 4 thế hệ kiến trúc NLP**:
1. **ML (Machine Learning truyền thống):** Dual TF-IDF (Word n-grams + Char n-grams) kết hợp `OneVsRestClassifier` (Logistic Regression).
2. **BERT / Transformer Encoder:** `cardiffnlp/twitter-roberta-base-2021-124m-topic-multi` (hoặc `roberta-base`) tinh chỉnh với hàm mất mát `BCEWithLogitsLoss`.
3. **Decision Model (`d1-liquid`):** Mô hình đưa ra quyết định có cấu trúc của Liquid AI (`LiquidAI/d1-2B` / `LiquidAI/d1-3B` hoặc Liquid System One API), tính toán xác suất chuẩn hóa đa nhãn trong 1 forward pass duy nhất (Zero Output Tokens).
4. **LLM (`Qwen3.5-4B`):** Mô hình ngôn ngữ lớn thế hệ mới `Qwen/Qwen3.5-4B` lượng tử hóa 4-bit (`bitsandbytes NF4`), trích xuất chủ đề đa nhãn dạng JSON có cấu trúc.

---

## 1. Cấu Trúc Dataset `cardiffnlp/tweet_topic_multi`

- **Nguồn gốc:** Bài báo *"Twitter Topic Classification"* (Antypas et al., COLING 2022).
- **Loại bài toán:** Multi-label Classification (mỗi tweet có thể thuộc 0, 1 hoặc nhiều chủ đề cùng lúc).
- **Tổng số nhãn (19 classes):**
  1. `arts_&_culture` (ID 0)
  2. `business_&_entrepreneurs` (ID 1)
  3. `celebrity_&_pop_culture` (ID 2)
  4. `diaries_&_daily_life` (ID 3)
  5. `family` (ID 4)
  6. `fashion_&_style` (ID 5)
  7. `film_tv_&_video` (ID 6)
  8. `fitness_&_health` (ID 7)
  9. `food_&_dining` (ID 8)
  10. `gaming` (ID 9)
  11. `learning_&_educational` (ID 10)
  12. `music` (ID 11)
  13. `news_&_social_concern` (ID 12)
  14. `other_hobbies` (ID 13)
  15. `relationships` (ID 14)
  16. `science_&_technology` (ID 15)
  17. `sports` (ID 16)
  18. `travel_&_adventure` (ID 17)
  19. `youth_&_student_life` (ID 18)

- **Các tập phân chia (Splits):**
  - `train_2020` (4,585 dòng): Dữ liệu huấn luyện gốc (giai đoạn 2019-2020).
  - `validation_2020` (573 dòng): Dùng để hiệu chỉnh ngưỡng xác suất tối ưu (Threshold Calibration).
  - `test_2020` (573 dòng): Đánh giá in-time (cùng phân phối thời gian 2020).
  - `test_2021` (1,679 dòng): Đánh giá out-of-time (đo lường độ suy giảm hiệu năng do temporal drift).
  - `train_all` (6,090 dòng) & `train_coling2022` (5,731 dòng): Bản mở rộng gộp.

- **Tiêu chí đánh giá chính:**
  - **Micro-F1:** Đánh giá tổng thể có trọng số mẫu.
  - **Macro-F1:** Đánh giá độ bao phủ công bằng giữa các lớp hiếm và lớp phổ biến.
  - **Exact Match Ratio:** Tỷ lệ đoán chính xác toàn bộ vector nhãn của tweet.
  - **Hamming Loss:** Tỷ lệ dự đoán sai trên từng vị trí nhãn.

---

## 2. Cấu Trúc Thư Mục

```text
tweet_topic_benchmark/
├── tweet_topic/                     # Package chính (mô hình, loss, đánh giá, add-on)
│   ├── core/                        # Giao diện chung, registry, factory
│   ├── data/                        # Tải dataset, taxonomy 19 nhãn
│   ├── models/                      # ml/, bert/ (pooling, loss), decision/ (Liquid d1), llm/ (Qwen, QLoRA, auditor)
│   ├── evaluation/                  # Metrics, ngưỡng, hiệu chuẩn (temperature scaling), phân loại có chọn lọc
│   └── addons/                      # Mô-đun cắm-và-chạy (tiền xử lý, ensemble, ví dụ)
├── configs/
│   ├── ml/  bert/  qwen_llm/        # Cấu hình các baseline ban đầu (chạy bằng scripts/run.py)
│   ├── v2/                          # Cấu hình các thí nghiệm kiểm chứng (RoBERTa, d1, QLoRA, auditor)
│   ├── v3/                          # Vòng 3: loss L1-L5, thang kỹ thuật P0-P6, d1 native
│   ├── v3_head/                     # Vòng 3: L2-L5 chạy lại với head của P0 (kiểm soát head)
│   └── v3_seeds/                    # Vòng 3: P0, P6, L2 với seed 1337 và 2024
├── notebooks/
│   ├── 00_..04_*.ipynb              # Notebook Kaggle độc lập của 4 baseline ban đầu (+ EDA)
│   ├── v2_*.ipynb                   # Notebook nhúng source package, chạy scripts/run.py trên Kaggle
│   └── v3_*.ipynb                   # Như v2, cho Vòng 3 (suite loss, technique_a/b, headctl, seed*, d1_native, ...)
├── scripts/
│   ├── run.py                       # Runner chính: dữ liệu → mô hình → ngưỡng → hiệu chuẩn → đánh giá
│   ├── build_package_notebooks.py   # Tạo notebook v2/v3 (nhúng source package)
│   ├── make_kaggle_bundle.py        # Tạo bundle để đẩy lên Kaggle
│   ├── collect_results.py           # Gom artifact tải về từ Kaggle vào outputs/<experiment>/
│   ├── postprocess_analysis.py      # Hiệu chuẩn + đường cong tỷ lệ tự động hóa từ xác suất đã lưu
│   ├── build_v3_report_tables.py    # In các bảng Vòng 3 từ outputs/v3_*
│   ├── generate_benchmark_figures.py# Vẽ biểu đồ từ kết quả thật (outputs/v2_results/, outputs/v3_*)
│   ├── generate_architecture_diagrams.py# Vẽ 7 sơ đồ pipeline kiến trúc (doc/figures/arch_*.png), không cần kết quả chạy
│   ├── kaggle_dispatcher.py         # Điều phối kernel của 4 baseline ban đầu
│   └── build_kaggle_notebooks.py    # Tạo lại notebook 01-04
├── outputs/v2_results/              # Kết quả cuối của các thí nghiệm v2 (metrics_summary)
├── outputs/v3_*/                    # Kết quả Vòng 3: metrics_summary, training_history, postprocess, ...
├── eda/  doc/  learning/            # EDA, báo cáo, ghi chú học tập
├── temp/                            # Các phần đã loại khỏi pipeline (xem temp/MOVED.md để khôi phục)
└── README.md
```

---

## 3. Hướng Dẫn Chạy Trên Kaggle

Có 2 cách chạy trên Kaggle:

### Cách 1: Tải trực tiếp file Notebook lên Kaggle (Khuyên dùng - 1 Click)
1. Vào [Kaggle](https://www.kaggle.com/) -> Chọn **Create** -> **New Notebook**.
2. Chọn **File** -> **Import Notebook** -> Chọn 1 trong 4 file trong thư mục `notebooks/`:
   - `01_kaggle_ml_baseline.ipynb` (Chạy CPU hoặc GPU)
   - `02_kaggle_bert_baseline.ipynb` (Bật GPU T4 hoặc P100)
   - `03_kaggle_d1_liquid_decision.ipynb` (Bật GPU T4 hoặc P100)
   - `04_kaggle_qwen3_5_4b_llm.ipynb` (Bật GPU T4)
3. Trong cài đặt bên phải notebook (Notebook options):
   - **Internet:** Bật `Internet On` (để tải weights và dataset từ Hugging Face).
   - **Accelerator:** Chọn `GPU T4 x2` hoặc `GPU P100`.
4. Bấm **Run All**.

### Cách 2: Chạy qua Command Line (CLI Runner)
Chạy trong terminal của Kaggle (các cấu hình nặng không nên chạy trên máy cá nhân):
```bash
python scripts/run.py --config configs/ml/linear_ovr.yaml                  # TF-IDF + Logistic Regression
python scripts/run.py --config configs/bert/twitter_roberta.yaml           # Twitter-RoBERTa
python scripts/run.py --config configs/qwen_llm/zero_shot_json.yaml        # Qwen3.5-4B zero-shot
python scripts/run.py --config configs/v2/d1_probe.yaml                    # Liquid d1 linear probe
```

*(Thêm cờ `--smoke` nếu muốn chạy test nhanh trên một tập con nhỏ để xác thực pipeline).*

### Cách 3 (v2): Chạy package `tweet_topic` qua Kaggle bằng `scripts/run.py`
Các notebook `01`–`04` tự cài đặt model inline, **không** import package `tweet_topic/`. Các notebook `notebooks/v2_*.ipynb`
nhúng sẵn source của package (zip base64) rồi chạy `scripts/run.py --config configs/v2/*.yaml`, nên mọi ý tưởng đã cài
trong package đều chạy được trên Kaggle. **Không chạy các cấu hình nặng trên máy cá nhân.**

| Suite | Cấu hình (`configs/v2/`) | Nội dung kiểm chứng |
|---|---|---|
| `encoders` | `roberta_dec2020_bce`, `..._bce_tempscale`, `..._asl_perclass`, `..._attnpool_msd_asl` | checkpoint sạch (pretrain-only); temperature scaling; ASL + ngưỡng theo lớp; attention pooling + MSD |
| `d1` | `d1_probe`, `d1_zero_shot_letters` | linear probe trên đặc trưng đóng băng của Liquid d1 (chính); chấm điểm option-letter zero-shot (phụ) |
| `qwen_qlora` | `qwen_qlora_cls` | Qwen3.5-4B QLoRA phân loại có giám sát (không sinh JSON) |
| `ml_audit` | `ml_ovr_preproc_audit` | add-on tiền xử lý + LLM auditor chỉ xử lý 20% tweet mơ hồ nhất |

```bash
S=~/.claude/skills/kaggle-multi-account/scripts/kaggle_acc.py
python scripts/build_package_notebooks.py                       # build lại notebook sau mỗi thay đổi code
python scripts/make_kaggle_bundle.py d1_smoke encoders_smoke    # tạo bundle trong kaggle/kernels_generic/
python $S push --acc acc2 --gpu kaggle/kernels_generic/tweet-topic-v2-d1-smoke        # 1) smoke test trước
python $S wait --poll 240 --out kaggle/outputs                  # chờ xong, tải log + kết quả
python scripts/make_kaggle_bundle.py d1 encoders qwen_qlora ml_audit                  # 2) chạy đầy đủ
python $S push --acc auto --gpu kaggle/kernels_generic/tweet-topic-v2-d1 kaggle/kernels_generic/tweet-topic-v2-encoders
python scripts/generate_benchmark_figures.py                    # vẽ biểu đồ từ kết quả thật
```
Smoke test đã xác nhận suite `d1` và `encoders` chạy hết trên Kaggle; `qwen_qlora` và `ml_audit` chưa được smoke test.
Dispatcher cũ (`kaggle_dispatcher.py`) chỉ đặt `KAGGLE_CONFIG_DIR` nên có thể đẩy nhầm tài khoản; nên dùng `kaggle_acc.py`.

### Cách 4 (v3): Vòng 3, ablation loss và kỹ thuật trên BERT-base
Báo cáo hiện hành: [`doc/MASTER_TWEET_TOPIC_SYSTEM_REPORT.md`](doc/MASTER_TWEET_TOPIC_SYSTEM_REPORT.md). Báo cáo gộp Vòng 1 đến 3, theo bảy phần, kèm tham số từng mô hình (Mục 6.7 và Phụ lục A). Hướng gác lại (ASL): [`doc/Future.md`](doc/Future.md).

| Suite (`notebooks/v3_<suite>.ipynb`) | Cấu hình | Nội dung |
|---|---|---|
| `loss` | `configs/v3/v3_l1..l5` | BCE, Weighted BCE, Focal (γ=2/α=0.25, γ=1/α=0.25, γ=2/không α) với head mặc định |
| `headctl` | `configs/v3_head/v3_h2..h5` | L2-L5 chạy lại với head của P0 (tách ảnh hưởng của head khỏi loss) |
| `technique_a`, `technique_b` | `configs/v3/v3_p0, p2..p6` | Thang kỹ thuật: cleaner, weighted BCE, mean / attention pooling, MSD (P1, P7, P8 tính sau từ xác suất đã lưu) |
| `seed1337`, `seed2024` | `configs/v3_seeds/*` | P0, P6, L2 với hai seed bổ sung, để đo nhiễu giữa các lần chạy |
| `d1_native` hoặc `gpu_d1_system_one[_noul]` | `configs/v3/v3_d1_system_one[_noul]` | Liquid d1-2B với API `system_one` gốc (cần `transformers>=5.19`; notebook tự cài) |

Các notebook `cpu_*` chạy từng cấu hình một trên CPU, chỉ dùng khi hết hạn mức GPU hoặc để đối chiếu chéo; không trộn kết quả CPU vào `outputs/v3_*`.

```bash
python scripts/build_package_notebooks.py                        # build notebook v3
python $S push --acc acc6 --gpu kaggle/<bundle>                  # đẩy lên Kaggle (mỗi tài khoản tối đa 2 phiên GPU song song)
python $S wait --poll 240 --out kaggle/outputs                   # chờ xong, tải kết quả
python scripts/collect_results.py --source kaggle/outputs --overwrite   # gom vào outputs/<experiment>/
python scripts/postprocess_analysis.py --all                     # hiệu chuẩn + đường cong tỷ lệ tự động hóa
python -I -X utf8 scripts/build_v3_report_tables.py              # in bảng cho báo cáo
python -I -X utf8 scripts/generate_benchmark_figures.py          # vẽ doc/figures/v3_*.png
python -I -X utf8 scripts/generate_architecture_diagrams.py     # vẽ doc/figures/arch_*.png (sơ đồ kiến trúc dạng nghiên cứu)
```
Lưu ý khi chạy: mỗi tài khoản Kaggle có hạn mức GPU 30 giờ mỗi tuần; tài khoản chưa xác minh số điện thoại không có Internet
trong kernel nên không tải được dataset và mô hình từ Hugging Face (kernel báo `Couldn't reach 'cardiffnlp/tweet_topic_multi' on the Hub`).
Kết quả GPU trên Kaggle (T4/P100) dùng fp16 cho d1; CPU dùng fp32, nên không so sánh từng chữ số giữa hai môi trường.

---

## 4. Bảng So Sánh Kỳ Vọng Giữa 4 Baseline

> Lưu ý: các giá trị "kỳ vọng" trong bảng này là ước lượng ban đầu, **chưa phải kết quả đo**. Kết quả đo nằm trong
> `outputs/` và `doc/MASTER_TWEET_TOPIC_SYSTEM_REPORT.md`.

| Đặc điểm | ML (TF-IDF + OvR) | BERT (Twitter-RoBERTa) | Decision Model (`d1-liquid`) | LLM (`Qwen3.5-4B`) |
|---|---|---|---|---|
| **Cơ chế hoạt động** | N-gram sparse features | Pre-trained dense encoder | Zero-token State-to-Decision | Autoregressive generative prompt |
| **Output** | Logistic probabilities | Logits $\to$ Sigmoid | Calibrated choice probabilities | JSON list of strings |
| **Yêu cầu GPU trên Kaggle** | Không bắt buộc (CPU đủ) | Bắt buộc (T4 / P100) | Bắt buộc (T4) hoặc API | Bắt buộc (T4 VRAM 16GB, 4-bit) |
| **Tốc độ suy luận** | Cực nhanh (< 1ms/sample) | Rất nhanh (~10ms/sample) | Rất nhanh (~15ms/sample) | Trung bình (~100-300ms/sample) |
| **Độ chính xác kỳ vọng (Macro-F1)** | ~30% - 35% | ~55% - 59% (SOTA benchmark) | ~48% - 54% | ~50% - 56% |
| **Khả năng chống suy giảm thời gian (Temporal drift)** | Trung bình | Tốt nhất khi dùng checkpoint 2021 | Rất tốt | Rất tốt nhờ tri thức rộng |
