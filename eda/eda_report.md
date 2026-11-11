# Báo Cáo EDA Chuyên Sâu: `cardiffnlp/tweet_topic_multi`

Bài báo gốc: *"Twitter Topic Classification"*, Dimosthenis Antypas et al., COLING 2022.  
Mục đích: Khảo sát cấu trúc phân bố, tính chất đa nhãn (multi-label), đặc trưng ngôn ngữ mạng xã hội và độ lệch trôi dạt thời gian (temporal drift).

---

## 1. Thống Kê Tổng Quan Các Tập Dữ Liệu (Dataset Splits)

| Tên Split | Số lượng mẫu | Label Cardinality (Nhãn/tweet) | Label Density | Tweet Chứa Mention (%) | Tweet Chứa URL (%) |
|---|---|---|---|---|---|
| `train_2020` | **4,585** | 1.708 | 0.0899 | 68.75% | 48.14% |
| `validation_2020` | **573** | 1.7347 | 0.0913 | 69.11% | 49.56% |
| `test_2020` | **573** | 1.6475 | 0.0867 | 68.76% | 48.52% |
| `test_2021` | **1,679** | 1.5509 | 0.0816 | 71.77% | 43.95% |

---

## 2. Phân Bố 19 Chủ Đề & Tỷ Lệ Mất Cân Bằng (Class Imbalance)

- **Lớp phổ biến nhất (Head):** `news_&_social_concern` với 1,493 mẫu (32.6% số tweet).
- **Lớp hiếm nhất (Tail):** `fashion_&_style` với 74 mẫu (1.61% số tweet).
- **Tỷ lệ mất cân bằng (Imbalance Ratio):** $\approx \mathbf{20.18:1}$.

### Bảng Chi Tiết Tần Số Nhãn Trên Các Split:

| ID | Tên Chủ Đề (Topic) | `train_2020` Count | `train_2020` % | `test_2020` % | `test_2021` % |
|---|---|---|---|---|---|
| `12` | `news_&_social_concern` | 1,493 | 32.6% | 34.0% | 19.5% |
| `16` | `sports` | 1,151 | 25.1% | 25.1% | 34.7% |
| `11` | `music` | 769 | 16.8% | 16.6% | 22.6% |
| `3` | `diaries_&_daily_life` | 731 | 15.9% | 13.4% | 8.9% |
| `6` | `film_tv_&_video` | 722 | 15.7% | 15.7% | 17.7% |
| `2` | `celebrity_&_pop_culture` | 714 | 15.6% | 16.4% | 14.6% |
| `0` | `arts_&_culture` | 256 | 5.6% | 5.8% | 2.9% |
| `7` | `fitness_&_health` | 231 | 5.0% | 3.5% | 3.3% |
| `15` | `science_&_technology` | 229 | 5.0% | 6.6% | 5.2% |
| `4` | `family` | 222 | 4.8% | 2.8% | 1.9% |
| `14` | `relationships` | 209 | 4.6% | 3.3% | 3.6% |
| `1` | `business_&_entrepreneurs` | 208 | 4.5% | 4.2% | 4.6% |
| `13` | `other_hobbies` | 195 | 4.3% | 4.4% | 4.8% |
| `9` | `gaming` | 189 | 4.1% | 3.5% | 4.6% |
| `10` | `learning_&_educational` | 131 | 2.9% | 2.6% | 1.7% |
| `8` | `food_&_dining` | 130 | 2.8% | 1.7% | 0.9% |
| `17` | `travel_&_adventure` | 94 | 2.1% | 2.1% | 0.9% |
| `18` | `youth_&_student_life` | 83 | 1.8% | 1.9% | 0.7% |
| `5` | `fashion_&_style` | 74 | 1.6% | 1.0% | 1.9% |

---

## 3. Đặc Điểm Đa Nhãn (Multi-Label Properties)

- **Label Cardinality:** Trung bình mỗi tweet có **1.708** chủ đề.
- **Label Density:** **0.0899** (trung bình một tweet chỉ kích hoạt 9.0% trong tổng số 19 chiều nhãn).
- **Phân phối số nhãn kích hoạt:**
  - `0` nhãn: 1 tweet (0.0%)
  - `1` nhãn: 2,210 tweet (48.2%)
  - `2` nhãn: 1,661 tweet (36.2%)
  - `3` nhãn: 572 tweet (12.5%)
  - `4` nhãn: 124 tweet (2.7%)
  - `5` nhãn: 15 tweet (0.3%)
  - `6` nhãn: 2 tweet (0.0%)

> [!IMPORTANT]
> Vì hơn 90% các vị trí nhãn trong ma trận $Y$ là số 0 (nhãn âm), hàm mất mát tiêu chuẩn `BCEWithLogitsLoss` dễ bị áp đảo bởi các mẫu âm tính dễ (easy negatives). Đây là lý do kiến trúc cần tích hợp kỹ thuật **Asymmetric Loss (ASL)** hoặc **Focal Loss** để cân bằng gradient.

---

## 4. Phân Tích Độ Lệch Thời Gian (Temporal Drift: 2020 vs 2021)

- **Khoảng cách Jensen-Shannon Divergence ($JS$):** **0.1377**
- Các chủ đề có sự dịch chuyển lớn nhất giữa 2020 và 2021:
  - `news_&_social_concern`: -13.09% (Từ 32.6% năm 2020 sang 19.5% năm 2021)
  - `sports`: +9.56% (Từ 25.1% năm 2020 sang 34.7% năm 2021)
  - `diaries_&_daily_life`: -7.07% (Từ 15.9% năm 2020 sang 8.9% năm 2021)
  - `music`: +5.86% (Từ 16.8% năm 2020 sang 22.6% năm 2021)
  - `family`: -2.94% (Từ 4.8% năm 2020 sang 1.9% năm 2021)

---

## 5. Thống Kê Chiều Dài Văn Bản & Đặc Trưng Ngôn Ngữ

- **Độ dài từ (Word Count):** Trung bình 28.34 từ/tweet (Trung vị: 27.0, Nhỏ nhất: 6, Lớn nhất: 61).
- **Độ dài ký tự (Char Count):** Trung bình 171.48 ký tự/tweet (Max: 364).
- **Artifacts mạng xã hội:**
  - 68.75% tweet chứa `@mention` (được chuẩn hóa dạng `{@username@}` hoặc `@user`).
  - 48.14% tweet chứa liên kết (chuẩn hóa dạng `{{URL}}`).
  - 39.28% tweet chứa hashtag `#`.

---

## 6. Danh Mục Biểu Đồ EDA Đã Tạo

Tất cả biểu đồ độ phân giải cao đã được xuất tại thư mục `eda/figures/`:
1. `label_distribution_by_split.png`: Tần suất 19 chủ đề trên cả 4 split.
2. `label_cardinality_distribution.png`: Phân bố số lượng nhãn/tweet.
3. `label_cooccurrence_heatmap.png`: Ma trận tương quan Jaccard giữa các cặp chủ đề.
4. `temporal_shift_2020_vs_2021.png`: Biểu đồ delta biến động chủ đề giữa 2020 và 2021.
5. `text_length_distribution.png`: Biểu đồ histogram độ dài từ của tweet.
