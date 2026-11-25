# Root-Cause Error Diagnosis trên Tập Validation (`validation_2020`, N = 573)

**Phạm vi đánh giá:** `validation_2020` (`573` tweets; `311` failed samples của Baseline `S4 @ 0.60` với `532` binary errors = `246 FP + 286 FN`) và phân tích từ vựng đối chiếu từ `train_2020` (`4,585` tweets).  
**Giao thức Zero Test Leakage:** `test_splits_accessed = []` — hoàn toàn không truy cập `test_2020` hay `test_2021`.  
**Dữ liệu tổng hợp:** `outputs/val_2020_forensic_root_cause.json` (Multi-Model Consensus trên 5 mô hình huấn luyện 100% trên Kaggle GPU: `S4 seed 42`, `S4 seed 1337`, `S4 seed 2024`, `ValFix-2B`, `ValFix-2A`) kèm đối chiếu chi tiết 311 failed samples trên `validation_2020`.

---

## 1. Phân Rã 532 Binary Errors Thành 4 Tầng qua 5-Model Consensus Audit

Thay vì chỉ đếm lỗi trên một lần chạy đơn lẻ (`S4 seed 42`), từng quyết định nhị phân được đối chiếu đồng thời trên **5 mô hình BERT độc lập** (`3` random seeds của `S4` + `2` cấu hình cải tiến `ValFix-2B, ValFix-2A`, cùng chuẩn Global Threshold `τ = 0.60`). Phương pháp **Multi-Model Consensus Audit** tách bạch định lượng giữa **Label-Noise / Data Ceiling**, **Structural Bias**, **Architectural Headroom** và **Seed Variance**:

| Error Stratum (Mutually Exclusive) | Errors / `532` | Share | FP / FN | Inter-Model Std ($\sigma_5$) | Primary Root Cause |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Stratum 1 — Unanimous Extreme `5/5`** (`mean5 ≥ 0.85` FP hoặc `≤ 0.15` FN) | **179** | **33.6%** | `73` / `106` | **`0.0349`** | **Crowd-Annotation Noise & Bayes Ceiling:** Lỗi gán nhãn đám đông, thiếu nhãn phụ hiển nhiên của annotator, hoặc Spurious Lexical Trigger cực mạnh. |
| **Stratum 2 — Unanimous Moderate `5/5`** (`0.15 < mean5 < 0.85`) | **161** | **30.3%** | `51` / `110` | **`0.0866`** | **Structural Bias:** Secondary-Label Suppression (`110 FN`, gấp đôi FP), Label Cardinality Compression (`k ≥ 2`), Co-occurrence Halo, nhóm `no_entity`. |
| **Stratum 3 — Majority Boundary `3–4/5`** (`3` hoặc `4` models sai) | **121** | **22.7%** | `68` / `53` | **`0.1641`** | **Architectural & Calibration Headroom:** Vùng `CleanerV2 + Last4 + LabelInteract + LS` cải thiện rõ nhất (`relationships +16.7`, `arts +11.0`, `travel +10.5`, `learning +6.4` F1). |
| **Stratum 4 — Seed Variance `1–2/5`** (`1` hoặc `2` models sai) | **71** | **13.3%** | **`54` / `17`** | **`0.1648`** | **Random-Seed Variance (76% là FP):** Dao động xác suất quanh ngưỡng `0.60`, dễ triệt tiêu bằng probability averaging. |
| **Total (`S4 @ 0.60`)** | **532** | **100.0%** | **246 / 286** | — | **`63.9%` (`340/532`) đồng thuận `5/5`; `75.0%` (`399/532`) đồng thuận `≥ 4/5`.** |

**Phân rã ở cấp độ tweet (`N = 573`):**
* **Consensus Easy Core (Exact Match trên cả `5/5` models):** `205 / 573` (`35.8%`).
* **Volatile Boundary Zone (fail ở `1–4` models):** `126 / 573` (`22.0%`) — *không gian cải thiện thực tế của kiến trúc, preprocessing và ensemble*.
* **Consensus Hard Core (fail trên cả `5/5` models):** **`242 / 573` (`42.2%`)** — giải thích vì sao các mô hình đơn lẻ tiệm cận trần Exact Match quanh `46%–48%` trên `validation_2020`.

> **Bất đối xứng định lượng giữa `FP` và `FN`:**
> * **`FN` mang tính cấu trúc cứng:** **`216 / 286` (`75.5%`) tổng FN là đồng thuận `5/5`** (`fashion` `8/8 = 100%`, `sports` `9/9 = 100%`, `other_hobbies` `19/21 = 90.5%`, `celebrity` `30/36 = 83.3%`, `news` `26/35 = 74.3%`).
> * **`FP` nhạy mạnh với variance:** Gần một nửa tổng FP (`122 / 246 = 49.6%`) không đồng thuận `5/5`. Bằng chứng trực tiếp: chỉ cần **probability averaging của 3 random seeds `S4` sẵn có (`42 + 1337 + 2024`)**, FP trên `validation_2020` giảm ngay từ **`246 → 201` (`-45 FP`)**, đưa hiệu năng lên **`0.7434` Micro-F1 (`+1.65`) / `0.6611` Macro-F1 (`+2.83`) / `48.34%` Exact Match (`+2.62%`)** trong khi `FN` giữ nguyên (`286 → 287`).

---

## 2. Root Cause #1 — Crowd-Annotation Noise & Bayes Error Ceiling trên `validation_2020`

Kiểm tra nguyên văn các mẫu **Stratum 1 (`5/5` models đồng thuận, $\sigma_5 = 0.035$)** cho thấy nhiều ca "high-confidence error" thực chất bắt nguồn từ **annotation noise trong ground truth crowdsourced của CardiffNLP**:

### 2.1. Top Unanimous `5/5` "FP" (`mean5_prob ≥ 0.97`) — Model Chính xác Hơn Ground Truth

| `ValIdx` | Raw Tweet (`validation_2020`) | Ground Truth | `S4` Prediction (`mean5` of FP class) | Error Diagnosis |
| :---: | :--- | :--- | :--- | :--- |
| **494** | *"Wayment Rick Ross’ world play in Entanglements got me hypeeeeeeeeee `{@August Alsina@}` `{@Yung Rnzl@}`"* | `sports` | `celebrity (0.975), music (0.988)` | **Mislabeled GT:** Ca khúc *Entanglements* của August Alsina & Rick Ross thuộc rap/music & celebrity, không liên quan `sports`. |
| **447** | *"Glad to know our Odisha recovering from to the bad times. Braving Fani... `{@CMO Odisha@}` `{@Naveen Patnaik@}`"* | `other_hobbies` | `news (0.987)` | **Mislabeled GT:** Tin khắc phục hậu quả bão Fani & Thủ hiến bang Odisha bị gán nhầm `other_hobbies`. |
| **24** | *"New video! 890 Jump cinematic!... `{@Star Citizen@}` `{@Cloud Imperium Games@}` #StarCitizen"* | `film_tv` | `gaming (0.997)` | **Missing Positive GT:** *Star Citizen* của *Cloud Imperium Games* là tựa game PC nổi tiếng. |
| **298** | *"NSU Staff: Pass all NSU students for the semester due to COVID-19 - Sign the Petition!"* | `learning, news` | `+ youth (0.992)` | **Incomplete Multi-Labeling:** Kiến nghị cho sinh viên vượt kỳ học nhưng thiếu `youth_&_student_life`. |
| **556** | *"understand the condition of students... #SwamyJiCancelUniversityExams `{@UGC INDIA@}`"* | `youth` | `+ learning (0.971)` | **Incomplete Multi-Labeling (đối xứng #298):** Hủy kỳ thi đại học nhưng thiếu `learning_&_educational`. |
| **486** | *"The next morning.... my mom can’t stop talking about how much she loves `{{USERNAME}}`"* | `diaries` | `+ family (0.977)` | **Missing Positive GT:** Đề cập trực tiếp *"my mom"* (tương tự `ValIdx 164` nhắc *"Blanca's family"* 2 lần nhưng GT bỏ sót `family`). |

### 2.2. Top Unanimous `5/5` "FN" (`mean5_prob < 0.01`) — Ground Truth Nhiễu theo Surface Words

| `ValIdx` | Raw Tweet (`validation_2020`) | Ground Truth | `S4` Prediction (`mean5` of FN class) | Error Diagnosis |
| :---: | :--- | :--- | :--- | :--- |
| **112** | *"Watermelon sugar high!! #LSS `{@Harry Styles@}` #WatermelonSugar"* | `celebrity, food, music` | `celebrity, music` (`food = 0.0045`) | **Surface-Word Annotation Bias:** Annotator gán nhầm `food` dựa vào cụm *"Watermelon Sugar"* (tên bài hát). |
| **114** | *"#RiQiNetMusicNewSongs Harry Styles - Watermelon Sugar `{{URL}}`"* | `family, music` | `celebrity, music` (`family = 0.0052`) | **Spurious GT Label:** Link bài hát thuần túy nhưng bị gán `family`. |
| **111** | *"Watch Harry Styles - Watermelon Sugar (Live on SNL) on YouTube"* | `celebrity, fashion, film_tv, music` | `celebrity, film_tv, music` (`fashion = 0.0066`) | **Spurious GT Label:** Không có yếu tố thời trang nhưng GT vẫn gắn `fashion`. |
| **523** | *"The Chinese Invasion Threat: Taiwan’s Defense and American Strategy in Asia... military documents"* | `film_tv, news, relationships` | `news` (`relationships = 0.0014`, `film_tv = 0.0080`) | **Label Definition Confusion:** Nhầm lẫn *international relations* (Mỹ–Đài) với `relationships` (quan hệ cá nhân). |
| **369** | *"Cloudpunk... may be one of my fav games of the yr. Likely the best cyberpunk game to date"* | `film_tv, gaming, sports` | `gaming` (`sports = 0.0065`) | **Spurious GT Label:** Review game cyberpunk bị gán nhầm `sports`. |
| **276** | *"The Guardian: Win or lose, Pete Buttigieg has made history in the White House race."* | `news, other_hobbies` | `news` (`hobbies = 0.0090`) | **Spurious GT Label:** Tin bầu cử Mỹ bị gán nhầm `other_hobbies`. |

> **Diagnostic Insight #1:** Một tỷ trọng đáng kể lỗi ở Stratum 1 bắt nguồn từ crowd-annotation noise và sự thiếu nhất quán ở ranh giới giữa các cặp nhãn gần nghĩa (`learning ↔ youth`, `diaries ↔ family`, `film_tv ↔ gaming`). Vì vậy, ép `train_loss` giảm quá sâu hoặc tối ưu 19 per-class thresholds trên `validation_2020` sẽ **overfit vào annotation noise** thay vì cải thiện generalization thực.

---

## 3. Root Cause #2 — Spurious Lexical Triggers & Entity Shortcut

Tính **Log2-Odds Ratio** trên `train_2020` cho các token xuất hiện trong các mẫu FP trên `validation_2020` bóc tách rõ cơ chế BERT bị kích hoạt sai bởi từ khóa có tương quan giả cao:

| FP Class (`val_2020`) | FP Count | Top `train_2020` Trigger Tokens (`Log2-Odds`) | Failure Mechanism (Verified Examples) |
| :--- | :---: | :--- | :--- |
| **`celebrity`** | **39** | `bts` (`+2.66`, 6 FP), `twt` (`+2.53`, 4 FP), `official` (`+1.73`, 5 FP), `{@Entity@}` (`FPR 9.3%` vs `5.9%`, **`1.57×`**) | Entity Shortcut + Co-occurrence Halo: `48.9%` (`23/47`) tweet `music` không có `celebrity` vẫn bị gán nhầm, đẩy $P(\text{celeb}\mid\text{music})$ từ `47.8%` lên **`67.8%`**. |
| **`film_tv`** | **27** | `video` (`+2.67`, 5 FP), `youtube` (`+2.22`, 6 FP), `official` (`+2.00`, 5 FP) | Mọi tweet chia sẻ music video trên YouTube bị kéo sang `film_tv` (`23.4%` spillover `music → film_tv`, `15 FP`). |
| **`learning`** | **19** | `school` (`+4.10`), `conversation` (`+3.31`), `read` (`+3.28`), `article` (`+2.60`) | Overconfidence Gap **`+46.3%`** (`Prec = 36.7%`): bài báo tin tức có cụm *"read this article"* hoặc *"school reopening"* kích hoạt FP `learning` (`13/19` FP đi từ `news`). |
| **`gaming`** | **13** | `games` (`+4.20`), `graphics` (`+3.80`), `game` (`+3.20`, **5/13 FP**) | Từ đa nghĩa **`game`** trong ngữ cảnh thể thao (*"Game 5"*, *"Game day"*, *"exhibition game"*, *"#WarGames"*) kích hoạt `gaming` (`7` FP từ `sports`). |
| **`fitness`** | **12** | `#covid19` (`+5.63`), `masks` (`+5.12`), `corona` (`+4.68`), `distancing` (`+4.54`) | Do `train_2020` gắn chặt từ vựng COVID-19 với `fitness_&_health`, các tin xã hội về khẩu trang/giãn cách bị kéo sang `fitness` (`10/12` FP từ `news`). |
| **`news`** | **16** | `emergency` (`+3.37`), `vote` (`+2.32`) | Điển hình **`ValIdx 40`** (*"EMERGENCY LAST 1.5 hours!!! PLEASE VOTE FOR EXO"*): cặp `emergency + vote` đẩy hẳn tweet fan K-pop sang **`news (0.994)`**, đè bẹp `music (0.012)` và `celebrity`. |

---

## 4. Root Cause #3 — Ba Nguyên nhân Cấu trúc của 286 False Negatives (`FN`)

Phân lớp `validation_2020` theo đặc trưng bề mặt chỉ rõ nguồn gốc của `286 FN`:

1. **No-Entity Performance Drop:**
   * Tweet **chứa `{@Entity@}`** (`n = 383`): Micro-F1 = **`0.7557`**, Exact Match = **`50.13%`** (fail rate `49.9%`).
   * Tweet **không chứa `{@Entity@}`** (`n = 190`): Micro-F1 giảm **`-8.31` pts xuống `0.6726`**, Exact Match giảm **`-13.29%` xuống `36.84%`** (fail rate **`63.2%`**).
   * *Nguyên nhân:* Thiếu entity anchors (đội bóng, nghệ sĩ, tổ chức), các tweet đời sống/sở thích/tình cảm dựa vào ngữ điệu ngắn và ẩn ý khiến `diaries` (`42 FN`), `other_hobbies` (`21 FN`) và `relationships` (`14 FN`) tụt xác suất dưới `0.60`.
2. **Label Cardinality Compression:**
   * `k = 1` (`n = 269`): `mean_pred = 1.316` (**over-prediction**: `109 FP` vs `24 FN`, Exact `69.9%`).
   * `k = 2` (`n = 209`): `mean_pred = 1.871` (`129 FN`, Exact `27.8%`).
   * `k = 3` (`n = 74`): `mean_pred = 2.068` (`98 FN`, Exact `20.3%`, Micro `0.6613`).
   * `k ≥ 4` (`n = 21`): `mean_pred = 2.667` vs `4.048` (`35 FN`, Exact **`4.8%`**, fail rate **`95.2%`**).
   * **`91.6%` (`262/286`) tổng FN** tập trung ở nhóm tweet có **$\ge 2$ labels**.
3. **Secondary-Label Suppression:**
   * `sports + news` $\rightarrow$ giữ `sports`, bỏ sót `news` **`80.0%`** (`8/10`).
   * `sports + celebrity` $\rightarrow$ giữ `sports`, bỏ sót `celebrity` **`75.0%`** (`9/12`).
   * `sports + film_tv` $\rightarrow$ giữ `sports`, bỏ sót `film_tv` **`66.7%`** (`8/12`).
   * `diaries + relationships` $\rightarrow$ giữ `diaries`, bỏ sót `relationships` **`54.5%`** (`6/11`).
   * `news + diaries` $\rightarrow$ giữ `news`, bỏ sót `diaries` **`53.8%`** (`14/26`).
   * `news + fitness` $\rightarrow$ giữ `news`, bỏ sót `fitness` **`45.5%`** (`10/22`).

---

## 5. Diagnostic Autopsy của `ValFix-2B / ValFix-2A`: Cơ chế Recovery (`+25`) vs Regression (`-18`)

Phân tích trực tiếp kết quả huấn luyện trên Kaggle GPU của `ValFix-2B` (`0.5/6.0`) và `ValFix-2A` (`0.45/4.5, ls=0.02`) trên `validation_2020`:

### 5.1. Cơ chế Recovery (`+1.85` Macro-F1, `25` Mẫu Phục hồi lên Exact Match)
* **Cải thiện mạnh trên 6 lớp nhiễu/đuôi:** `relationships` **`+16.71` F1** (`0.412 → 0.579` nhờ `CleanerV2` chuẩn hóa `❤ -> "love"` + `LabelInteract`), `arts` **`+10.96` F1** (`14 → 8 FP`), `travel` **`+10.52` F1** (`0 FP`), `learning` **`+6.38` F1** (`19 → 10 FP`), `business` **`+5.18` F1** (`18 → 12 FP`), `fitness` **`+4.12` F1**.
* **Cấu trúc 25 mẫu phục hồi thành Exact Match (`ValFix-2A @ 0.55`):** Cắt **`19 FP`** (`arts -3`, `film_tv -3`, `diaries -3`, `business -2`, `family -2`, `celebrity -2`) và thu hồi **`12 FN`** (`news +3`, `celebrity +2`, `relationships +2`, `music +2`). Tiêu biểu: `ValIdx 422` (khử `[UNK]` `?♂` + `EmptyFallback` $\rightarrow$ Exact `news`), `ValIdx 234` (khôi phục `relationships` từ *"happy valentines day"*), `ValIdx 170` (loại FP `business`), `ValIdx 39` (khôi phục `celebrity` cho *Jeffree Star*).

### 5.2. Cơ chế Regression (`18` Mẫu Bị Hồi quy ở `ValFix-2A @ 0.55`)
* Trong `18` mẫu hồi quy, **`17 / 20` (`85%`) lỗi phát sinh là FP mới**, chỉ có `3` lỗi `FN` mới.
* **Nguyên nhân #1 — Tác dụng phụ của `Last-4 Layer Fusion` + Threshold `τ = 0.55` trên `gaming` (`F1 -11.33` pts):**
  * Các tầng giữa BERT (layers 9–10) nhạy hơn với surface lexical cues so với layer 12. Khi kết hợp với việc hạ threshold xuống `0.55`, token `"game"` trong các câu thể thao thuần (`ValIdx 1` *"decisive Game 5"*, `ValIdx 20` *"Game day!... exhibition game"*, `ValIdx 120` *"#WarGames WWE"*) tạo ra **`+4` regressions `gaming` FP**. *(Riêng `ValIdx 14` — "Cricket 19 PS4" — thực chất là tựa game PS4 nên `[gaming, sports]` chính xác hơn GT `[sports]`.)*
* **Nguyên nhân #2 — Threshold `0.55` làm tái phát Halo FP ở `celebrity` (`+4`) và `film_tv` (`+3`):**
  * **Kiểm chứng đối chứng:** Khi giữ nguyên trọng số `ValFix-2A` tại **`τ = 0.60`**, số FP hồi quy giảm quá nửa (**`17 → 8`**), số mẫu hồi quy giảm `18 → 14`, đưa **Exact Match lên đỉnh `47.47%` (`+10` net Exact)** và tổng binary errors xuống mức thấp nhất **`510`** (`-22` errors, `-37 FP`).

---

## 6. Tổng kết Đòn bẩy Kỹ thuật (High ROI vs Low ROI)

| Intervention / Strategy | Target Error Stratum | ROI Assessment | Empirical Evidence trên `validation_2020` |
| :--- | :---: | :---: | :--- |
| **1. Multi-Seed Probability Averaging (`3` seeds `S4`)** | Stratum 4 (`13.3%`, 76% FP) | **Rất cao (Verified)** | Trung bình `3` seeds `S4` cắt **`-45 FP`**, đạt **`0.7434 Mi (+1.65) / 0.6611 Ma (+2.83) / 48.34% EM (+2.62%)`** mà không cần thay đổi kiến trúc. |
| **2. `ValFix-2A @ 0.60` (`CleanerV2 + Last4 + LabelInteract + LS`)** | Stratum 3 (`22.7%`) | **Cao (Verified)** | Giảm `90.5%` `[UNK]`, cải thiện mạnh `relationships (+16.7)`, `arts (+11.0)`, `travel (+10.5)`, `learning (+6.4)`, cắt `-37 FP` (`EM 47.47%`); giữ `τ = 0.60` để tránh bùng FP `gaming`. |
| **3. Selective Prediction theo Decision Margin `|p − 0.60|`** | Stratum 3 + 4 | **Rất cao (Production)** | Chuyển `20%` mẫu margin thấp sang Human/LLM Escalation giúp nâng mạnh độ chính xác luồng tự động `80%` với chi phí tối thiểu. |
| **4. Xử lý Spurious Triggers & Label Cleaning (`Confident Learning`)** | Stratum 1 + 2 (`63.9%`) | **Trung bình – Khó** | Để vượt Consensus Hard Core (`242` mẫu), cần domain-adapted pre-training (như `TimeLMs`) hoặc lọc nhiễu nhãn trên `train_2020`. |
| **5. Unregularized Per-Class Thresholds trên `val` (`N = 573`)** | — | **Overfitting Trap** | Chỉ giảm tối đa `25` oracle errors trên `val` nhưng tụt `-10.4` pts Macro-F1 khi đánh giá OOT. |
