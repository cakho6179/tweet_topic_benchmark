"""Automated Exploratory Data Analysis (EDA) for cardiffnlp/tweet_topic_multi."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from scipy.spatial.distance import jensenshannon

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eda_runner")

EDA_DIR = Path(__file__).resolve().parent
FIGURES_DIR = EDA_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TOPIC_CLASSES = [
    "arts_&_culture",
    "business_&_entrepreneurs",
    "celebrity_&_pop_culture",
    "diaries_&_daily_life",
    "family",
    "fashion_&_style",
    "film_tv_&_video",
    "fitness_&_health",
    "food_&_dining",
    "gaming",
    "learning_&_educational",
    "music",
    "news_&_social_concern",
    "other_hobbies",
    "relationships",
    "science_&_technology",
    "sports",
    "travel_&_adventure",
    "youth_&_student_life",
]


def analyze_split(split_data) -> Dict[str, Any]:
    texts = [x["text"] for x in split_data]
    labels = np.array([x["label"] for x in split_data], dtype=int)
    n_samples, n_classes = labels.shape

    # Class counts
    class_counts = labels.sum(axis=0)
    class_freqs = class_counts / n_samples

    # Multi-label statistics
    labels_per_sample = labels.sum(axis=1)
    cardinality = float(np.mean(labels_per_sample))
    density = float(cardinality / n_classes)
    
    count_dist = pd.Series(labels_per_sample).value_counts().sort_index().to_dict()

    # Text length stats
    char_lens = [len(t) for t in texts]
    word_lens = [len(t.split()) for t in texts]

    # Twitter artifact detection
    has_mention = sum(1 for t in texts if "@" in t or "{@" in t) / n_samples
    has_url = sum(1 for t in texts if "{{URL}}" in t or "http" in t) / n_samples
    has_hashtag = sum(1 for t in texts if "#" in t) / n_samples

    return {
        "n_samples": n_samples,
        "class_counts": class_counts.tolist(),
        "class_freqs": class_freqs.tolist(),
        "cardinality": round(cardinality, 4),
        "density": round(density, 4),
        "label_count_dist": count_dist,
        "char_len_stats": {
            "min": int(np.min(char_lens)),
            "mean": round(float(np.mean(char_lens)), 2),
            "median": round(float(np.median(char_lens)), 2),
            "max": int(np.max(char_lens)),
            "std": round(float(np.std(char_lens)), 2),
        },
        "word_len_stats": {
            "min": int(np.min(word_lens)),
            "mean": round(float(np.mean(word_lens)), 2),
            "median": round(float(np.median(word_lens)), 2),
            "max": int(np.max(word_lens)),
            "std": round(float(np.std(word_lens)), 2),
        },
        "artifacts": {
            "mention_pct": round(has_mention * 100, 2),
            "url_pct": round(has_url * 100, 2),
            "hashtag_pct": round(has_hashtag * 100, 2),
        },
        "labels_matrix": labels,
        "texts": texts,
    }


def plot_eda(data_dict: Dict[str, Any]):
    sns.set_theme(style="whitegrid")
    
    # 1. Label distribution comparison across splits
    plt.figure(figsize=(14, 8))
    df_freq = pd.DataFrame({
        "train_2020": data_dict["train_2020"]["class_freqs"],
        "validation_2020": data_dict["validation_2020"]["class_freqs"],
        "test_2020": data_dict["test_2020"]["class_freqs"],
        "test_2021": data_dict["test_2021"]["class_freqs"],
    }, index=TOPIC_CLASSES)

    df_freq.plot(kind="barh", figsize=(12, 10), width=0.8)
    plt.title("Topic Class Frequency Across Dataset Splits", fontsize=14, fontweight="bold")
    plt.xlabel("Frequency (Proportion of Tweets)", fontsize=12)
    plt.ylabel("Topic Class", fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "label_distribution_by_split.png", dpi=200)
    plt.close()

    # 2. Label Cardinality Distribution
    plt.figure(figsize=(10, 5))
    splits_to_plot = ["train_2020", "test_2020", "test_2021"]
    card_data = []
    for s in splits_to_plot:
        dist = data_dict[s]["label_count_dist"]
        for count, num in dist.items():
            card_data.append({"Split": s, "Number of Labels": count, "Count": num})
    
    df_card = pd.DataFrame(card_data)
    sns.barplot(data=df_card, x="Number of Labels", y="Count", hue="Split")
    plt.title("Distribution of Active Topics Per Tweet", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Topics Assigned to a Single Tweet", fontsize=12)
    plt.ylabel("Number of Tweets", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "label_cardinality_distribution.png", dpi=200)
    plt.close()

    # 3. Label Co-occurrence & Jaccard Similarity Heatmap
    Y_train = data_dict["train_2020"]["labels_matrix"]
    co_occur = Y_train.T @ Y_train
    diag = np.diag(co_occur)
    denom = diag[:, None] + diag[None, :] - co_occur
    jaccard = np.divide(co_occur, denom, out=np.zeros_like(co_occur, dtype=float), where=denom != 0)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        jaccard,
        xticklabels=TOPIC_CLASSES,
        yticklabels=TOPIC_CLASSES,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
    )
    plt.title("Topic Label Jaccard Co-occurrence Matrix (train_2020)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "label_cooccurrence_heatmap.png", dpi=200)
    plt.close()

    # 4. Temporal Shift: 2020 vs 2021
    p2020 = np.array(data_dict["train_2020"]["class_freqs"])
    p2021 = np.array(data_dict["test_2021"]["class_freqs"])
    diff = p2021 - p2020
    
    plt.figure(figsize=(12, 8))
    colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in diff]
    bars = plt.barh(TOPIC_CLASSES, diff, color=colors)
    plt.axvline(0, color="black", linestyle="--", alpha=0.7)
    plt.title("Temporal Topic Drift: Frequency Change from 2020 to 2021 (test_2021 - train_2020)", fontsize=14, fontweight="bold")
    plt.xlabel("Delta in Proportion (+ denotes increase in 2021)", fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "temporal_shift_2020_vs_2021.png", dpi=200)
    plt.close()

    # 5. Text Length Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot([len(t.split()) for t in data_dict["train_2020"]["texts"]], bins=40, kde=True, color="#3498db")
    plt.title("Word Length Distribution of Tweets (train_2020)", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Words per Tweet", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "text_length_distribution.png", dpi=200)
    plt.close()


def generate_markdown_report(data_dict: Dict[str, Any]) -> str:
    train_2020 = data_dict["train_2020"]
    test_2021 = data_dict["test_2021"]
    
    # Calculate JS Divergence for temporal shift
    p_2020 = np.array(train_2020["class_freqs"])
    p_2021 = np.array(test_2021["class_freqs"])
    # Normalize to probability distributions
    p_2020_norm = p_2020 / np.sum(p_2020)
    p_2021_norm = p_2021 / np.sum(p_2021)
    js_div = float(jensenshannon(p_2020_norm, p_2021_norm))

    # Top & bottom classes in train_2020
    counts = train_2020["class_counts"]
    sorted_indices = np.argsort(counts)[::-1]
    
    max_c_idx = sorted_indices[0]
    min_c_idx = sorted_indices[-1]
    imbalance_ratio = round(counts[max_c_idx] / max(1, counts[min_c_idx]), 2)

    report = f"""# Báo Cáo EDA Chuyên Sâu: `cardiffnlp/tweet_topic_multi`

Bài báo gốc: *"Twitter Topic Classification"*, Dimosthenis Antypas et al., COLING 2022.  
Mục đích: Khảo sát cấu trúc phân bố, tính chất đa nhãn (multi-label), đặc trưng ngôn ngữ mạng xã hội và độ lệch trôi dạt thời gian (temporal drift).

---

## 1. Thống Kê Tổng Quan Các Tập Dữ Liệu (Dataset Splits)

| Tên Split | Số lượng mẫu | Label Cardinality (Nhãn/tweet) | Label Density | Tweet Chứa Mention (%) | Tweet Chứa URL (%) |
|---|---|---|---|---|---|
| `train_2020` | **{data_dict['train_2020']['n_samples']:,}** | {data_dict['train_2020']['cardinality']} | {data_dict['train_2020']['density']} | {data_dict['train_2020']['artifacts']['mention_pct']}% | {data_dict['train_2020']['artifacts']['url_pct']}% |
| `validation_2020` | **{data_dict['validation_2020']['n_samples']:,}** | {data_dict['validation_2020']['cardinality']} | {data_dict['validation_2020']['density']} | {data_dict['validation_2020']['artifacts']['mention_pct']}% | {data_dict['validation_2020']['artifacts']['url_pct']}% |
| `test_2020` | **{data_dict['test_2020']['n_samples']:,}** | {data_dict['test_2020']['cardinality']} | {data_dict['test_2020']['density']} | {data_dict['test_2020']['artifacts']['mention_pct']}% | {data_dict['test_2020']['artifacts']['url_pct']}% |
| `test_2021` | **{data_dict['test_2021']['n_samples']:,}** | {data_dict['test_2021']['cardinality']} | {data_dict['test_2021']['density']} | {data_dict['test_2021']['artifacts']['mention_pct']}% | {data_dict['test_2021']['artifacts']['url_pct']}% |

---

## 2. Phân Bố 19 Chủ Đề & Tỷ Lệ Mất Cân Bằng (Class Imbalance)

- **Lớp phổ biến nhất (Head):** `{TOPIC_CLASSES[max_c_idx]}` với {counts[max_c_idx]:,} mẫu ({train_2020['class_freqs'][max_c_idx]*100:.1f}% số tweet).
- **Lớp hiếm nhất (Tail):** `{TOPIC_CLASSES[min_c_idx]}` với {counts[min_c_idx]:,} mẫu ({train_2020['class_freqs'][min_c_idx]*100:.2f}% số tweet).
- **Tỷ lệ mất cân bằng (Imbalance Ratio):** $\\approx \\mathbf{{{imbalance_ratio}:1}}$.

### Bảng Chi Tiết Tần Số Nhãn Trên Các Split:

| ID | Tên Chủ Đề (Topic) | `train_2020` Count | `train_2020` % | `test_2020` % | `test_2021` % |
|---|---|---|---|---|---|
"""
    for idx in sorted_indices:
        c_name = TOPIC_CLASSES[idx]
        c_tr = train_2020["class_counts"][idx]
        p_tr = train_2020["class_freqs"][idx] * 100
        p_te20 = data_dict["test_2020"]["class_freqs"][idx] * 100
        p_te21 = test_2021["class_freqs"][idx] * 100
        report += f"| `{idx}` | `{c_name}` | {c_tr:,} | {p_tr:.1f}% | {p_te20:.1f}% | {p_te21:.1f}% |\n"

    report += f"""
---

## 3. Đặc Điểm Đa Nhãn (Multi-Label Properties)

- **Label Cardinality:** Trung bình mỗi tweet có **{train_2020['cardinality']}** chủ đề.
- **Label Density:** **{train_2020['density']}** (trung bình một tweet chỉ kích hoạt {train_2020['density']*100:.1f}% trong tổng số 19 chiều nhãn).
- **Phân phối số nhãn kích hoạt:**
"""
    for n_labels, cnt in train_2020["label_count_dist"].items():
        pct = (cnt / train_2020["n_samples"]) * 100
        report += f"  - `{n_labels}` nhãn: {cnt:,} tweet ({pct:.1f}%)\n"

    report += f"""
> [!IMPORTANT]
> Vì hơn 90% các vị trí nhãn trong ma trận $Y$ là số 0 (nhãn âm), hàm mất mát tiêu chuẩn `BCEWithLogitsLoss` dễ bị áp đảo bởi các mẫu âm tính dễ (easy negatives). Đây là lý do kiến trúc cần tích hợp kỹ thuật **Asymmetric Loss (ASL)** hoặc **Focal Loss** để cân bằng gradient.

---

## 4. Phân Tích Độ Lệch Thời Gian (Temporal Drift: 2020 vs 2021)

- **Khoảng cách Jensen-Shannon Divergence ($JS$):** **{js_div:.4f}**
- Các chủ đề có sự dịch chuyển lớn nhất giữa 2020 và 2021:
"""
    drift_diff = p_2021 - p_2020
    drift_sorted = np.argsort(np.abs(drift_diff))[::-1]
    for d_idx in drift_sorted[:5]:
        cls_name = TOPIC_CLASSES[d_idx]
        delta_pct = drift_diff[d_idx] * 100
        sign = "+" if delta_pct > 0 else ""
        report += f"  - `{cls_name}`: {sign}{delta_pct:.2f}% (Từ {p_2020[d_idx]*100:.1f}% năm 2020 sang {p_2021[d_idx]*100:.1f}% năm 2021)\n"

    report += f"""
---

## 5. Thống Kê Chiều Dài Văn Bản & Đặc Trưng Ngôn Ngữ

- **Độ dài từ (Word Count):** Trung bình {train_2020['word_len_stats']['mean']} từ/tweet (Trung vị: {train_2020['word_len_stats']['median']}, Nhỏ nhất: {train_2020['word_len_stats']['min']}, Lớn nhất: {train_2020['word_len_stats']['max']}).
- **Độ dài ký tự (Char Count):** Trung bình {train_2020['char_len_stats']['mean']} ký tự/tweet (Max: {train_2020['char_len_stats']['max']}).
- **Artifacts mạng xã hội:**
  - {train_2020['artifacts']['mention_pct']}% tweet chứa `@mention` (được chuẩn hóa dạng `{{@username@}}` hoặc `@user`).
  - {train_2020['artifacts']['url_pct']}% tweet chứa liên kết (chuẩn hóa dạng `{{{{URL}}}}`).
  - {train_2020['artifacts']['hashtag_pct']}% tweet chứa hashtag `#`.

---

## 6. Danh Mục Biểu Đồ EDA Đã Tạo

Tất cả biểu đồ độ phân giải cao đã được xuất tại thư mục `eda/figures/`:
1. `label_distribution_by_split.png`: Tần suất 19 chủ đề trên cả 4 split.
2. `label_cardinality_distribution.png`: Phân bố số lượng nhãn/tweet.
3. `label_cooccurrence_heatmap.png`: Ma trận tương quan Jaccard giữa các cặp chủ đề.
4. `temporal_shift_2020_vs_2021.png`: Biểu đồ delta biến động chủ đề giữa 2020 và 2021.
5. `text_length_distribution.png`: Biểu đồ histogram độ dài từ của tweet.
"""
    return report


def main():
    logger.info("Loading cardiffnlp/tweet_topic_multi for comprehensive EDA...")
    ds = load_dataset("cardiffnlp/tweet_topic_multi")
    
    splits = ["train_2020", "validation_2020", "test_2020", "test_2021"]
    data_dict = {}
    for s in splits:
        logger.info("Analyzing split: %s (%d rows)...", s, len(ds[s]))
        data_dict[s] = analyze_split(ds[s])

    logger.info("Plotting EDA figures to %s...", FIGURES_DIR)
    plot_eda(data_dict)

    logger.info("Generating EDA report...")
    report_md = generate_markdown_report(data_dict)
    
    report_path = EDA_DIR / "eda_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    logger.info("Saved EDA report to: %s", report_path)

    # Save machine-readable JSON
    json_summary = {
        s: {k: v for k, v in data_dict[s].items() if k not in ["labels_matrix", "texts"]}
        for s in splits
    }
    json_path = EDA_DIR / "eda_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2)
    logger.info("Saved EDA summary JSON to: %s", json_path)


if __name__ == "__main__":
    main()
