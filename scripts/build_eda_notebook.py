"""Builds the 00_kaggle_eda_tweet_topic.ipynb Kaggle Notebook."""

import json
from pathlib import Path

def make_cell(src, ctype="code"):
    return {
        "cell_type": ctype,
        "metadata": {},
        "outputs": [],
        "source": "\n".join(src) if isinstance(src, list) else src
    }

cells = [
    make_cell([
        "# Exploratory Data Analysis (EDA): cardiffnlp/tweet_topic_multi",
        "**Dataset:** [cardiffnlp/tweet_topic_multi](https://huggingface.co/datasets/cardiffnlp/tweet_topic_multi)",
        "",
        "This notebook explores the official TweetTopic multi-label dataset:",
        "- 1. Dataset splits overview (train_2020, val_2020, test_2020, test_2021)",
        "- 2. Class frequency and imbalance ratio",
        "- 3. Multi-label cardinality and label distribution",
        "- 4. Label co-occurrence and Jaccard similarity matrix",
        "- 5. Temporal topic drift (2020 vs 2021 shifts)"
    ], "markdown"),
    make_cell([
        "!pip install -q datasets matplotlib seaborn scipy pandas numpy"
    ]),
    make_cell([
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "from datasets import load_dataset",
        "from scipy.spatial.distance import jensenshannon",
        "",
        "sns.set_theme(style='whitegrid')",
        "print('Loading cardiffnlp/tweet_topic_multi dataset from Hugging Face...')",
        "ds = load_dataset('cardiffnlp/tweet_topic_multi')",
        "print(ds)"
    ]),
    make_cell([
        "TOPIC_CLASSES = [",
        "    'arts_&_culture', 'business_&_entrepreneurs', 'celebrity_&_pop_culture',",
        "    'diaries_&_daily_life', 'family', 'fashion_&_style', 'film_tv_&_video',",
        "    'fitness_&_health', 'food_&_dining', 'gaming', 'learning_&_educational',",
        "    'music', 'news_&_social_concern', 'other_hobbies', 'relationships',",
        "    'science_&_technology', 'sports', 'travel_&_adventure', 'youth_&_student_life'",
        "]",
        "",
        "def get_stats(split_name):",
        "    split = ds[split_name]",
        "    labels = np.array([x['label'] for x in split])",
        "    texts = [x['text'] for x in split]",
        "    cardinality = np.mean(labels.sum(axis=1))",
        "    class_freqs = labels.sum(axis=0) / len(labels)",
        "    word_lens = [len(t.split()) for t in texts]",
        "    return {",
        "        'n': len(split),",
        "        'cardinality': cardinality,",
        "        'freqs': class_freqs,",
        "        'labels': labels,",
        "        'word_lens': word_lens,",
        "    }",
        "",
        "splits = ['train_2020', 'validation_2020', 'test_2020', 'test_2021']",
        "stats = {s: get_stats(s) for s in splits}",
        "pd.DataFrame({s: {'Samples': stats[s]['n'], 'Cardinality': round(stats[s]['cardinality'], 3)} for s in splits}).T"
    ]),
    make_cell([
        "# 1. Label Frequencies Across Splits",
        "plt.figure(figsize=(12, 9))",
        "df_freq = pd.DataFrame({s: stats[s]['freqs'] for s in splits}, index=TOPIC_CLASSES)",
        "df_freq.sort_values(by='train_2020', ascending=True).plot(kind='barh', figsize=(12, 10), width=0.8)",
        "plt.title('Topic Frequency Across Splits (Proportion of Tweets)', fontsize=14, fontweight='bold')",
        "plt.xlabel('Proportion')",
        "plt.show()"
    ]),
    make_cell([
        "# 2. Number of Active Topics per Tweet (Label Cardinality)",
        "plt.figure(figsize=(10, 5))",
        "for s in ['train_2020', 'test_2020', 'test_2021']:",
        "    counts = pd.Series(stats[s]['labels'].sum(axis=1)).value_counts(normalize=True).sort_index()",
        "    plt.plot(counts.index, counts.values, marker='o', label=s)",
        "plt.title('Proportion of Tweets by Number of Assigned Labels', fontsize=14, fontweight='bold')",
        "plt.xlabel('Number of Labels')",
        "plt.ylabel('Proportion of Tweets')",
        "plt.legend()",
        "plt.show()"
    ]),
    make_cell([
        "# 3. Topic Label Jaccard Co-occurrence Matrix",
        "Y = stats['train_2020']['labels']",
        "co_occur = Y.T @ Y",
        "diag = np.diag(co_occur)",
        "denom = diag[:, None] + diag[None, :] - co_occur",
        "jaccard = np.divide(co_occur, denom, out=np.zeros_like(co_occur, dtype=float), where=denom != 0)",
        "",
        "plt.figure(figsize=(14, 12))",
        "sns.heatmap(jaccard, xticklabels=TOPIC_CLASSES, yticklabels=TOPIC_CLASSES, cmap='YlGnBu', annot=True, fmt='.2f')",
        "plt.title('Topic Co-occurrence Jaccard Index (train_2020)', fontsize=14, fontweight='bold')",
        "plt.xticks(rotation=45, ha='right')",
        "plt.show()"
    ]),
    make_cell([
        "# 4. Temporal Drift (2020 vs 2021)",
        "diff = stats['test_2021']['freqs'] - stats['train_2020']['freqs']",
        "plt.figure(figsize=(12, 8))",
        "colors = ['#2ecc71' if d >= 0 else '#e74c3c' for d in diff]",
        "plt.barh(TOPIC_CLASSES, diff, color=colors)",
        "plt.axvline(0, color='black', linestyle='--')",
        "plt.title('Temporal Drift: Change from 2020 to 2021 (test_2021 - train_2020)', fontsize=14, fontweight='bold')",
        "plt.xlabel('Delta in Frequency')",
        "plt.gca().invert_yaxis()",
        "plt.show()"
    ])
]

nb = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "nbformat": 4,
    "nbformat_minor": 4
}
dest = Path("tweet_topic_benchmark/notebooks/00_kaggle_eda_tweet_topic.ipynb")
dest.write_text(json.dumps(nb, indent=2), encoding="utf-8")
print("Generated:", dest)
