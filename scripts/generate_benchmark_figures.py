"""Draws the benchmark figures from *measured* results only.

Sources
-------
* Legacy runs whose numbers are recorded in the Kaggle logs (B1 and B2). They are listed in ``LEGACY`` below with the
  exact values printed by those logs.
* The curated final runs in ``outputs/v2_results/<experiment>.json`` (copies of the ``metrics_summary.json`` of the
  full Kaggle runs; smoke runs are deliberately excluded). New experiments appear in the figures automatically.

Nothing is hard-coded for models that have not been measured: a model/split without a number is simply absent from
the chart. Usage:  python scripts/generate_benchmark_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "doc" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (label, {split: (micro_f1, macro_f1)})  -- values copied from outputs/multi_kaggle_results/*/*.log
LEGACY: List[tuple] = [
    ("B1 TF-IDF + LR (OvR)", {"test_2020": (0.6890, 0.5227), "test_2021": (0.6154, 0.3997)}),
    ("B2 RoBERTa (topic-multi ckpt)*", {"test_2020": (0.6659, 0.2955), "test_2021": (0.6502, 0.2648)}),
]


LABELS = {
    "v2_roberta_dec2020_bce": "RoBERTa clean ckpt, BCE",
    "v2_roberta_dec2020_bce_tempscale": "RoBERTa clean ckpt, BCE + temp. scaling",
    "v2_roberta_dec2020_asl_perclass": "RoBERTa, ASL + per-class thr.",
    "v2_roberta_dec2020_attnpool_msd_asl": "RoBERTa, attn-pool + MSD + ASL",
    "v2_d1_probe": "Liquid d1 linear probe",
    "v2_d1_zero_shot_letters": "Liquid d1 zero-shot (option letters)",
    "v2_qwen_qlora_cls": "Qwen3.5-4B QLoRA classifier",
    "v2_ml_ovr_preproc_audit": "TF-IDF + preprocessor",
}


def load_v2_results() -> List[tuple]:
    rows = []
    for path in sorted((ROOT / "outputs" / "v2_results").glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        splits = {
            name: (m["micro_f1"], m["macro_f1"])
            for name, m in data.get("test_metrics", {}).items()
            if name in ("test_2020", "test_2021")
        }
        if splits:
            name = data.get("experiment_name", path.stem)
            rows.append((LABELS.get(name, name), splits))
    return rows


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"


def leaderboard(rows: List[tuple]) -> None:
    labels = [r[0] for r in rows]
    x = np.arange(len(rows))
    w = 0.2
    series = [
        ("Micro-F1 2020 (in-time)", "test_2020", 0, "#1f77b4"),
        ("Micro-F1 2021 (out-of-time)", "test_2021", 0, "#aec7e8"),
        ("Macro-F1 2020 (in-time)", "test_2020", 1, "#2ca02c"),
        ("Macro-F1 2021 (out-of-time)", "test_2021", 1, "#98df8a"),
    ]
    fig, ax = plt.subplots(figsize=(max(9.0, 1.6 * len(rows) + 4), 5.5), dpi=200)
    for k, (name, split, idx, color) in enumerate(series):
        vals = [r[1].get(split, (np.nan, np.nan))[idx] for r in rows]
        bars = ax.bar(x + (k - 1.5) * w, vals, w, label=name, color=color)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 0.85)
    ax.set_title("In-time (test_2020) vs out-of-time (test_2021)")
    ax.legend(fontsize=8)
    fig.text(0.01, 0.01, "* checkpoint already fine-tuned on train_all (includes 2021): out-of-time column is not clean.",
             fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT_DIR / "benchmark_leaderboard_in_time_vs_out_time.png")
    plt.close(fig)


def drift(rows: List[tuple]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200, sharex=True)
    for ax, (idx, name) in zip(axes, [(0, "Micro-F1"), (1, "Macro-F1")]):
        for label, splits in rows:
            if "test_2020" in splits and "test_2021" in splits:
                ax.plot(["test_2020", "test_2021"], [splits["test_2020"][idx], splits["test_2021"][idx]],
                        marker="o", label=label)
        ax.set_title(name)
        ax.set_ylabel(name)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.suptitle("Change from 2020 to 2021 per model")
    fig.tight_layout(rect=(0, 0.17, 1, 0.95))
    fig.savefig(OUT_DIR / "temporal_drift_resilience.png")
    plt.close(fig)


def coverage_curves() -> Optional[Path]:
    """Coverage vs micro-F1 on the automated part, for every run that recorded a `selective` block."""
    runs: Dict[str, list] = {}
    for path in sorted((ROOT / "outputs" / "v2_results").glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        curve = data.get("selective", {}).get("test_2021")
        if curve:
            name = data.get("experiment_name", path.stem)
            runs[LABELS.get(name, name)] = curve
    if not runs:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=200)
    for name, curve in runs.items():
        ax.plot([r["coverage"] for r in curve], [r["auto_micro_f1"] for r in curve], marker="o", label=name)
    ax.invert_xaxis()
    ax.set_xlabel("coverage (fraction automated, most confident first)")
    ax.set_ylabel("micro-F1 on automated part")
    ax.set_title("Selective classification on test_2021")
    ax.legend(fontsize=7)
    fig.tight_layout()
    dest = OUT_DIR / "coverage_quality_curve.png"
    fig.savefig(dest)
    plt.close(fig)
    return dest


def ece_chart() -> Optional[Path]:
    """Micro-ECE on test_2021 before and after calibration, for every run that recorded it."""
    items = []
    for path in sorted((ROOT / "outputs" / "v2_results").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        cal = data.get("calibration")
        if cal and "test_2021_raw" in cal:
            name = data.get("experiment_name", path.stem)
            items.append((LABELS.get(name, name), cal["test_2021_raw"]["micro_ece"], cal["test_2021_calibrated"]["micro_ece"]))
    if not items:
        return None
    items.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=200)
    y = np.arange(len(items))
    ax.barh(y + 0.2, [i[1] for i in items], 0.4, label="before temperature scaling", color="#d62728")
    ax.barh(y - 0.2, [i[2] for i in items], 0.4, label="after temperature scaling", color="#2ca02c")
    ax.set_yticks(y)
    ax.set_yticklabels([i[0] for i in items], fontsize=8)
    ax.set_xlabel("micro-ECE on test_2021 (lower = better calibrated)")
    ax.set_title("Probability calibration")
    ax.legend(fontsize=8)
    fig.tight_layout()
    dest = OUT_DIR / "calibration_ece.png"
    fig.savefig(dest)
    plt.close(fig)
    return dest


def main() -> None:
    _style()
    rows = LEGACY + load_v2_results()
    leaderboard(rows)
    drift(rows)
    cov = coverage_curves()
    ece = ece_chart()
    print(f"Drew {len(rows)} models.", "Coverage curve:" if cov else "No coverage data.", cov or "", "ECE:", ece or "none")


if __name__ == "__main__":
    main()
