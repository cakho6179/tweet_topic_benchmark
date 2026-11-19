"""Draws the benchmark figures from *measured* results only.

Sources
-------
* Legacy runs whose numbers are recorded in the Kaggle logs (B1 and B2), listed in ``LEGACY`` with the exact
  values printed by those logs.
* The curated Vòng 2 runs in ``outputs/v2_results/<experiment>.json``.
* The Vòng 3 runs in ``outputs/v3_*/metrics_summary.json`` (plus ``training_history.json``, ``postprocess.json``).
  P1/P7/P8 are derived from the saved probabilities of P0/P6, not from new training runs.

Nothing is hard-coded for models that have not been measured: a model/split without a number is simply absent
from the chart. Usage:

    python scripts/postprocess_analysis.py --all --output-dir outputs/v3_results
    python scripts/generate_benchmark_figures.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

V3_LABELS = {
    "v3_l1_bce": "L1 BCE",
    "v3_l2_weighted_bce": "L2 Weighted BCE",
    "v3_l3_focal_g2": "L3 Focal (γ=2, α=0.25)",
    "v3_l4_focal_g1": "L4 Focal (γ=1, α=0.25)",
    "v3_l5_focal_g2_noalpha": "L5 Focal (γ=2, α=None)",
    "v3_p0_custom_cls_bce": "P0 custom CLS + BCE",
    "v3_p2_cleaner_bce": "P2 + cleaner",
    "v3_p3_cleaner_weighted_bce": "P3 + weighted BCE",
    "v3_p4_cleaner_weighted_mean": "P4 + mean pooling",
    "v3_p5_cleaner_weighted_attention": "P5 + attention pooling",
    "v3_p6_cleaner_weighted_attn_msd": "P6 + MSD",
    "v3_d1_system_one": "d1 native choice",
    "v3_d1_system_one_noul": "d1 native 19 × noul",
}

LOSS_ORDER = [
    "v3_l1_bce",
    "v3_l2_weighted_bce",
    "v3_l3_focal_g2",
    "v3_l4_focal_g1",
    "v3_l5_focal_g2_noalpha",
]

LADDER_ORDER = [
    "v3_p0_custom_cls_bce",
    "v3_p2_cleaner_bce",
    "v3_p3_cleaner_weighted_bce",
    "v3_p4_cleaner_weighted_mean",
    "v3_p5_cleaner_weighted_attention",
    "v3_p6_cleaner_weighted_attn_msd",
]

STEP_LABELS = {
    "v3_p0_custom_cls_bce": "P0",
    "v3_p2_cleaner_bce": "P2",
    "v3_p3_cleaner_weighted_bce": "P3",
    "v3_p4_cleaner_weighted_mean": "P4",
    "v3_p5_cleaner_weighted_attention": "P5",
    "v3_p6_cleaner_weighted_attn_msd": "P6",
}

STEP_NAMES = {
    "v3_p0_custom_cls_bce": "P0 CLS + BCE",
    "v3_p2_cleaner_bce": "P2 + cleaner",
    "v3_p3_cleaner_weighted_bce": "P3 + weighted BCE",
    "v3_p4_cleaner_weighted_mean": "P4 + mean pool",
    "v3_p5_cleaner_weighted_attention": "P5 + attention pool",
    "v3_p6_cleaner_weighted_attn_msd": "P6 + MSD",
}


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def load_v2_results(outputs_root: Path) -> List[tuple]:
    rows = []
    for path in sorted((outputs_root / "v2_results").glob("*.json")):
        data = _read_json(path)
        if not data:
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


def load_v3_runs(outputs_root: Path) -> List[Dict[str, Any]]:
    """Every finished ``v3_*`` run, keyed by experiment name."""
    runs: Dict[str, Dict[str, Any]] = {}
    for metrics_path in sorted(outputs_root.glob("v3_*/metrics_summary.json")):
        summary = _read_json(metrics_path)
        if not summary or "test_metrics" not in summary:
            continue
        name = summary.get("experiment_name", metrics_path.parent.name)
        runs[name] = {"name": name, "dir": metrics_path.parent, "summary": summary}
    # Curated copies (outputs/v3_results/<name>.json) override nothing but fill gaps.
    for path in sorted((outputs_root / "v3_results").glob("*.json")):
        summary = _read_json(path)
        if not summary or "test_metrics" not in summary:
            continue
        name = summary.get("experiment_name", path.stem)
        run_dir = outputs_root / name
        runs.setdefault(name, {"name": name, "dir": run_dir if run_dir.exists() else path.parent, "summary": summary})
    return [runs[k] for k in sorted(runs)]


def _v3_postprocess(run: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cached = run["dir"] / "postprocess.json"
    if cached.exists():
        return _read_json(cached)
    try:
        from scripts.postprocess_analysis import DEFAULT_COVERAGES, analyze_run

        return analyze_run(run["dir"], DEFAULT_COVERAGES)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] postprocess unavailable for {run['name']}: {exc}")
        return None


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"


def _annotate_bars(ax, bars, values, fontsize: float = 7.0) -> None:
    for bar, value in zip(bars, values):
        if not np.isnan(value):
            ax.annotate(
                f"{value:.2f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=fontsize,
            )


# ----------------------------------------------------------------------------------------------------------------
# Vòng 2 figures
# ----------------------------------------------------------------------------------------------------------------
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
        _annotate_bars(ax, bars, vals)
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


def coverage_curves(outputs_root: Path) -> Optional[Path]:
    """Coverage vs micro-F1 on the automated part, for every Vòng 2 run that recorded a `selective` block."""
    runs: Dict[str, list] = {}
    for path in sorted((outputs_root / "v2_results").glob("*.json")):
        data = _read_json(path)
        if not data:
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
    ax.set_title("Selective classification on test_2021 (Vòng 2)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    dest = OUT_DIR / "coverage_quality_curve.png"
    fig.savefig(dest)
    plt.close(fig)
    return dest


def ece_chart(outputs_root: Path) -> Optional[Path]:
    """Micro-ECE on test_2021 before and after calibration, for every Vòng 2 run that recorded it."""
    items = []
    for path in sorted((outputs_root / "v2_results").glob("*.json")):
        data = _read_json(path)
        if not data:
            continue
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
    ax.set_title("Probability calibration (Vòng 2)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    dest = OUT_DIR / "calibration_ece.png"
    fig.savefig(dest)
    plt.close(fig)
    return dest


# ----------------------------------------------------------------------------------------------------------------
# Vòng 3 figures
# ----------------------------------------------------------------------------------------------------------------
def v3_loss_ablation(runs: List[Dict[str, Any]]) -> Optional[Path]:
    by_name = {r["name"]: r for r in runs}
    rows = [by_name[name] for name in LOSS_ORDER if name in by_name]
    if not rows:
        return None

    labels = [V3_LABELS[r["name"]] for r in rows]
    x = np.arange(len(rows))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=200)
    for ax, idx, title in [(axes[0], 0, "Micro-F1"), (axes[1], 1, "Macro-F1")]:
        v2020 = [r["summary"]["test_metrics"].get("test_2020", {}).get("micro_f1" if idx == 0 else "macro_f1", np.nan) for r in rows]
        v2021 = [r["summary"]["test_metrics"].get("test_2021", {}).get("micro_f1" if idx == 0 else "macro_f1", np.nan) for r in rows]
        b20 = ax.bar(x - w / 2, v2020, w, label="test_2020 (in-time)", color="#4c78a8")
        b21 = ax.bar(x + w / 2, v2021, w, label="test_2021 (out-of-time)", color="#f58518")
        _annotate_bars(ax, b20, v2020)
        _annotate_bars(ax, b21, v2021)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_ylim(0, 0.85)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Vòng 3 — Loss ablation on bert-base-uncased (global threshold)")
    fig.tight_layout()
    dest = OUT_DIR / "v3_loss_ablation.png"
    fig.savefig(dest)
    plt.close(fig)
    return dest


def v3_ladder_rows(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cumulative progressive ablation: P0 global, P1-P6 per-class, P7 temperature-scaled per-class."""
    by_name = {r["name"]: r for r in runs}
    rows: List[Dict[str, Any]] = []

    def _add(step: str, label: str, metrics: Dict[str, Dict[str, Any]]) -> None:
        rows.append({
            "step": step,
            "label": label,
            "test_2020": metrics.get("test_2020", {}),
            "test_2021": metrics.get("test_2021", {}),
        })

    p0 = by_name.get("v3_p0_custom_cls_bce")
    if p0 is not None:
        _add("P0", "P0 CLS + BCE", p0["summary"].get("test_metrics", {}))
        comparison = p0["summary"].get("threshold_comparison", {})
        _add(
            "P1",
            "P1 + per-class thr.",
            {split: comparison.get(split, {}).get("per_class_tuned", {}) for split in ("test_2020", "test_2021")},
        )

    for name in LADDER_ORDER[1:]:
        run = by_name.get(name)
        if run is None:
            continue
        comparison = run["summary"].get("threshold_comparison", {})
        _add(
            STEP_LABELS[name],
            STEP_NAMES[name],
            {split: comparison.get(split, {}).get("per_class_tuned", {}) for split in ("test_2020", "test_2021")},
        )

    p6 = by_name.get("v3_p6_cleaner_weighted_attn_msd")
    if p6 is not None:
        post = _v3_postprocess(p6)
        if post:
            _add(
                "P7",
                "P7 + temp. scaling",
                {split: post.get("test", {}).get(split, {}).get("calibrated", {}).get("per_class_tuned", {})
                 for split in ("test_2020", "test_2021")},
            )
    return rows


def v3_progressive_ladder(runs: List[Dict[str, Any]]) -> Optional[Path]:
    rows = v3_ladder_rows(runs)
    if not rows:
        return None
    steps = [r["step"] for r in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), dpi=200, sharey=False)
    for ax, split, title in [(axes[0], "test_2020", "test_2020 (in-time)"), (axes[1], "test_2021", "test_2021 (out-of-time)")]:
        micro = [r[split].get("micro_f1", np.nan) for r in rows]
        macro = [r[split].get("macro_f1", np.nan) for r in rows]
        ax.plot(x, micro, marker="o", label="Micro-F1", color="#1f77b4")
        ax.plot(x, macro, marker="s", label="Macro-F1", color="#2ca02c")
        for xi, value in zip(x, micro):
            if not np.isnan(value):
                ax.annotate(f"{value:.2f}", (xi, value), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['step']}\n{r['label'].split(' ', 1)[1]}" for r in rows], fontsize=7)
        ax.set_ylabel("F1")
        ax.set_title(title)
        ax.set_ylim(0, 0.9)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Vòng 3 — Progressive technique ablation (P0 global, P1-P7 per-class thresholds)")
    fig.tight_layout()
    dest = OUT_DIR / "v3_progressive_ladder.png"
    fig.savefig(dest)
    plt.close(fig)
    return dest


def v3_training_curves(runs: List[Dict[str, Any]]) -> Optional[Path]:
    by_name = {r["name"]: r for r in runs}
    selected = [by_name[name] for name in LADDER_ORDER if name in by_name]
    histories = {}
    for run in selected:
        history = _read_json(run["dir"] / "training_history.json") or run["summary"].get("training_history")
        if history:
            histories[run["name"]] = history
    if not histories:
        return None

    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(histories)))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), dpi=200)
    panels = [
        ("train_loss", "Train loss"),
        ("val_macro_f1_tuned", "Validation Macro-F1 (tuned)"),
        ("val_bce_loss", "Validation BCE loss"),
    ]
    for ax, (key, title) in zip(axes, panels):
        for color, (name, history) in zip(colors, histories.items()):
            xs = [h["epoch"] for h in history if key in h]
            ys = [h[key] for h in history if key in h]
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", ms=4, color=color, label=STEP_NAMES.get(name, name))
            best_epoch = by_name[name]["summary"].get("best_epoch")
            if best_epoch in xs:
                ax.scatter([best_epoch], [ys[xs.index(best_epoch)]], s=90, facecolors="none",
                           edgecolors="black", linewidths=1.2, zorder=5)
        ax.set_xlabel("epoch")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(sorted({h["epoch"] for history in histories.values() for h in history}))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Vòng 3 — Progressive ablation training logs (circles mark the selected checkpoint)")
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    dest = OUT_DIR / "v3_training_curves.png"
    fig.savefig(dest)
    plt.close(fig)
    return dest


def v3_coverage_curve(runs: List[Dict[str, Any]]) -> Optional[Path]:
    by_name = {r["name"]: r for r in runs}
    chosen = [name for name in ("v3_l1_bce", "v3_p6_cleaner_weighted_attn_msd") if name in by_name]
    curves: List[tuple] = []
    for name in chosen:
        post = _v3_postprocess(by_name[name])
        if not post:
            continue
        calibrated = post.get("coverage", {}).get("test_2021", {}).get("calibrated", {}).get("margin")
        raw = post.get("coverage", {}).get("test_2021", {}).get("raw", {}).get("margin")
        if calibrated:
            curves.append((f"{V3_LABELS.get(name, name)} — calibrated", calibrated, "-"))
        if raw:
            curves.append((f"{V3_LABELS.get(name, name)} — raw", raw, "--"))
    if not curves:
        return None

    fig, ax = plt.subplots(figsize=(7.6, 4.9), dpi=200)
    for label, curve, linestyle in curves:
        ax.plot([r["coverage"] for r in curve], [r["auto_micro_f1"] for r in curve],
                marker="o", linestyle=linestyle, label=label)
    ax.invert_xaxis()
    ax.set_xlabel("coverage (fraction automated, most confident first)")
    ax.set_ylabel("micro-F1 on automated part")
    ax.set_title("Vòng 3 — Selective classification on test_2021 (margin)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    dest = OUT_DIR / "v3_coverage_curve.png"
    fig.savefig(dest)
    plt.close(fig)
    return dest


def v3_ece_chart(runs: List[Dict[str, Any]]) -> Optional[Path]:
    items = []
    for run in runs:
        post = _v3_postprocess(run)
        if not post:
            continue
        cal = post.get("calibration", {}).get("test_2021")
        if cal:
            items.append((
                V3_LABELS.get(run["name"], run["name"]),
                cal["raw"]["micro_ece"],
                cal["per_class_temperature"]["micro_ece"],
            ))
    if not items:
        return None
    items.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(items) + 2.0), dpi=200)
    y = np.arange(len(items))
    ax.barh(y + 0.2, [i[1] for i in items], 0.4, label="raw", color="#d62728")
    ax.barh(y - 0.2, [i[2] for i in items], 0.4, label="per-class temperature", color="#2ca02c")
    ax.set_yticks(y)
    ax.set_yticklabels([i[0] for i in items], fontsize=8)
    ax.set_xlabel("micro-ECE on test_2021 (lower = better calibrated)")
    ax.set_title("Vòng 3 — Probability calibration (post-hoc)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    dest = OUT_DIR / "v3_calibration_ece.png"
    fig.savefig(dest)
    plt.close(fig)
    return dest


def v3_head_vs_loss(runs: List[Dict[str, Any]]) -> Optional[Path]:
    """Macro-F1 (test_2021) of every loss under the stock head versus the P0 head (same loss, only the head differs)."""
    by_name = {r["name"]: r["summary"]["test_metrics"].get("test_2021", {}) for r in runs}
    losses = [
        ("BCE", "v3_l1_bce", "v3_p0_custom_cls_bce"),
        ("Weighted BCE", "v3_l2_weighted_bce", "v3_h2_weighted_bce"),
        ("Focal γ=2, α=0.25", "v3_l3_focal_g2", "v3_h3_focal_g2"),
        ("Focal γ=1, α=0.25", "v3_l4_focal_g1", "v3_h4_focal_g1"),
        ("Focal γ=2, không α", "v3_l5_focal_g2_noalpha", "v3_h5_focal_g2_noalpha"),
    ]
    if not all(a in by_name and b in by_name for _, a, b in losses):
        return None
    x = np.arange(len(losses))
    w = 0.38
    stock = [by_name[a].get("macro_f1", np.nan) for _, a, _ in losses]
    custom = [by_name[b].get("macro_f1", np.nan) for _, _, b in losses]
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=200)
    b1 = ax.bar(x - w / 2, stock, w, label="Head mặc định của thư viện (L1-L5)", color="#9aa5b1")
    b2 = ax.bar(x + w / 2, custom, w, label="Head tùy chỉnh như P0 (BCE = P0, còn lại H2-H5)", color="#2a9d8f")
    _annotate_bars(ax, b1, stock, fontsize=8)
    _annotate_bars(ax, b2, custom, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([l for l, _, _ in losses], rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("Macro-F1 trên test_2021")
    ax.set_ylim(0, 0.75)
    ax.set_title("Cùng một loss, đổi head: khác biệt giữa các loss co lại nhiều")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    dest = OUT_DIR / "v3_head_vs_loss.png"
    fig.savefig(dest)
    plt.close(fig)
    return dest


def v3_seed_spread(runs: List[Dict[str, Any]]) -> Optional[Path]:
    """Mean ± sd over seeds {42, 1337, 2024} for the headline configurations (test_2021)."""
    by_name = {r["name"]: r["summary"]["test_metrics"].get("test_2021", {}) for r in runs}
    groups = [
        ("P0\nBERT + BCE", "v3_p0_custom_cls_bce"),
        ("P6\n+ cleaner, weighted BCE,\nattention, MSD", "v3_p6_cleaner_weighted_attn_msd"),
        ("L2\nhead chuẩn + weighted BCE", "v3_l2_weighted_bce"),
    ]
    data = []
    for label, base in groups:
        names = [base, base + "_s1337", base + "_s2024"]
        if not all(n in by_name for n in names):
            return None
        data.append((label, [[by_name[n].get(k, np.nan) for n in names] for k in ("micro_f1", "macro_f1")]))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=200)
    colors = ["#4c78a8", "#2a9d8f", "#9aa5b1"]
    for ax, idx, title in [(axes[0], 0, "Micro-F1 (test_2021)"), (axes[1], 1, "Macro-F1 (test_2021)")]:
        for i, (label, series) in enumerate(data):
            vals = np.array(series[idx], dtype=float)
            ax.bar(i, vals.mean(), 0.55, yerr=vals.std(ddof=1), capsize=5, color=colors[i], alpha=0.85)
            ax.scatter([i] * len(vals), vals, color="black", s=14, zorder=3)
            ax.annotate(f"{vals.mean():.3f}", (i, vals.mean()), xytext=(0, 14), textcoords="offset points", ha="center", fontsize=8)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels([d[0] for d in data], fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0.4 if idx else 0.55, 0.78 if not idx else 0.66)
    fig.suptitle("Ba seed (42, 1337, 2024): chấm đen là từng seed, cột là trung bình ± độ lệch chuẩn")
    fig.tight_layout()
    dest = OUT_DIR / "v3_seed_spread.png"
    fig.savefig(dest)
    plt.close(fig)
    return dest


def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser(description="Draw benchmark figures from measured JSON artifacts")
    parser.add_argument("--outputs-root", default=str(ROOT / "outputs"), help="Directory containing run artifacts")
    parser.add_argument("--figures-dir", default=str(OUT_DIR), help="Directory for the generated PNG files")
    args = parser.parse_args()
    outputs_root = Path(args.outputs_root).resolve()

    OUT_DIR = Path(args.figures_dir).resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _style()
    rows = LEGACY + load_v2_results(outputs_root)
    leaderboard(rows)
    drift(rows)
    cov = coverage_curves(outputs_root)
    ece = ece_chart(outputs_root)

    v3runs = load_v3_runs(outputs_root)
    v3_loss = v3_loss_ablation(v3runs)
    v3_ladder = v3_progressive_ladder(v3runs)
    v3_curves = v3_training_curves(v3runs)
    v3_cov = v3_coverage_curve(v3runs)
    v3_ece = v3_ece_chart(v3runs)
    v3_head = v3_head_vs_loss(v3runs)
    v3_seeds = v3_seed_spread(v3runs)

    print(f"Drew {len(rows)} Vòng 2 models.")
    print(f"Vòng 3 runs: {len(v3runs)}")
    for label, path in [
        ("coverage", cov),
        ("ECE", ece),
        ("v3 loss ablation", v3_loss),
        ("v3 progressive ladder", v3_ladder),
        ("v3 training curves", v3_curves),
        ("v3 coverage", v3_cov),
        ("v3 ECE", v3_ece),
        ("v3 head vs loss", v3_head),
        ("v3 seed spread", v3_seeds),
    ]:
        print(f"  {label}: {path or 'no data'}")


if __name__ == "__main__":
    main()
