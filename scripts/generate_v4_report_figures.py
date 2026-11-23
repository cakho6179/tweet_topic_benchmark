"""Generates V4 diagnostic figures and computes multi-dimensional error analysis from measured JSON outputs."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tweet_topic.data.taxonomy import TOPIC_CLASSES  # noqa: E402

OUT_ROOT = ROOT / "outputs"
FIG_DIRS = [
    ROOT / "doc" / "figures",
    Path(r"D:\VSF\Sentiment_Topic_Classification\doc\figures"),
]

LADDER_CONFIGS = [
    ("S0", "S0: Stock Head + BCE", "v4_s0_stock_bce", "default"),
    ("S1", "S1: + Custom Head & Diff LR", "v4_s1_custom_head_bce", "default"),
    ("S2", "S2: + BertTweetCleaner", "v4_s2_cleaner_bce", "default"),
    ("S3", "S3: + Hybrid Pool & MSD", "v4_s3_hybrid_pool_msd_bce", "default"),
    ("S4", "S4: + Damped W-BCE", "v4_s4_damped_wbce", "default"),
    ("S5", "S5: + Shrinkage Threshold", "v4_s4_damped_wbce", "shrinkage_tuned"),
]

LOSS_CONFIGS = [
    ("BCE (S3)", "v4_s3_hybrid_pool_msd_bce"),
    ("Damped W-BCE (S4)", "v4_s4_damped_wbce"),
    ("Naive W-BCE (w=30)", "v4_loss_naive_wbce"),
    ("Focal (g=2, no alpha)", "v4_loss_focal_noalpha"),
    ("Asymmetric Loss", "v4_loss_asl"),
]


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save_to_all(fig: plt.Figure, filename: str) -> None:
    for d in FIG_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {filename}")


def _get_preds_and_probs(run_dir: Path, split: str, mode: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    summary = _load_json(run_dir / "metrics_summary.json")
    preds_payload = _load_json(run_dir / "test_predictions.json")
    assert summary is not None and preds_payload is not None, f"Missing artifacts in {run_dir}"
    split_d = preds_payload[split]
    yt = np.asarray(split_d["y_true"], dtype=int)
    probs = np.asarray(split_d.get("y_probs", split_d.get("y_probs_raw")), dtype=float)
    if mode == "default":
        yp = np.asarray(split_d["y_pred"], dtype=int)
    else:
        cmp_block = summary.get("threshold_comparison", {})
        if mode == "shrinkage_tuned":
            tau = np.asarray(cmp_block["tau_shrinkage"], dtype=float)
            yp = (probs >= tau[None, :]).astype(int)
        elif mode == "per_class_tuned":
            tau = np.asarray(cmp_block["tau_per_class"], dtype=float)
            yp = (probs >= tau[None, :]).astype(int)
        elif mode == "global_tuned":
            tau = float(cmp_block["tau_global"])
            yp = (probs >= tau).astype(int)
        elif mode == "default_0.5":
            yp = (probs >= 0.5).astype(int)
        else:
            yp = np.asarray(split_d["y_pred"], dtype=int)
    return yt, yp, summary


def compute_diagnostics(yt: np.ndarray, yp: np.ndarray) -> Dict[str, Any]:
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    p_mi, r_mi, f_mi, _ = precision_recall_fscore_support(yt, yp, average="micro", zero_division=0)
    p_ma, r_ma, f_ma, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
    per_f1 = f1_score(yt, yp, average=None, zero_division=0)
    per_p, per_r, _, _ = precision_recall_fscore_support(yt, yp, average=None, zero_division=0)
    dead = int(np.sum(per_f1 == 0))
    exact = float(np.mean(np.all(yt == yp, axis=1)))
    pred_card = float(np.mean(np.sum(yp, axis=1)))
    n_labels = np.sum(yt, axis=1)
    card1 = float(f1_score(yt[n_labels == 1], yp[n_labels == 1], average="micro", zero_division=0))
    card2 = float(f1_score(yt[n_labels == 2], yp[n_labels == 2], average="micro", zero_division=0))
    card3p = float(f1_score(yt[n_labels >= 3], yp[n_labels >= 3], average="micro", zero_division=0))
    per_class = {}
    for i, cname in enumerate(TOPIC_CLASSES):
        c_tp = int(np.sum((yt[:, i] == 1) & (yp[:, i] == 1)))
        c_fp = int(np.sum((yt[:, i] == 0) & (yp[:, i] == 1)))
        c_fn = int(np.sum((yt[:, i] == 1) & (yp[:, i] == 0)))
        per_class[cname] = {
            "support": int(np.sum(yt[:, i])),
            "tp": c_tp,
            "fp": c_fp,
            "fn": c_fn,
            "precision": round(float(per_p[i]), 4),
            "recall": round(float(per_r[i]), 4),
            "f1": round(float(per_f1[i]), 4),
        }
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "micro_p": round(float(p_mi), 4),
        "micro_r": round(float(r_mi), 4),
        "micro_f1": round(float(f_mi), 4),
        "macro_p": round(float(p_ma), 4),
        "macro_r": round(float(r_ma), 4),
        "macro_f1": round(float(f_ma), 4),
        "exact_match": round(exact, 4),
        "dead_classes": dead,
        "pred_cardinality": round(pred_card, 3),
        "card1_micro_f1": round(card1, 4),
        "card2_micro_f1": round(card2, 4),
        "card3p_micro_f1": round(card3p, 4),
        "per_class": per_class,
    }


def plot_epoch_convergence() -> None:
    targets = [
        ("S0: Stock Head + BCE", "v4_s0_stock_bce", "#7f8c8d", "--"),
        ("S1: Custom Head + BCE", "v4_s1_custom_head_bce", "#2b8cbe", "-"),
        ("S3: Hybrid Pool + MSD + BCE", "v4_s3_hybrid_pool_msd_bce", "#e67e22", "-"),
        ("S4: + Damped W-BCE", "v4_s4_damped_wbce", "#2ca02c", "-"),
        ("Naive W-BCE (w=30)", "v4_loss_naive_wbce", "#d62728", ":"),
    ]
    loaded = []
    for label, folder, color, ls in targets:
        s = _load_json(OUT_ROOT / folder / "metrics_summary.json")
        if s and s.get("training_history"):
            loaded.append((label, s, color, ls))
    if not loaded:
        print("Skipping v4_epoch_convergence.png (no histories found)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    panels = [
        ("train_loss", "Train Loss"),
        ("val_bce_loss", "Validation BCE Loss (Comparable Scale)"),
        ("val_macro_f1_shrinkage", "Validation Macro-F1 (Shrinkage)"),
    ]
    max_ep = 1
    for ax, (key, title) in zip(axes, panels):
        for label, s, color, ls in loaded:
            hist = s["training_history"]
            xs = [h["epoch"] for h in hist if key in h]
            ys = [h[key] for h in hist if key in h]
            if not xs:
                continue
            max_ep = max(max_ep, max(xs))
            ax.plot(xs, ys, marker="o", ms=4.5, lw=2.0, color=color, linestyle=ls, label=label)
            be = s.get("best_epoch")
            if be in xs:
                bx = be
                by = ys[xs.index(be)]
                ax.scatter([bx], [by], s=110, facecolors="none", edgecolors="black", linewidths=1.6, zorder=6)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch (Max 10, Early Stopping Patience = 3)", fontsize=9.5)
        ax.set_xticks(range(1, max_ep + 1))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Training & Convergence Dynamics Across 10 Epochs (Circles Mark Early-Stopping Selected Checkpoints)",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.98))
    _save_to_all(fig, "v4_epoch_convergence.png")


def plot_progressive_improvement(ladder_stats: List[Dict[str, Any]]) -> None:
    if not ladder_stats:
        return
    steps = [r["short"] for r in ladder_stats]
    x = np.arange(len(steps))
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.5))
    short_xlabels = {
        "S0": "S0\nStock Head\n+ BCE",
        "S1": "S1\n+ Custom Head\n& Diff LR",
        "S2": "S2\n+ TweetText\nCleaner",
        "S3": "S3\n+ Hybrid Pool\n& MSD",
        "S4": "S4\n+ Damped\nW-BCE",
        "S5": "S5\n+ Shrinkage\nThresh",
    }

    for ax, split, title in [
        (axes[0], "test_2020", "In-Time Evaluation (test_2020)"),
        (axes[1], "test_2021", "Out-of-Time Evaluation (test_2021)"),
    ]:
        mi = [r[split]["micro_f1"] for r in ladder_stats]
        ma = [r[split]["macro_f1"] for r in ladder_stats]
        ax.plot(x, mi, marker="o", lw=2.5, ms=7, color="#1f77b4", label="Micro-F1 (BERT V4)")
        ax.plot(x, ma, marker="s", lw=2.5, ms=7, color="#2ca02c", label="Macro-F1 (BERT V4)")

        for xi, (v_mi, v_ma) in enumerate(zip(mi, ma)):
            ax.annotate(f"{v_mi:.4f}", (xi, v_mi), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8, fontweight="bold", color="#1f77b4")
            ax.annotate(f"{v_ma:.4f}", (xi, v_ma), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=8, fontweight="bold", color="#2ca02c")

        # Reference lines
        sota_mi, sota_ma = (0.6780, 0.5340) if split == "test_2020" else (0.6550, 0.4510)
        tfidf_mi, tfidf_ma = (0.6890, 0.5227) if split == "test_2020" else (0.6154, 0.3997)
        ax.axhline(sota_mi, color="#1f77b4", ls="--", lw=1.1, alpha=0.6, label=f"Antypas et al. (2022) Micro ({sota_mi:.3f})")
        ax.axhline(sota_ma, color="#2ca02c", ls="--", lw=1.1, alpha=0.6, label=f"Antypas et al. (2022) Macro ({sota_ma:.3f})")
        ax.axhline(tfidf_mi, color="#7f8c8d", ls=":", lw=1.1, alpha=0.7, label=f"TF-IDF Baseline Micro ({tfidf_mi:.3f})")

        ax.set_xticks(x)
        ax.set_xticklabels([short_xlabels.get(r["short"], r["short"]) for r in ladder_stats], fontsize=8.2)
        ax.set_ylim(0.22, 0.85)
        ax.set_ylabel("F1 Score", fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7.5, loc="lower right")

    fig.suptitle(
        "Strict One-Factor-at-a-Time BERT Improvement Ladder (S0 -> S5)",
        fontsize=12.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    _save_to_all(fig, "v4_progressive_improvement.png")


def plot_error_diagnostics(ladder_stats: List[Dict[str, Any]]) -> None:
    if not ladder_stats:
        return
    steps = [r["short"] for r in ladder_stats]
    x = np.arange(len(steps))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))

    # Panel 1: Dead classes
    dead = [r["test_2021"]["dead_classes"] for r in ladder_stats]
    bars = axes[0].bar(x, dead, width=0.55, color=["#d62728" if d > 0 else "#2ca02c" for d in dead], alpha=0.85)
    for b, d in zip(bars, dead):
        axes[0].annotate(f"{d}/19", (b.get_x() + b.get_width() / 2, d), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(steps, fontsize=9, fontweight="bold")
    ymax_dead = max(max(dead) + 1.5, 3.0)
    axes[0].set_ylim(0, ymax_dead)
    axes[0].set_yticks(range(0, int(np.ceil(ymax_dead)) + 1))
    axes[0].set_ylabel("Collapsed Classes (F1 = 0)", fontsize=9.5, fontweight="bold")
    axes[0].set_title("A. Dead Tail-Class Collapse Elimination", fontsize=10.5, fontweight="bold")

    # Panel 2: TP, FP, FN counts
    w = 0.26
    tp = [r["test_2021"]["tp"] for r in ladder_stats]
    fn = [r["test_2021"]["fn"] for r in ladder_stats]
    fp = [r["test_2021"]["fp"] for r in ladder_stats]
    b1 = axes[1].bar(x - w, tp, w, label="True Positives (TP)", color="#2ca02c")
    b2 = axes[1].bar(x, fn, w, label="False Negatives (FN)", color="#d62728")
    b3 = axes[1].bar(x + w, fp, w, label="False Positives (FP)", color="#ff7f0e")
    for bars_g in (b1, b2, b3):
        for b in bars_g:
            h = b.get_height()
            axes[1].annotate(f"{int(h)}", (b.get_x() + b.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=6.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(steps, fontsize=9, fontweight="bold")
    axes[1].set_ylim(0, max(tp) * 1.24)
    axes[1].set_ylabel("Decision Count (N x 19 = 31,901)", fontsize=9.5, fontweight="bold")
    axes[1].set_title("B. TP Recovery vs. FN & FP Reduction (test_2021)", fontsize=10.5, fontweight="bold")
    axes[1].legend(fontsize=7.3, loc="upper center", ncol=3, frameon=True)

    # Panel 3: Multi-label cardinality slice F1
    c1 = [r["test_2021"]["card1_micro_f1"] for r in ladder_stats]
    c2 = [r["test_2021"]["card2_micro_f1"] for r in ladder_stats]
    c3 = [r["test_2021"]["card3p_micro_f1"] for r in ladder_stats]
    axes[2].plot(x, c1, marker="o", lw=2.2, label="1-Label Tweets", color="#1f77b4")
    axes[2].plot(x, c2, marker="s", lw=2.2, label="2-Label Tweets", color="#9467bd")
    axes[2].plot(x, c3, marker="^", lw=2.2, label="3+ Label Tweets", color="#e67e22")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(steps, fontsize=9, fontweight="bold")
    axes[2].set_ylim(0.50, 0.82)
    axes[2].set_ylabel("Micro-F1", fontsize=9.5, fontweight="bold")
    axes[2].set_title("C. Error Reduction by Label Cardinality", fontsize=10.5, fontweight="bold")
    axes[2].legend(fontsize=8, loc="lower right")

    fig.suptitle("Multi-Dimensional Error Diagnostics Across BERT Improvement Steps (test_2021)", fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_to_all(fig, "v4_error_diagnostics.png")


def plot_loss_and_threshold_ablation() -> None:
    loss_rows = []
    for label, folder in LOSS_CONFIGS:
        run_dir = OUT_ROOT / folder
        if (run_dir / "metrics_summary.json").exists():
            yt, yp, s = _get_preds_and_probs(run_dir, "test_2021", "default")
            d = compute_diagnostics(yt, yp)
            loss_rows.append((label, d["micro_f1"], d["macro_f1"]))

    s4_dir = OUT_ROOT / "v4_s4_damped_wbce"
    thresh_rows = []
    if (s4_dir / "metrics_summary.json").exists():
        for tlabel, mode in [
            ("Fixed 0.50", "default_0.5"),
            ("Global Tuned", "global_tuned"),
            ("Unregularized Per-Class", "per_class_tuned"),
            ("Shrinkage Per-Class (k=20)", "shrinkage_tuned"),
        ]:
            _, yp_t, _ = _get_preds_and_probs(s4_dir, "test_2021", mode)
            yt_t = np.asarray(_load_json(s4_dir / "test_predictions.json")["test_2021"]["y_true"], dtype=int)
            d = compute_diagnostics(yt_t, yp_t)
            thresh_rows.append((tlabel, d["micro_f1"], d["macro_f1"]))

    if not loss_rows or not thresh_rows:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    w = 0.36

    x1 = np.arange(len(loss_rows))
    b1 = axes[0].bar(x1 - w / 2, [r[1] for r in loss_rows], w, label="Micro-F1", color="#1f77b4")
    b2 = axes[0].bar(x1 + w / 2, [r[2] for r in loss_rows], w, label="Macro-F1", color="#2ca02c")
    for b in list(b1) + list(b2):
        h = b.get_height()
        axes[0].annotate(f"{h:.3f}", (b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.2)
    axes[0].set_xticks(x1)
    axes[0].set_xticklabels([r[0] for r in loss_rows], rotation=15, ha="right", fontsize=8.5)
    axes[0].set_ylim(0.45, 0.80)
    axes[0].set_ylabel("test_2021 F1", fontsize=9.5, fontweight="bold")
    axes[0].set_title("A. Controlled Loss Comparison (Same S3 Architecture)", fontsize=10.5, fontweight="bold")
    axes[0].legend(fontsize=8)

    x2 = np.arange(len(thresh_rows))
    b3 = axes[1].bar(x2 - w / 2, [r[1] for r in thresh_rows], w, label="Micro-F1", color="#1f77b4")
    b4 = axes[1].bar(x2 + w / 2, [r[2] for r in thresh_rows], w, label="Macro-F1", color="#2ca02c")
    for b in list(b3) + list(b4):
        h = b.get_height()
        axes[1].annotate(f"{h:.3f}", (b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.5)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([r[0] for r in thresh_rows], rotation=15, ha="right", fontsize=8.5)
    axes[1].set_ylim(0.45, 0.80)
    axes[1].set_ylabel("test_2021 F1", fontsize=9.5, fontweight="bold")
    axes[1].set_title("B. Post-Processing Thresholding Strategies on S4", fontsize=10.5, fontweight="bold")
    axes[1].legend(fontsize=8)

    fig.suptitle("Controlled Ablation of Imbalance Losses & Shrinkage Thresholding (test_2021)", fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_to_all(fig, "v4_loss_and_threshold_ablation.png")


def sync_supporting_figures() -> None:
    eda_src = Path(r"D:\VSF\Sentiment_Topic_Classification\eda\figures")
    eda_dst = ROOT / "eda" / "figures"
    if eda_src.exists():
        eda_dst.mkdir(parents=True, exist_ok=True)
        for p in eda_src.glob("*.png"):
            shutil.copyfile(p, eda_dst / p.name)

    arch_src = ROOT / "doc" / "figures"
    arch_dst = Path(r"D:\VSF\Sentiment_Topic_Classification\doc\figures")
    arch_dst.mkdir(parents=True, exist_ok=True)
    for name in ("arch_encoder_stock_vs_custom.png", "arch_pooling_msd.png", "temporal_drift_resilience.png"):
        if (arch_src / name).exists():
            shutil.copyfile(arch_src / name, arch_dst / name)


def main() -> None:
    _style()
    ladder_stats: List[Dict[str, Any]] = []
    for short, label, folder, mode in LADDER_CONFIGS:
        run_dir = OUT_ROOT / folder
        if not (run_dir / "metrics_summary.json").exists():
            continue
        yt20, yp20, s20 = _get_preds_and_probs(run_dir, "test_2020", mode)
        yt21, yp21, _ = _get_preds_and_probs(run_dir, "test_2021", mode)
        d20 = compute_diagnostics(yt20, yp20)
        d21 = compute_diagnostics(yt21, yp21)
        ladder_stats.append({
            "short": short,
            "label": label,
            "folder": folder,
            "mode": mode,
            "best_epoch": s20.get("best_epoch"),
            "epochs_ran": len(s20.get("training_history", [])),
            "test_2020": d20,
            "test_2021": d21,
        })

    # Loss diagnostics on test_2021
    loss_stats: List[Dict[str, Any]] = []
    for label, folder in LOSS_CONFIGS:
        run_dir = OUT_ROOT / folder
        if not (run_dir / "metrics_summary.json").exists():
            continue
        yt21, yp21, s = _get_preds_and_probs(run_dir, "test_2021", "default")
        d21 = compute_diagnostics(yt21, yp21)
        loss_stats.append({
            "label": label,
            "folder": folder,
            "best_epoch": s.get("best_epoch"),
            "test_2021": d21,
        })

    # Multi-seed stats
    seed_runs = {}
    for seed_tag, folder in [("42", "v4_s4_damped_wbce"), ("1337", "v4_s4_damped_wbce_s1337"), ("2024", "v4_s4_damped_wbce_s2024")]:
        run_dir = OUT_ROOT / folder
        if (run_dir / "metrics_summary.json").exists():
            yt21_g, yp21_g, s = _get_preds_and_probs(run_dir, "test_2021", "default")
            yt21_s, yp21_s, _ = _get_preds_and_probs(run_dir, "test_2021", "shrinkage_tuned")
            seed_runs[seed_tag] = {
                "best_epoch": s.get("best_epoch"),
                "global": compute_diagnostics(yt21_g, yp21_g),
                "shrinkage": compute_diagnostics(yt21_s, yp21_s),
            }

    out_summary = {
        "ladder": ladder_stats,
        "loss_ablation": loss_stats,
        "multi_seed": seed_runs,
    }
    (OUT_ROOT / "v4_error_analysis_summary.json").write_text(json.dumps(out_summary, indent=2), encoding="utf-8")

    plot_epoch_convergence()
    plot_progressive_improvement(ladder_stats)
    plot_error_diagnostics(ladder_stats)
    plot_loss_and_threshold_ablation()
    sync_supporting_figures()

    print("\n=== V4 LADDER SUMMARY (test_2020 -> test_2021) ===")
    for r in ladder_stats:
        d20, d21 = r["test_2020"], r["test_2021"]
        print(
            f"{r['short']:3s} | ep {r['best_epoch']}/{r['epochs_ran']} | "
            f"2020: {d20['micro_f1']:.4f}/{d20['macro_f1']:.4f} | "
            f"2021: {d21['micro_f1']:.4f}/{d21['macro_f1']:.4f} | "
            f"TP={d21['tp']} FP={d21['fp']} FN={d21['fn']} Dead={d21['dead_classes']}/19 | "
            f"Card(1/2/3+)={d21['card1_micro_f1']:.3f}/{d21['card2_micro_f1']:.3f}/{d21['card3p_micro_f1']:.3f}"
        )


if __name__ == "__main__":
    main()
