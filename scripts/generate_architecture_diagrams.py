"""Draws the model-architecture and data-representation pipeline diagrams used in doc/MASTER_TWEET_TOPIC_SYSTEM_REPORT.md.

The diagrams are drawn with matplotlib patches (research-paper block-diagram style): rounded blocks coloured by role
(data / frozen / trainable / loss-output / post-processing), arrows with tensor shapes, and a legend. Every number printed
in a block comes from a config or a measured log (see the report's Phụ lục A); nothing here is a result.

    python scripts/generate_architecture_diagrams.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "doc" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["savefig.dpi"] = 170
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.facecolor"] = "white"

# role -> (fill, edge)
STYLE = {
    "data": ("#F1F3F5", "#868E96"),
    "frozen": ("#D0EBFF", "#1C7ED6"),
    "train": ("#FFE8CC", "#E8590C"),
    "out": ("#D3F9D8", "#2F9E44"),
    "post": ("#E5DBFF", "#7048E8"),
}
LEGEND = [
    ("data", "Data / tensor"),
    ("frozen", "Frozen module"),
    ("train", "Trainable module"),
    ("out", "Output / loss"),
    ("post", "Post-processing (no training)"),
]
INK = "#212529"
MUTED = "#495057"


def new_canvas(width: float, height: float, scale: float = 0.1):
    fig, ax = plt.subplots(figsize=(width * scale, height * scale))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def box(ax, cx: float, cy: float, w: float, h: float, lines: Sequence[str], kind: str = "data",
        fs: float = 8.0, dashed: bool = False, bold_first: bool = True, zorder: int = 3) -> Tuple[float, float, float, float]:
    fill, edge = STYLE[kind]
    patch = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                           linewidth=1.4, edgecolor=edge, facecolor=fill, linestyle="--" if dashed else "-", zorder=zorder)
    ax.add_patch(patch)
    n = len(lines)
    step = fs * 0.205  # data-units per text line (1 unit = 0.1 inch; 1pt = 0.0139 inch)
    top = cy + (n - 1) * step / 2
    for i, line in enumerate(lines):
        ax.text(cx, top - i * step, line, ha="center", va="center", fontsize=fs if i or not bold_first else fs + 0.4,
                fontweight="bold" if (i == 0 and bold_first) else "normal", color=INK, zorder=zorder + 1)
    return cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2  # left, right, bottom, top


def arrow(ax, p1: Tuple[float, float], p2: Tuple[float, float], label: Optional[str] = None, color: str = MUTED,
          label_xy: Optional[Tuple[float, float]] = None, fs: float = 6.8, style: str = "-|>", lw: float = 1.3,
          rad: float = 0.0, ls: str = "-") -> None:
    ax.annotate("", xy=p2, xytext=p1, zorder=2,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=0, shrinkB=0, linestyle=ls,
                                connectionstyle=f"arc3,rad={rad}"))
    if label:
        lx, ly = label_xy if label_xy else ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + 1.6)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=fs, color=MUTED, style="italic", zorder=5,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))


def group(ax, x0: float, y0: float, x1: float, y1: float, title: str, fs: float = 8.0) -> None:
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.5", linewidth=1.0,
                                edgecolor="#ADB5BD", facecolor="none", linestyle=(0, (4, 3)), zorder=1))
    ax.text(x0 + 1.2, y1 - 1.4, title, ha="left", va="top", fontsize=fs, color=MUTED, fontweight="bold", zorder=5)


def legend(ax, kinds: Iterable[str], y: float, x: float = 0, ncol: int = 5) -> None:
    handles = [Patch(facecolor=STYLE[k][0], edgecolor=STYLE[k][1], label=dict(LEGEND)[k]) for k in kinds]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(x / ax.get_xlim()[1], y / ax.get_ylim()[1]), ncol=ncol,
              frameon=False, fontsize=7.5, handlelength=1.6, columnspacing=1.6)


def title(ax, text: str, width: float, height: float, fs: float = 11) -> None:
    ax.text(width / 2, height - 0.5, text, ha="center", va="top", fontsize=fs, fontweight="bold", color=INK)


def save(fig, name: str) -> None:
    dest = OUT_DIR / name
    fig.savefig(dest)
    plt.close(fig)
    print("wrote", dest)


# --------------------------------------------------------------------------------------------------------------------
def fig_overview() -> None:
    W, H = 150, 62
    fig, ax = new_canvas(W, H)
    title(ax, "End-to-end pipeline: from a tweet to 19 topic decisions", W, H)

    box(ax, 9, 31, 14, 10, ["Tweet text", "x"], "data")
    box(ax, 27, 31, 17, 14, ["Preprocessing", "TweetTextCleaner", "(optional)"], "data", dashed=True, fs=7.4)
    arrow(ax, (16, 31), (18.5, 31))

    ys = [48, 38, 28, 18]
    names = [
        ["TF-IDF + Logistic Regression", "(One-vs-Rest, 19 binary LR)"],
        ["Transformer encoder", "Twitter-RoBERTa / BERT-base"],
        ["Liquid d1-2B", "probe / option letters / system_one"],
        ["Qwen3.5-4B", "zero-shot JSON / QLoRA classifier"],
    ]
    group(ax, 38, 10.5, 68, 55, "Model families (Section VI)")
    for y, ln in zip(ys, names):
        box(ax, 53, y - 1.5, 27, 7.4, ln, "frozen", fs=7.2)
        arrow(ax, (35.5, 31), (39.5, y - 1.5), lw=1.0)
        arrow(ax, (66.5, y - 1.5), (77.5, 31), lw=1.0)

    box(ax, 85, 31, 14, 13, ["19 probabilities", "p ∈ [0,1]^19", "(sigmoid per topic)"], "out", fs=7.4)

    # decision branch
    box(ax, 104, 44, 16, 9, ["Temperature", "scaling  p = σ(z/T)", "(optional)"], "post", fs=7.2, dashed=True)
    box(ax, 124, 44, 16, 9, ["Threshold τ", "0.5 / global /", "per-class"], "post", fs=7.2)
    box(ax, 142, 44, 14, 9, ["Topics ŷ", "ŷ ∈ {0,1}^19"], "out", fs=7.4)
    arrow(ax, (92, 34), (96, 43.5))
    arrow(ax, (112, 44), (116, 44))
    arrow(ax, (132, 44), (135, 44))

    # selective branch
    box(ax, 104, 18, 16, 9, ["Uncertainty score", "margin or entropy", "of the 19 decisions"], "post", fs=7.2)
    box(ax, 124, 18, 16, 9, ["Selective", "classification", "keep top-k% by score"], "post", fs=7.2)
    box(ax, 142, 18, 14, 11, ["Auto-labelled", "(confident)", "↓", "Human review", "(uncertain)"], "out", fs=7.0, bold_first=False)
    arrow(ax, (92, 28), (96, 19))
    arrow(ax, (112, 18), (116, 18))
    arrow(ax, (132, 18), (135, 18))

    ax.text(124, 55, "Decision rule", ha="center", fontsize=8, color=MUTED, fontweight="bold")
    ax.text(124, 28.5, "Escalation rule", ha="center", fontsize=8, color=MUTED, fontweight="bold")
    ax.text(14, 6.4, "Thresholds, temperature T and coverage k are fitted on validation_2020 only; models are trained on train_2020.",
            fontsize=7.4, color=MUTED, ha="left")
    legend(ax, ["data", "frozen", "out", "post"], y=0, x=14, ncol=4)
    save(fig, "arch_overview.png")


# --------------------------------------------------------------------------------------------------------------------
def fig_repr() -> None:
    W, H = 150, 70
    fig, ax = new_canvas(W, H)
    title(ax, "Data representation: three views of the same tweet, one label representation", W, H)

    box(ax, 10, 38, 15, 12, ["Tweet", "(raw or cleaned)"], "data")
    rows = [(56, "(a) Sparse lexical features  (B1, TF-IDF audit)"),
            (38, "(b) Contextual subword embeddings  (RoBERTa, BERT)"),
            (20, "(c) LLM hidden states  (d1-2B, Qwen3.5-4B)")]
    first_left = {56: 26.0, 38: 28.0, 20: 28.0}  # left edge of the first block in each row
    for y, t in rows:
        ax.text(23, y + 8.2, t, fontsize=8.4, fontweight="bold", color=MUTED, ha="left")
        arrow(ax, (17.5, 38), (first_left[y] - 0.4, y), lw=1.0)

    # (a)  the two vectorisers run in parallel on the same tweet and are concatenated
    box(ax, 39, 56, 26, 9, ["Word TF-IDF", "n-gram (1,2) · 40,000 feat."], "data", fs=7.2)
    ax.text(55.5, 56, "+", fontsize=14, fontweight="bold", color=MUTED, ha="center", va="center")
    box(ax, 72, 56, 26, 9, ["Char TF-IDF (char_wb)", "n-gram (3,5) · 60,000 feat."], "data", fs=7.2)
    box(ax, 108, 56, 30, 9, ["hstack", "x ∈ R^100,000  (sparse, 1 row / tweet)"], "out", fs=7.2)
    arrow(ax, (85.5, 56), (92.5, 56))

    # (b)
    box(ax, 39, 38, 22, 9, ["Tokenizer", "WordPiece / BPE · L ≤ 128"], "data", fs=7.2)
    box(ax, 67, 38, 24, 9, ["input_ids, attention_mask", "(B × L)"], "data", fs=7.2)
    box(ax, 96, 38, 24, 9, ["Encoder (12 layers)", "H ∈ R^{B×L×768}"], "train", fs=7.2)
    box(ax, 128, 38, 26, 9, ["Pooling", "h ∈ R^{B×768}  (CLS / mean / attn)"], "out", fs=7.2)
    arrow(ax, (50, 38), (55, 38))
    arrow(ax, (79, 38), (84, 38))
    arrow(ax, (108, 38), (115, 38))

    # (c)
    box(ax, 39, 20, 22, 9, ["Tokenizer + prompt", "(or question set)"], "data", fs=7.2)
    box(ax, 72, 20, 28, 9, ["LLM backbone (frozen / 4-bit)", "H ∈ R^{B×L×d}"], "frozen", fs=7.2)
    box(ax, 108, 20, 30, 9, ["Masked mean pooling", "h ∈ R^{B×d}  (cached for d1 probe)"], "out", fs=7.2)
    arrow(ax, (50, 20), (58, 20))
    arrow(ax, (86, 20), (93, 20))

    box(ax, 75, 5.5, 62, 6.5, ["Label representation", "multi-hot  y ∈ {0,1}^19  (51.8% of tweets have ≥ 2 labels)"], "out", fs=7.2)
    ax.text(5, 5.5, "Targets", fontsize=8.4, fontweight="bold", color=MUTED, va="center")
    save(fig, "arch_data_representation.png")


# --------------------------------------------------------------------------------------------------------------------
def fig_ml() -> None:
    W, H = 150, 42
    fig, ax = new_canvas(W, H)
    title(ax, "Classical baseline: TF-IDF + One-vs-Rest Logistic Regression", W, H)
    box(ax, 11, 20, 16, 10, ["Tweet", "(cleaned in the", "audit variant)"], "data", fs=7.2)
    box(ax, 40, 29, 28, 9, ["Word TF-IDF", "n-gram (1,2) · 40,000 · sublinear tf"], "data", fs=7.2)
    box(ax, 40, 11, 28, 9, ["Char TF-IDF (char_wb)", "n-gram (3,5) · 60,000 · sublinear tf"], "data", fs=7.2)
    arrow(ax, (19, 22), (26, 28))
    arrow(ax, (19, 18), (26, 12))
    box(ax, 70, 20, 17, 10, ["hstack", "x ∈ R^100,000", "(sparse)"], "data", fs=7.2)
    arrow(ax, (54, 28), (61.5, 22))
    arrow(ax, (54, 12), (61.5, 18))
    box(ax, 101, 20, 30, 17, ["19 × Logistic Regression", "p_c = σ(w_cᵀx + b_c)", "C = 2.0 · class_weight = balanced", "lbfgs · max_iter 1000"], "train", fs=7.2)
    arrow(ax, (78.5, 20), (86, 20))
    box(ax, 133, 20, 22, 11, ["19 probabilities", "p ∈ [0,1]^19", "global τ (val, Micro-F1)"], "out", fs=7.2)
    arrow(ax, (116, 20), (122, 20))
    ax.text(101, 5.5, "Classifier Chains (P(y_c | x, y_<c)) is implemented but has no measured result in this report.",
            fontsize=7.2, color=MUTED, ha="center")
    legend(ax, ["data", "train", "out"], y=0, x=4, ncol=3)
    save(fig, "arch_ml_tfidf.png")


# --------------------------------------------------------------------------------------------------------------------
def _stack(ax, cx: float, ys: Sequence[float], specs, w: float = 38, h: float = 6.6, fs: float = 7.2) -> None:
    prev = None
    for y, (lines, kind, dashed) in zip(ys, specs):
        box(ax, cx, y, w, h, lines, kind, fs=fs, dashed=dashed)
        if prev is not None:
            arrow(ax, (cx, prev - h / 2), (cx, y + h / 2))
        prev = y


def fig_encoder_heads() -> None:
    W, H = 150, 82
    fig, ax = new_canvas(W, H)
    title(ax, "Transformer encoder with two classification heads", W, H)

    ys = [66, 57, 48, 39, 30, 21, 12]
    group(ax, 2, 8, 70, 74, "(a) Stock head  ·  L1–L5 and the Vòng 2 BCE / ASL baselines")
    group(ax, 80, 8, 148, 74, "(b) Custom head  ·  P0–P6, H2–H5 and Vòng 2 attn-pool + MSD")

    left = [
        (["Input tokens", "input_ids · attention_mask  (B × L, L ≤ 128)"], "data", False),
        (["Pretrained encoder  (12 layers, d = 768)", "BERT-base (v3) / Twitter-RoBERTa (v2)  ·  lr 2e-5"], "train", False),
        (["Take the first token", "[CLS] (BERT) / <s> (RoBERTa)  →  B × 768"], "data", False),
        (["Dense 768→768 + tanh", "BERT: pretrained pooler  ·  RoBERTa: new layer"], "train", False),
        (["Dropout  p = 0.1", "RoBERTa also applies it before the dense layer"], "data", False),
        (["Linear 768 → 19", "lr 2e-5 (same as the encoder)"], "train", False),
        (["Logits z (B × 19)  →  σ  →  p", "loss: BCE / Weighted BCE / Focal / ASL"], "out", False),
    ]
    right = [
        (["Input tokens", "input_ids · attention_mask  (B × L, L ≤ 128)"], "data", False),
        (["Pretrained encoder  (12 layers, d = 768)", "same backbone  ·  lr 2e-5"], "train", False),
        (["Pooling  (one of three)", "[CLS]  |  mean over tokens  |  learned attention  →  B × 768"], "train", False),
        (["Multi-sample dropout  (p = 0.2)", "M = 5 in P6 and the Vòng 2 attn-pool model; M = 1 in P0–P5"], "data", False),
        (["Shared Linear 768 → 19", "head lr 1e-3  (50× the encoder)"], "train", False),
        (["Mean of the M losses  (training)", "single pass, no dropout  (inference)"], "data", False),
        (["Logits z (B × 19)  →  σ  →  p", "loss: BCE / Weighted BCE / Focal"], "out", False),
    ]
    _stack(ax, 36, ys, left, w=60, h=7.0)
    _stack(ax, 114, ys, right, w=60, h=7.0)

    ax.text(75, 3.2, "The two heads differ in four ways at once: head learning rate (2e-5 vs 1e-3), pretrained pooler (yes vs none), "
                     "dropout (0.1 vs 0.2) and initialisation.\nWhich of the four causes the 0.26 → 0.57 Macro-F1 gap is not isolated.",
            fontsize=7.6, color=MUTED, ha="center", va="center")
    legend(ax, ["data", "train", "out"], y=-6.5, x=40, ncol=3)
    save(fig, "arch_encoder_stock_vs_custom.png")


def fig_pooling_msd() -> None:
    W, H = 150, 56
    fig, ax = new_canvas(W, H)
    title(ax, "Learned attention pooling and multi-sample dropout (MSD)", W, H)
    group(ax, 2, 4, 72, 51, "(a) Learned attention pooling")
    group(ax, 78, 4, 148, 51, "(b) Multi-sample dropout head")

    # (a)  top row: H -> W_a -> v_a ; right column: v_a -> softmax ; bottom row (right to left): softmax -> weighted sum
    box(ax, 15, 36, 16, 9, ["H", "L × 768"], "data", fs=7.2)
    box(ax, 38, 36, 21, 9, ["W_a: 768→256", "u_i = tanh(W_a h_i + b)"], "train", fs=6.8)
    box(ax, 60, 36, 17, 9, ["v_a ∈ R^256", "s_i = u_iᵀ v_a"], "train", fs=6.8)
    box(ax, 60, 14, 17, 9, ["softmax over i", "α_i  (padding masked)"], "data", fs=6.8)
    box(ax, 15, 14, 18, 9, ["Σ α_i h_i", "h_pool ∈ R^768"], "out", fs=6.8)
    arrow(ax, (23, 36), (27.5, 36))
    arrow(ax, (48.5, 36), (51.5, 36))
    arrow(ax, (60, 31.5), (60, 18.5), label="scores s_i", label_xy=(65.5, 25))
    arrow(ax, (51.5, 14), (24, 14), label="weights α_i", label_xy=(38, 16.5))
    arrow(ax, (15, 31.5), (15, 18.5), color="#868E96", ls="--", label="values h_i", label_xy=(15, 25))

    # (b)
    box(ax, 88, 28, 14, 9, ["h_pool", "B × 768"], "data", fs=7.2)
    for y, t in [(41, "mask 1  (p)"), (28, "mask 2  (p)"), (15, "mask M  (p)")]:
        box(ax, 108, y, 14, 7, ["Dropout", t], "data", fs=6.8)
        arrow(ax, (95, 28), (101, y), lw=1.0)
        arrow(ax, (115, y), (121, 28 + (y - 28) * 0.45), lw=1.0)
    box(ax, 127, 28, 12, 22, ["Shared", "Linear", "768→19", "", "lr 1e-3"], "train", fs=6.8, bold_first=False)
    box(ax, 142, 28, 10, 14, ["mean_m", "ℓ(z_m, y)"], "out", fs=6.8, bold_first=False)
    arrow(ax, (133, 28), (137, 28))
    ax.text(110, 6.2, "Only the dropout masks differ between the M branches; weights are shared.", fontsize=6.8, color=MUTED, ha="center")
    legend(ax, ["data", "train", "out"], y=-2.5, x=50, ncol=3)
    save(fig, "arch_pooling_msd.png")


# --------------------------------------------------------------------------------------------------------------------
def fig_d1() -> None:
    W, H = 150, 92
    fig, ax = new_canvas(W, H)
    title(ax, "Liquid d1-2B: three ways to use the same decision model", W, H)

    group(ax, 2, 66, 148, 86, "(a) Option-letter zero-shot  ·  Vòng 2 (reference scorer)")
    box(ax, 15, 76, 22, 9, ["Tweet + prompt", "19 topics listed A–S"], "data", fs=7.0)
    box(ax, 43, 76, 20, 9, ["d1-2B", "frozen · 1 pass"], "frozen", fs=7.0)
    box(ax, 72, 76, 26, 9, ["Next-token logits", "of the 19 letter tokens"], "data", fs=7.0)
    box(ax, 101, 76, 20, 9, ["Softmax over", "19 letters"], "data", fs=7.0)
    box(ax, 129, 76, 26, 9, ["p ∈ [0,1]^19", "global τ = 0.7 (val)"], "out", fs=7.0)
    for a, b in [(26, 33), (53, 59), (85, 91), (111, 116)]:
        arrow(ax, (a, 76), (b, 76))
    ax.text(75, 68.4, "Softmax forces the options to compete, so this is only a weak multi-label scorer.", fontsize=6.8,
            color=MUTED, ha="center")

    group(ax, 2, 36, 148, 62, "(b) Linear probe  ·  Vòng 2 (supervised head on frozen features)")
    xs = [11, 31, 51, 71, 95, 120, 141]
    box(ax, xs[0], 47, 14, 9, ["Tweet"], "data", fs=7.0)
    box(ax, xs[1], 47, 18, 9, ["d1-2B", "frozen"], "frozen", fs=7.0)
    box(ax, xs[2], 47, 18, 9, ["Masked mean", "pool of last H"], "data", fs=7.0)
    box(ax, xs[3], 47, 18, 9, ["Feature cache", "(computed once)"], "data", fs=7.0, dashed=True)
    box(ax, xs[4], 47, 26, 9, ["MSD, 4 branches", "p ∈ {0.1, 0.2, 0.3, 0.4}"], "train", fs=7.0)
    box(ax, xs[5], 47, 20, 9, ["Linear → 19", "lr 1e-3 · 80 ep max"], "train", fs=7.0)
    box(ax, xs[6], 47, 12, 9, ["p", "per-class τ"], "out", fs=7.0)
    for a, b in [(18, 22), (40, 42), (60, 62), (80, 82), (108, 110), (130, 135)]:
        arrow(ax, (a, 47), (b, 47))
    ax.text(75, 39.2, "Loss: BCE with pos_weight clamped to 30 · AdamW · weight decay 0.01 · early stopping (patience 10).",
            fontsize=6.8, color=MUTED, ha="center")

    group(ax, 2, 4, 148, 32, "(c) Native system_one API  ·  Vòng 3 (no training)")
    box(ax, 14, 17, 22, 12, ["Tweet +", "question(s)", "+ answer set"], "data", fs=7.0)
    box(ax, 43, 17, 22, 12, ["d1 system_one", "1 forward pass", "(fp16 on T4/P100)"], "frozen", fs=7.0)
    box(ax, 80, 23, 30, 8, ["choice_multiple", "1 question · 19 options · pick several"], "data", fs=6.8)
    box(ax, 80, 11, 30, 8, ["noul", "19 independent yes / no questions"], "data", fs=6.8)
    box(ax, 112, 17, 20, 11, ["19 probabilities", "p ∈ [0,1]^19"], "out", fs=7.0)
    box(ax, 136, 17, 20, 11, ["global τ (val)", "0.15 choice", "0.55 noul"], "post", fs=7.0)
    arrow(ax, (25, 17), (32, 17))
    arrow(ax, (54, 19), (64.5, 23), lw=1.1)
    arrow(ax, (54, 15), (64.5, 11), lw=1.1)
    arrow(ax, (95, 23), (102, 19), lw=1.1)
    arrow(ax, (95, 11), (102, 15), lw=1.1)
    arrow(ax, (122, 17), (126, 17))
    legend(ax, ["data", "frozen", "train", "out", "post"], y=-3.5, x=22, ncol=5)
    save(fig, "arch_d1.png")


# --------------------------------------------------------------------------------------------------------------------
def fig_qwen() -> None:
    W, H = 150, 66
    fig, ax = new_canvas(W, H)
    title(ax, "Qwen3.5-4B: zero-shot JSON baseline vs supervised QLoRA classifier", W, H)

    group(ax, 2, 44, 148, 60, "(a) Zero-shot JSON generation  ·  B4, Vòng 1")
    box(ax, 14, 51.5, 22, 8, ["Tweet + prompt", "“return a JSON list”"], "data", fs=7.0)
    box(ax, 42, 51.5, 22, 8, ["Qwen3.5-4B", "4-bit NF4 · frozen"], "frozen", fs=7.0)
    box(ax, 72, 51.5, 26, 8, ["Greedy generation", "≤ 48 new tokens"], "data", fs=7.0)
    box(ax, 101, 51.5, 20, 8, ["JSON array", "of topic names"], "data", fs=7.0)
    box(ax, 126, 51.5, 16, 8, ["Regex parse", "→ ŷ"], "post", fs=7.0)
    for a, b in [(25, 31), (53, 59), (85, 91), (111, 118)]:
        arrow(ax, (a, 51.5), (b, 51.5))

    group(ax, 2, 4, 148, 41, "(b) Supervised QLoRA classifier  ·  Vòng 2")
    box(ax, 12, 24, 14, 8, ["Tweet"], "data", fs=7.0)
    box(ax, 30, 24, 14, 8, ["Tokenizer", "L ≤ 128"], "data", fs=7.0)
    box(ax, 62, 25, 36, 20, [], "frozen", fs=7.0)
    ax.text(62, 31.2, "Qwen3.5-4B  ·  4-bit NF4 (double quant.)", ha="center", va="center", fontsize=7.4, fontweight="bold", color=INK, zorder=5)
    ax.text(62, 28.4, "backbone frozen", ha="center", va="center", fontsize=7.0, color=INK, zorder=5)
    box(ax, 62, 21.2, 32, 8.4, ["LoRA adapters  r = 16 · α = 32 · dropout 0.05", "on q, k, v, o, gate, up, down"], "train", fs=6.6, bold_first=False, zorder=4)
    box(ax, 96, 24, 16, 8, ["Masked mean", "pool"], "data", fs=7.0)
    box(ax, 116, 24, 18, 8, ["MSD · 4 branches", "Linear → 19"], "train", fs=7.0)
    box(ax, 138, 24, 14, 8, ["p ∈ [0,1]^19", "per-class τ"], "out", fs=7.0)
    for a, b in [(19, 23), (37, 43.4), (80.4, 88), (104, 107), (125, 131)]:
        arrow(ax, (a, 24), (b, 24))
    ax.text(75, 9.2, "Trainable: 21,282,323 of 2,611,376,147 parameters (0.815%)  ·  Weighted BCE (pos_weight ≤ 30)  ·  AdamW lr 2e-4  ·  "
                     "3 epochs  ·  batch 16 × grad. accumulation 2",
            fontsize=6.9, color=MUTED, ha="center")
    legend(ax, ["data", "frozen", "train", "out", "post"], y=-4.0, x=22, ncol=5)
    save(fig, "arch_qwen_qlora.png")


def main() -> None:
    fig_overview()
    fig_repr()
    fig_ml()
    fig_encoder_heads()
    fig_pooling_msd()
    fig_d1()
    fig_qwen()


if __name__ == "__main__":
    main()
