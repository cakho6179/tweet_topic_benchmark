"""Builds the Vòng 3 markdown tables from saved JSON artifacts.

The report keeps ``—`` placeholders until a run exists. This script turns the artifacts into the exact table
bodies used in ``doc/MASTER_TWEET_TOPIC_SYSTEM_REPORT.md`` so the numbers are copied from JSON, not retyped.

Usage:
    python scripts/build_v3_report_tables.py
    python scripts/build_v3_report_tables.py --output outputs/v3_results/report_tables.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_benchmark_figures import load_v3_runs, v3_ladder_rows  # noqa: E402
from scripts.postprocess_analysis import DEFAULT_COVERAGES, analyze_run  # noqa: E402

RARE = ["fashion_&_style", "food_&_dining", "learning_&_educational", "arts_&_culture", "other_hobbies"]
LOSS_NAMES = {
    "v3_l1_bce": ("L1", "BCE"),
    "v3_l2_weighted_bce": ("L2", "Weighted BCE"),
    "v3_l3_focal_g2": ("L3", "Focal (γ=2, α=0.25)"),
    "v3_l4_focal_g1": ("L4", "Focal (γ=1, α=0.25)"),
    "v3_l5_focal_g2_noalpha": ("L5", "Focal (γ=2, α=None)"),
}


def _post(run: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cached = run["dir"] / "postprocess.json"
    if cached.exists():
        try:
            return json.loads(cached.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
    try:
        return analyze_run(run["dir"], DEFAULT_COVERAGES)
    except Exception:  # noqa: BLE001
        return None


def _num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(value):
        return "—"
    return f"{value:.{digits}f}"


def _delta(value: Optional[float]) -> str:
    return "—" if value is None or math.isnan(value) else f"{value:+.4f}"


def _metric(summary: Dict[str, Any], split: str, key: str) -> Optional[float]:
    return summary.get("test_metrics", {}).get(split, {}).get(key)


def _rare_f1(summary: Dict[str, Any], split: str) -> Dict[str, Optional[float]]:
    per_class = summary.get("test_metrics", {}).get(split, {}).get("per_class_f1", {})
    return {name: per_class.get(name) for name in RARE}


def _ece(post: Optional[Dict[str, Any]], split: str, kind: str) -> Optional[float]:
    if not post:
        return None
    return post.get("calibration", {}).get(split, {}).get(kind, {}).get("micro_ece")


def loss_table(runs: List[Dict[str, Any]]) -> str:
    by_name = {r["name"]: r for r in runs}
    lines = [
        "| Mã | Loss | `test_2020` Micro | `test_2020` Macro | `test_2021` Micro | `test_2021` Macro | Fashion F1 (2021) | Food F1 (2021) | Micro-ECE (2021) |",
        "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for name, (code, label) in LOSS_NAMES.items():
        run = by_name.get(name)
        if run is None:
            lines.append(f"| {code} | {label} | — | — | — | — | — | — | — |")
            continue
        summary = run["summary"]
        rare = _rare_f1(summary, "test_2021")
        post = _post(run)
        lines.append(
            "| {code} | {label} | {m20} | {g20} | {m21} | {g21} | {fashion} | {food} | {ece} |".format(
                code=code,
                label=label,
                m20=_num(_metric(summary, "test_2020", "micro_f1")),
                g20=_num(_metric(summary, "test_2020", "macro_f1")),
                m21=_num(_metric(summary, "test_2021", "micro_f1")),
                g21=_num(_metric(summary, "test_2021", "macro_f1")),
                fashion=_num(rare["fashion_&_style"]),
                food=_num(rare["food_&_dining"]),
                ece=_num(_ece(post, "test_2021", "raw")),
            )
        )
    return "\n".join(lines)


def ladder_tables(runs: List[Dict[str, Any]]) -> tuple[str, str]:
    rows = v3_ladder_rows(runs)
    by_name = {r["name"]: r for r in runs}
    p6 = by_name.get("v3_p6_cleaner_weighted_attn_msd")
    if p6 is not None:
        post = _post(p6)
        if post:
            p8_2020 = _coverage_at(post, "test_2020", 0.8)
            p8_2021 = _coverage_at(post, "test_2021", 0.8)
            rows.append({
                "step": "P8",
                "label": "P8 + selective (80% coverage)",
                "test_2020": p8_2020,
                "test_2021": p8_2021,
                "coverage": 0.8,
            })

    base = next((r for r in rows if r["step"] == "P0"), None)
    base_micro = base["test_2021"].get("micro_f1") if base else None
    base_macro = base["test_2021"].get("macro_f1") if base else None

    result = [
        "| Bước | Cấu hình | `test_2020` Micro / Macro | `test_2021` Micro / Macro | Δ Micro (2021) so với P0 | Δ Macro (2021) so với P0 |",
        "|:---:|:---|:---:|:---:|:---:|:---:|",
    ]
    for row in rows:
        micro = row["test_2021"].get("micro_f1")
        macro = row["test_2021"].get("macro_f1")
        dm = None if row.get("coverage") or micro is None or base_micro is None else micro - base_micro
        dg = None if row.get("coverage") or macro is None or base_macro is None else macro - base_macro
        suffix = "*" if row.get("coverage") else ""
        result.append(
            "| {step} | {label} | {m20} / {g20} | {m21} / {g21} | {dm} | {dg} |".format(
                step=row["step"],
                label=row["label"],
                m20=_num(row["test_2020"].get("micro_f1")),
                g20=_num(row["test_2020"].get("macro_f1")),
                m21=_num(micro),
                g21=_num(macro),
                dm=_delta(dm) + suffix,
                dg=_delta(dg) + suffix,
            )
        )

    incremental = [
        "| Bước | Kỹ thuật | Δ Micro-F1 (2021) | Δ Macro-F1 (2021) |",
        "|:---:|:---|:---:|:---:|",
    ]
    previous: Optional[Dict[str, Any]] = None
    for row in rows:
        if row["step"] == "P0":
            previous = row
            continue
        if row.get("coverage"):
            incremental.append(f"| {row['step']} | {row['label']} (đổi coverage) | — | — |")
            continue
        micro = row["test_2021"].get("micro_f1")
        macro = row["test_2021"].get("macro_f1")
        prev_micro = previous["test_2021"].get("micro_f1") if previous else None
        prev_macro = previous["test_2021"].get("macro_f1") if previous else None
        dm = micro - prev_micro if micro is not None and prev_micro is not None else None
        dg = macro - prev_macro if macro is not None and prev_macro is not None else None
        incremental.append(f"| {row['step']} | {row['label'].split(' ', 1)[1]} | {_delta(dm)} | {_delta(dg)} |")
        previous = row

    result.append("")
    result.append("* P8 được đo trên 80% tweet tự tin nhất, không phải toàn bộ test split.")
    return "\n".join(result), "\n".join(incremental)


def _coverage_at(post: Dict[str, Any], split: str, coverage: float) -> Dict[str, Any]:
    rows = post.get("coverage", {}).get(split, {}).get("calibrated", {}).get("margin", [])
    row = next((r for r in rows if abs(r["coverage"] - coverage) < 1e-9), None)
    if row is None:
        return {}
    return {"micro_f1": row["auto_micro_f1"], "macro_f1": row.get("auto_macro_f1")}


def threshold_table(runs: List[Dict[str, Any]]) -> str:
    by_name = {r["name"]: r for r in runs}
    lines = [
        "| Run | Split | 0.5 Micro / Macro | Global Micro / Macro | Per-class Micro / Macro |",
        "|:---|:---|:---:|:---:|:---:|",
    ]
    for name, label in [("v3_l1_bce", "L1 (BCE)"), ("v3_p6_cleaner_weighted_attn_msd", "P6 (full)")]:
        run = by_name.get(name)
        if run is None:
            lines.append(f"| {label} | test_2020 | — / — | — / — | — / — |")
            lines.append(f"| {label} | test_2021 | — / — | — / — | — / — |")
            continue
        comparison = run["summary"].get("threshold_comparison", {})
        for split in ("test_2020", "test_2021"):
            block = comparison.get(split, {})
            cells = []
            for mode in ("default_0.5", "global_tuned", "per_class_tuned"):
                metrics = block.get(mode, {})
                cells.append(f"{_num(metrics.get('micro_f1'))} / {_num(metrics.get('macro_f1'))}")
            lines.append(f"| {label} | {split} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ece_table(runs: List[Dict[str, Any]]) -> str:
    by_name = {r["name"]: r for r in runs}
    lines = [
        "| Run | Split | Micro-ECE raw | Micro-ECE calibrated | Macro-ECE raw | Macro-ECE calibrated |",
        "|:---|:---|:---:|:---:|:---:|:---:|",
    ]
    for name, label in [("v3_l1_bce", "L1"), ("v3_p6_cleaner_weighted_attn_msd", "P6")]:
        run = by_name.get(name)
        if run is None:
            lines.append(f"| {label} | test_2021 | — | — | — | — |")
            continue
        post = _post(run)
        if not post:
            lines.append(f"| {label} | test_2021 | — | — | — | — |")
            continue
        cal = post.get("calibration", {}).get("test_2021", {})
        lines.append(
            "| {label} | test_2021 | {mr} | {mc} | {gr} | {gc} |".format(
                label=label,
                mr=_num(cal.get("raw", {}).get("micro_ece")),
                mc=_num(cal.get("per_class_temperature", {}).get("micro_ece")),
                gr=_num(cal.get("raw", {}).get("macro_ece")),
                gc=_num(cal.get("per_class_temperature", {}).get("macro_ece")),
            )
        )
    return "\n".join(lines)


def d1_table(runs: List[Dict[str, Any]]) -> str:
    by_name = {r["name"]: r for r in runs}
    lines = [
        "| Cấu hình | `test_2020` Micro / Macro | `test_2021` Micro / Macro | Ghi chú |",
        "|:---|:---:|:---:|:---|",
    ]
    for name, label in [
        ("v3_d1_system_one", "d1-2B native `system_one` (choice, multiple)"),
        ("v3_d1_system_one_noul", "d1-2B native `system_one` (19 × noul)"),
    ]:
        run = by_name.get(name)
        if run is None:
            lines.append(f"| {label} | — / — | — / — | Zero-shot |")
            continue
        summary = run["summary"]
        lines.append(
            "| {label} | {m20} / {g20} | {m21} / {g21} | Zero-shot, không huấn luyện |".format(
                label=label,
                m20=_num(_metric(summary, "test_2020", "micro_f1")),
                g20=_num(_metric(summary, "test_2020", "macro_f1")),
                m21=_num(_metric(summary, "test_2021", "micro_f1")),
                g21=_num(_metric(summary, "test_2021", "macro_f1")),
            )
        )
    lines.append("| d1-2B option-letter zero-shot (Vòng 2) | 0.5677 / 0.4757 | 0.5763 / 0.4806 | Mốc lịch sử |")
    lines.append("| d1-2B frozen-feature linear probe (Vòng 2) | 0.6432 / 0.5519 | 0.5976 / 0.4640 | Mốc lịch sử |")
    return "\n".join(lines)


def rare_class_table(runs: List[Dict[str, Any]]) -> str:
    by_name = {r["name"]: r for r in runs}
    run = by_name.get("v3_p6_cleaner_weighted_attn_msd") or by_name.get("v3_l1_bce")
    label = "BERT Vòng 3 (P6)" if "v3_p6_cleaner_weighted_attn_msd" in by_name else "BERT Vòng 3 (L1)"
    lines = [
        "| Mô hình | `fashion_&_style` | `food_&_dining` | `learning_&_educational` | `arts_&_culture` | `other_hobbies` |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
    ]
    if run is None:
        lines.append(f"| **{label}** | — | — | — | — | — |")
        return "\n".join(lines)
    rare = _rare_f1(run["summary"], "test_2021")
    lines.append(f"| **{label}** | " + " | ".join(_num(rare[name]) for name in RARE) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Vòng 3 report tables from JSON artifacts")
    parser.add_argument("--outputs-root", default=str(ROOT / "outputs"))
    parser.add_argument("--output", default=None, help="Write the tables to this markdown file")
    args = parser.parse_args()

    runs = load_v3_runs(Path(args.outputs_root).resolve())
    ladder, incremental = ladder_tables(runs)
    text = "\n\n".join(
        [
            "## Bảng 4.3 — Loss ablation\n\n" + loss_table(runs),
            "## Bảng 5.2 — Progressive ladder\n\n" + ladder,
            "## Bảng 5.3 — Incremental gains\n\n" + incremental,
            "## Bảng 6.1 — Threshold comparison\n\n" + threshold_table(runs),
            "## Bảng 6.2 — ECE\n\n" + ece_table(runs),
            "## Bảng 7.3 — Liquid d1 native\n\n" + d1_table(runs),
            "## Bảng 8.1 — Rare-class F1 (test_2021)\n\n" + rare_class_table(runs),
        ]
    )
    print(text)
    if args.output:
        dest = Path(args.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text + "\n", encoding="utf-8")
        print(f"\nWrote {dest}")


if __name__ == "__main__":
    main()
