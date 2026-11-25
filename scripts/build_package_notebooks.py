"""Builds self-contained Kaggle notebooks that run the *package* (``scripts/run.py`` + YAML configs).

Why this exists
---------------
The older notebooks (01-06) re-implement their model inline and never import ``tweet_topic/``. Anything added to
the package (attention pooling, MSD, temperature scaling, coverage curves, the d1 probe, the QLoRA classifier, the
auditor) is therefore invisible to them. These notebooks close that gap: the project source is embedded in the
notebook as a base64 zip, unpacked on Kaggle, and executed with ``python scripts/run.py --config ...``.

Vòng 3 suites
-------------
``loss``       BERT-base loss ablation (L1-L5).
``technique``  BERT-base progressive technique ablation (P0, P2-P6; P1/P7/P8 are post-hoc).
``d1_native``  Liquid d1-2B native ``system_one`` zero-shot (choice_multiple and noul).

The d1 checkpoint requires ``transformers>=5.19`` and the vision dependencies, so only that notebook upgrades
transformers. The BERT notebooks use the Kaggle image's preinstalled version.

Nothing is executed locally. Rebuild after every code change:

    python scripts/build_package_notebooks.py

Outputs go to ``notebooks/`` as ``v3_<suite>.ipynb`` (full) and ``v3_<suite>_smoke.ipynb`` (64 train / 32 eval
samples) so a suite can be validated on Kaggle before spending the full GPU budget.
"""

from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT / "notebooks"
PREFIX = "v3"

# suite key -> (title, [configs run sequentially])
SUITES = {
    "loss": (
        "Vòng 3 — BERT-base loss ablation: BCE, Weighted BCE, Focal (gamma/alpha variants)",
        [
            "configs/v3/v3_l1_bce.yaml",
            "configs/v3/v3_l2_weighted_bce.yaml",
            "configs/v3/v3_l3_focal_g2.yaml",
            "configs/v3/v3_l4_focal_g1.yaml",
            "configs/v3/v3_l5_focal_g2_noalpha.yaml",
        ],
    ),
    "technique": (
        "Vòng 3 — BERT-base progressive technique ablation: P0 baseline, cleaner, weighted BCE, "
        "mean/attention pooling, MSD",
        [
            "configs/v3/v3_p0_custom_cls_bce.yaml",
            "configs/v3/v3_p2_cleaner_bce.yaml",
            "configs/v3/v3_p3_cleaner_weighted_bce.yaml",
            "configs/v3/v3_p4_cleaner_weighted_mean.yaml",
            "configs/v3/v3_p5_cleaner_weighted_attention.yaml",
            "configs/v3/v3_p6_cleaner_weighted_attn_msd.yaml",
        ],
    ),
    "d1_native": (
        "Vòng 3 — Liquid d1-2B native system_one zero-shot: multi-choice and 19 x noul",
        [
            "configs/v3/v3_d1_system_one.yaml",
            "configs/v3/v3_d1_system_one_noul.yaml",
        ],
    ),
}

PIP_LINES = {
    "loss": "!pip install -q accelerate datasets scikit-learn scipy pyyaml",
    "technique": "!pip install -q accelerate datasets scikit-learn scipy pyyaml",
    "d1_native": (
        # torchaudio is not needed and its CUDA build can mismatch the image's torch, which breaks the image-processor
        # import inside transformers. Upgrade only transformers; keep the image's torch/torchvision pair intact.
        "!pip uninstall -y -q torchaudio\n"
        "!pip install -q -U \"transformers>=5.19\"\n"
        "!pip install -q accelerate datasets scikit-learn scipy pyyaml torchvision pillow einops"
    ),
}


# The technique ladder split in two halves so that it can run on two GPU sessions in parallel.
SUITES["technique_a"] = (
    "Vòng 3 — technique ladder, nửa 1: P0, P2, P3",
    [
        "configs/v3/v3_p0_custom_cls_bce.yaml",
        "configs/v3/v3_p2_cleaner_bce.yaml",
        "configs/v3/v3_p3_cleaner_weighted_bce.yaml",
    ],
)
SUITES["technique_b"] = (
    "Vòng 3 — technique ladder, nửa 2: P4, P5, P6",
    [
        "configs/v3/v3_p4_cleaner_weighted_mean.yaml",
        "configs/v3/v3_p5_cleaner_weighted_attention.yaml",
        "configs/v3/v3_p6_cleaner_weighted_attn_msd.yaml",
    ],
)
PIP_LINES["technique_a"] = PIP_LINES["technique"]
PIP_LINES["technique_b"] = PIP_LINES["technique"]

# Loss ablation with the *same head as P0* (custom head, head lr 1e-3). The stock-head L-series is confounded by
# head under-training (best epoch = last epoch for every run), so this controls for the head.
SUITES["headctl"] = (
    "Vòng 3 — loss ablation với head của P0: Weighted BCE, Focal (γ=2/α=0.25, γ=1/α=0.25, γ=2/không α)",
    [
        "configs/v3_head/v3_h2_weighted_bce.yaml",
        "configs/v3_head/v3_h3_focal_g2.yaml",
        "configs/v3_head/v3_h4_focal_g1.yaml",
        "configs/v3_head/v3_h5_focal_g2_noalpha.yaml",
    ],
)
PIP_LINES["headctl"] = PIP_LINES["loss"]

# Repeat runs of the headline configs with other seeds (P0 / P6 / L2); one kernel per seed.
for _seed in (1337, 2024):
    SUITES[f"seed{_seed}"] = (
        f"Vòng 3 — lặp lại với seed {_seed}: P0, P6, L2",
        [
            f"configs/v3_seeds/v3_p0_custom_cls_bce_s{_seed}.yaml",
            f"configs/v3_seeds/v3_p6_cleaner_weighted_attn_msd_s{_seed}.yaml",
            f"configs/v3_seeds/v3_l2_weighted_bce_s{_seed}.yaml",
        ],
    )
    PIP_LINES[f"seed{_seed}"] = PIP_LINES["loss"]

# V4 strict one-factor-at-a-time BERT suites (pure ASCII titles for Kaggle compatibility).
SUITES["v4_ladder_a"] = (
    "V4 - BERT Ladder A (10 epochs + Early Stopping): S0 Stock Head, S1 Custom Head, S2 Cleaner",
    [
        "configs/v4/v4_s0_stock_bce.yaml",
        "configs/v4/v4_s1_custom_head_bce.yaml",
        "configs/v4/v4_s2_cleaner_bce.yaml",
    ],
)
SUITES["v4_ladder_b"] = (
    "V4 - BERT Ladder B (10 epochs + Early Stopping): S3 Hybrid Pool + MSD, S4 Damped Weighted BCE",
    [
        "configs/v4/v4_s3_hybrid_pool_msd_bce.yaml",
        "configs/v4/v4_s4_damped_wbce.yaml",
    ],
)
SUITES["v4_loss_ctl"] = (
    "V4 - BERT Controlled Loss Ablation on S3 Architecture: Naive W-BCE, Focal No-Alpha, ASL",
    [
        "configs/v4/v4_loss_naive_wbce.yaml",
        "configs/v4/v4_loss_focal_noalpha.yaml",
        "configs/v4/v4_loss_asl.yaml",
    ],
)
SUITES["v4_seeds"] = (
    "V4 - BERT Multi-Seed Reproducibility for S4: seed 1337 and seed 2024",
    [
        "configs/v4/v4_s4_damped_wbce_s1337.yaml",
        "configs/v4/v4_s4_damped_wbce_s2024.yaml",
    ],
)
SUITES["v4_valfix"] = (
    "V4 - Validation-Only Error-Driven Fix Suite (ZERO Test Leakage): ValFix-2B and ValFix-2A",
    [
        "configs/v4/v4_s4_valfix_2b.yaml",
        "configs/v4/v4_s4_valfix_2a.yaml",
    ],
)
SUITES["v5_15ep_a"] = (
    "V5 - 15 Epochs + Adaptive LR Reduction (ReduceLROnPlateau) Ladder A: S0, S1, S2",
    [
        "configs/v5_15ep/v5_s0_stock_bce_15ep.yaml",
        "configs/v5_15ep/v5_s1_custom_head_bce_15ep.yaml",
        "configs/v5_15ep/v5_s2_cleaner_bce_15ep.yaml",
    ],
)
SUITES["v5_15ep_b"] = (
    "V5 - 15 Epochs + Adaptive LR Reduction (ReduceLROnPlateau & LLRD) Ladder B: S3, S4, S4+LLRD",
    [
        "configs/v5_15ep/v5_s3_hybrid_pool_msd_bce_15ep.yaml",
        "configs/v5_15ep/v5_s4_damped_wbce_15ep.yaml",
        "configs/v5_15ep/v5_s4_damped_wbce_15ep_llrd.yaml",
    ],
)
for _v4k in ("v4_ladder_a", "v4_ladder_b", "v4_loss_ctl", "v4_seeds", "v4_valfix", "v5_15ep_a", "v5_15ep_b"):
    PIP_LINES[_v4k] = PIP_LINES["loss"]

BASE_SUITES = list(SUITES)

# One notebook per config for CPU-only execution (used when the weekly GPU quota is exhausted). A single config per
# kernel keeps each session far below Kaggle's 12-hour limit and lets the runs execute in parallel across accounts.
CPU_SINGLE_KEYS = []
for _suite, _prefix in (("loss", "cpu_"), ("technique", "cpu_"), ("d1_native", "gpu_")):
    for _cfg in SUITES[_suite][1]:
        _stem = Path(_cfg).stem  # e.g. v3_l1_bce
        _key = _prefix + _stem[len("v3_"):]
        SUITES[_key] = (f"Vòng 3 — {_stem} (một cấu hình)", [_cfg])
        PIP_LINES[_key] = PIP_LINES[_suite]
        CPU_SINGLE_KEYS.append(_key)


def _make_cell(src: str, ctype: str = "code") -> dict:
    cell = {"cell_type": ctype, "metadata": {}, "source": src}
    if ctype == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    return cell


def _project_zip_b64() -> str:
    """Zips only what the runner needs: package source, runner script, configs."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for pattern in ("tweet_topic/**/*.py", "configs/**/*.yaml", "scripts/run.py"):
            for path in sorted(ROOT.glob(pattern)):
                if "__pycache__" in path.parts or "egg-info" in str(path):
                    continue
                zf.write(path, path.relative_to(ROOT).as_posix())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_notebook(suite: str, smoke: bool, payload: str) -> Path:
    title, configs = SUITES[suite]
    pip_line = PIP_LINES[suite]
    env_note = (
        "\n\nThis suite upgrades `transformers` to >=5.19 and installs torchvision/pillow for the d1-2B "
        "remote code. Restart is not required; the next cell imports it fresh."
        if suite == "d1_native"
        else ""
    )
    cells = [
        _make_cell(
            f"# {title}\n\n"
            f"Mode: **{'SMOKE (tiny subset, validates the pipeline)' if smoke else 'FULL'}**\n\n"
            "Runs the project package through `scripts/run.py`. Configs: "
            + ", ".join(f"`{c}`" for c in configs)
            + ".\n\nResults land in `/kaggle/working/tweet_topic_benchmark/outputs/<experiment>/` "
            "(`metrics_summary.json`, `training_history.json`, `test_predictions.json`, `val_probabilities.json`)."
            + env_note,
            "markdown",
        ),
        _make_cell(pip_line),
        _make_cell(
            "# Unpack the embedded project source\n"
            "import base64, io, zipfile, pathlib\n"
            f"PAYLOAD = '{payload}'\n"
            "ROOT = pathlib.Path('/kaggle/working/tweet_topic_benchmark')\n"
            "ROOT.mkdir(parents=True, exist_ok=True)\n"
            "zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))).extractall(ROOT)\n"
            "print(sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob('*.py'))[:8], '...')"
        ),
        _make_cell(
            "# Run each config in its own process; keep going if one fails, report at the end\n"
            "import subprocess, sys\n"
            f"CONFIGS = {json.dumps(configs)}\n"
            f"SMOKE = {smoke}\n"
            "failures = []\n"
            "for cfg in CONFIGS:\n"
            "    cmd = [sys.executable, 'scripts/run.py', '--config', cfg] + (['--smoke'] if SMOKE else [])\n"
            "    print('\\n' + '=' * 70 + '\\nRUN', ' '.join(cmd) + '\\n' + '=' * 70, flush=True)\n"
            "    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n"
            "    for line in proc.stdout:\n"
            "        print(line, end='', flush=True)\n"
            "    if proc.wait() != 0:\n"
            "        failures.append(cfg)\n"
            "print('\\nFAILED CONFIGS:', failures if failures else 'none')\n"
            "if failures:\n"
            "    raise RuntimeError(f'{len(failures)} config(s) failed: {failures}')"
        ),
        _make_cell(
            "# Compact summary of what was produced\n"
            "import json\n"
            "for p in sorted((ROOT / 'outputs').glob('*/metrics_summary.json')):\n"
            "    s = json.loads(p.read_text())\n"
            "    print(s['experiment_name'])\n"
            "    vm = s.get('validation_metrics', {})\n"
            "    if vm:\n"
            "        print(f\"  validation: micro-F1 {vm['micro_f1']:.4f} | macro-F1 {vm['macro_f1']:.4f} | exact {vm['exact_match_ratio']:.4f}\")\n"
            "    for split, m in s.get('test_metrics', {}).items():\n"
            "        print(f\"  {split}: micro-F1 {m['micro_f1']:.4f} | macro-F1 {m['macro_f1']:.4f} | exact {m['exact_match_ratio']:.4f}\")"
        ),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    stem = suite if suite.startswith(("v4_", "v5_")) else f"{PREFIX}_{suite}"
    dest = NOTEBOOKS_DIR / f"{stem}{'_smoke' if smoke else ''}.ipynb"
    dest.write_text(json.dumps(nb, indent=1, ensure_ascii=suite.startswith(("v4_", "v5_"))), encoding="utf-8")
    return dest


def main() -> None:
    payload = _project_zip_b64()
    print(f"Embedded source payload: {len(payload) / 1024:.0f} KiB (base64)")
    for suite in BASE_SUITES:
        for smoke in (True, False):
            print("Generated:", build_notebook(suite, smoke, payload))
    for key in CPU_SINGLE_KEYS:
        print("Generated:", build_notebook(key, False, payload))


if __name__ == "__main__":
    main()
