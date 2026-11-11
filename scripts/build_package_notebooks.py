"""Builds self-contained Kaggle notebooks that run the *package* (``scripts/run.py`` + YAML configs).

Why this exists
---------------
The older notebooks (01-06) re-implement their model inline and never import ``tweet_topic/``. Anything added to the
package (attention pooling, MSD, temperature scaling, coverage curves, the d1 probe, the QLoRA classifier, the
auditor) is therefore invisible to them. These notebooks close that gap: the project source is embedded in the
notebook as a base64 zip, unpacked on Kaggle, and executed with ``python scripts/run.py --config ...``.

Nothing is executed locally. Rebuild after every code change:

    python scripts/build_package_notebooks.py

Outputs go to ``notebooks/`` as ``v2_<suite>.ipynb`` (full) and ``v2_<suite>_smoke.ipynb`` (64 train / 32 eval
samples, a few minutes) so a suite can be validated on Kaggle before spending the full GPU budget.
"""

from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT / "notebooks"

# suite key -> (title, [configs run sequentially])
SUITES = {
    "encoders": (
        "Twitter-RoBERTa (clean checkpoint): BCE control, temperature scaling, ASL + per-class thresholds, "
        "attention pooling + MSD",
        [
            "configs/v2/roberta_dec2020_bce.yaml",
            "configs/v2/roberta_dec2020_bce_tempscale.yaml",
            "configs/v2/roberta_dec2020_asl_perclass.yaml",
            "configs/v2/roberta_dec2020_attnpool_msd_asl.yaml",
        ],
    ),
    "d1": (
        "Liquid d1-2B: frozen-feature linear probe (main) and option-letter zero-shot (secondary)",
        ["configs/v2/d1_probe.yaml", "configs/v2/d1_zero_shot_letters.yaml"],
    ),
    "qwen_qlora": (
        "Qwen3.5-4B QLoRA supervised multi-label classifier",
        ["configs/v2/qwen_qlora_cls.yaml"],
    ),
    "ml_audit": (
        "TF-IDF + preprocessor add-on, with Qwen3.5-4B chain-of-thought auditor on ambiguous tweets",
        ["configs/v2/ml_ovr_preproc_audit.yaml"],
    ),
}

PIP_LINE = "!pip install -q transformers accelerate bitsandbytes datasets torch scikit-learn peft scipy pyyaml einops timm"


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
    cells = [
        _make_cell(
            f"# {title}\n\n"
            f"Mode: **{'SMOKE (tiny subset, validates the pipeline)' if smoke else 'FULL'}**\n\n"
            "Runs the project package through `scripts/run.py`. Configs: "
            + ", ".join(f"`{c}`" for c in configs)
            + ".\n\nResults land in `/kaggle/working/tweet_topic_benchmark/outputs/<experiment>/` "
            "(`metrics_summary.json`, `test_predictions.json`, `val_probabilities.json`).",
            "markdown",
        ),
        _make_cell(PIP_LINE),
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
            "    for split, m in s['test_metrics'].items():\n"
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
    dest = NOTEBOOKS_DIR / f"v2_{suite}{'_smoke' if smoke else ''}.ipynb"
    dest.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return dest


def main() -> None:
    payload = _project_zip_b64()
    print(f"Embedded source payload: {len(payload) / 1024:.0f} KiB (base64)")
    for suite in SUITES:
        for smoke in (True, False):
            print("Generated:", build_notebook(suite, smoke, payload))


if __name__ == "__main__":
    main()
