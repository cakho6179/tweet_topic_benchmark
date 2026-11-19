"""Creates Kaggle kernel bundles (notebook + kernel-metadata.json) for the v3 suites.

Usage:
    python scripts/make_kaggle_bundle.py v3_loss_smoke v3_d1_native_smoke   # validate on a tiny subset
    python scripts/make_kaggle_bundle.py v3_loss v3_technique v3_d1_native  # full suites

Bundle names match the notebooks produced by ``scripts/build_package_notebooks.py``. They land in
``kaggle/kernels_generic/tweet-topic-<suite>/`` and are pushed with the multi-account tool:
    python ~/.claude/skills/kaggle-multi-account/scripts/kaggle_acc.py push --acc acc2 --gpu <bundle>
That tool rewrites the kernel id to the chosen account, so the placeholder owner below is never used.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(suites: list) -> None:
    for suite in suites:
        src = ROOT / "notebooks" / f"{suite}.ipynb"
        if not src.exists():
            raise SystemExit(f"Missing {src}; run scripts/build_package_notebooks.py first.")
        slug = "tweet-topic-" + suite.replace("_", "-")
        out = ROOT / "kaggle" / "kernels_generic" / slug
        out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out / "notebook.ipynb")
        meta = {
            "id": f"placeholder/{slug}", "title": slug, "code_file": "notebook.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
            "model_sources": [],
        }
        (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("Bundle:", out)


if __name__ == "__main__":
    main(sys.argv[1:] or ["v3_loss_smoke"])
