"""Multi-Kaggle Multi-Account Dispatcher & Orchestrator.

Manages multi-kernel Kaggle workflows across multiple dedicated Kaggle accounts:
  - Supports multi-account pool (e.g. from ~/.kaggle_accounts/accounts.json)
  - Distributes each baseline kernel to a separate Kaggle account to bypass concurrent GPU quotas
  - Pushes all 4 kernels to run concurrently on Kaggle Cloud in parallel
  - Polls job statuses across the running kernels with account-specific credentials
  - Pulls results, metrics, and predictions from all kernels to generate an aggregated leaderboard
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("multi_kaggle_orchestrator")

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
KERNELS_DIR = BENCHMARK_DIR / "kernel_runners"
OUTPUTS_DIR = BENCHMARK_DIR / "outputs" / "multi_kaggle_results"

KAGGLE_ACCOUNTS_DIR = Path.home() / ".kaggle_accounts"
ACCOUNTS_JSON = KAGGLE_ACCOUNTS_DIR / "accounts.json"

KERNELS_SPEC = {
    "kernel_ml": {
        "slug": "tweet-topic-01-ml-baseline",
        "title": "tweet-topic-01-ml-baseline",
        "notebook_source": BENCHMARK_DIR / "notebooks" / "01_kaggle_ml_baseline.ipynb",
        "is_gpu": False,
        "default_acc": "acc1",
    },
    "kernel_bert": {
        "slug": "tweet-topic-02-bert-baseline",
        "title": "tweet-topic-02-bert-baseline",
        "notebook_source": BENCHMARK_DIR / "notebooks" / "02_kaggle_bert_baseline.ipynb",
        "is_gpu": True,
        "default_acc": "acc2",
    },
    "kernel_d1_liquid": {
        "slug": "tweet-topic-03-decision-d1-liquid",
        "title": "tweet-topic-03-decision-d1-liquid",
        "notebook_source": BENCHMARK_DIR / "notebooks" / "03_kaggle_d1_liquid_decision.ipynb",
        "is_gpu": True,
        "default_acc": "acc3",
    },
    "kernel_qwen_llm": {
        "slug": "tweet-topic-04-llm-qwen3-5-4b",
        "title": "tweet-topic-04-llm-qwen3-5-4b",
        "notebook_source": BENCHMARK_DIR / "notebooks" / "04_kaggle_qwen3_5_4b_llm.ipynb",
        "is_gpu": True,
        "default_acc": "acc4",
    },
}


# v2 suites run the *package* (scripts/run.py + configs/v2/*.yaml) from notebooks built by
# scripts/build_package_notebooks.py. They are opt-in: they are NOT part of the default "push_all" fleet, so a bare
# push_all cannot accidentally spend GPU quota on them. Target them explicitly, e.g.
#   --kernels kernel_v2_d1_smoke           (validate the pipeline on a tiny subset first)
#   --kernels kernel_v2_d1,kernel_v2_encoders
# Only acc1-acc3 exist in ~/.kaggle_accounts; each account can run at most 2 GPU sessions at once.
_V2_SUITE_ACCOUNTS = {"encoders": "acc1", "d1": "acc2", "qwen_qlora": "acc3", "ml_audit": "acc1"}
for _suite, _acc in _V2_SUITE_ACCOUNTS.items():
    for _smoke in (True, False):
        _key = f"kernel_v2_{_suite}{'_smoke' if _smoke else ''}"
        _slug = f"tweet-topic-v2-{_suite.replace('_', '-')}{'-smoke' if _smoke else ''}"
        KERNELS_SPEC[_key] = {
            "slug": _slug,
            "title": _slug,
            "notebook_source": BENCHMARK_DIR / "notebooks" / f"v2_{_suite}{'_smoke' if _smoke else ''}.ipynb",
            "is_gpu": True,
            "default_acc": _acc,
            "opt_in": True,
        }


def load_accounts_pool() -> Dict[str, Dict[str, Any]]:
    """Loads Kaggle accounts from ~/.kaggle_accounts/accounts.json if present."""
    if ACCOUNTS_JSON.exists():
        try:
            with open(ACCOUNTS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info("Loaded multi-account pool with %d accounts: %s", len(data), list(data.keys()))
            return data
        except Exception as e:
            logger.warning("Failed to load accounts.json: %s", e)
    return {}


def get_kernel_account_info(k_key: str, accounts_pool: Dict[str, Any], single_user: Optional[str] = None) -> tuple[str, Optional[Path]]:
    """Returns (username, config_dir) for a given kernel."""
    if single_user:
        return single_user, None

    spec = KERNELS_SPEC.get(k_key, {})
    default_acc = spec.get("default_acc", "acc1")

    if accounts_pool and default_acc in accounts_pool:
        acc_info = accounts_pool[default_acc]
        username = acc_info.get("user")
        acc_dir = KAGGLE_ACCOUNTS_DIR / default_acc
        if (acc_dir / "kaggle.json").exists():
            return username, acc_dir

    # Fallback to single account
    username = os.environ.get("KAGGLE_USERNAME")
    if not username:
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if kaggle_json.exists():
            try:
                data = json.loads(kaggle_json.read_text(encoding="utf-8"))
                username = data.get("username", "kaggle-user")
            except Exception:
                username = "kaggle-user"
        else:
            username = "kaggle-user"

    return username, None


def run_command(cmd: List[str], config_dir: Optional[Path] = None) -> str:
    env = os.environ.copy()
    if config_dir:
        env["KAGGLE_CONFIG_DIR"] = str(config_dir)

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        return res.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error("Command failed: %s\nStderr: %s", " ".join(cmd), e.stderr)
        return e.stderr.strip()


def get_target_specs(target_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    if not target_keys:
        return {k: v for k, v in KERNELS_SPEC.items() if not v.get("opt_in")}
    return {k: v for k, v in KERNELS_SPEC.items() if k in target_keys}


def prepare_kernels(accounts_pool: Dict[str, Any], single_user: Optional[str] = None, target_keys: Optional[List[str]] = None):
    """Sets up kernel-metadata.json and copies notebooks for each baseline runner."""
    KERNELS_DIR.mkdir(parents=True, exist_ok=True)
    specs = get_target_specs(target_keys)
    logger.info("Preparing Multi-Kaggle kernel runners for %d kernels...", len(specs))
    for k_key, spec in specs.items():
        k_dir = KERNELS_DIR / k_key
        k_dir.mkdir(parents=True, exist_ok=True)

        username, config_dir = get_kernel_account_info(k_key, accounts_pool, single_user)
        nb_dest = k_dir / "notebook.ipynb"
        if spec["notebook_source"].exists():
            shutil.copyfile(spec["notebook_source"], nb_dest)

        meta = {
            "id": f"{username}/{spec['slug']}",
            "title": spec["title"],
            "code_file": "notebook.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_gpu": spec["is_gpu"],
            "enable_gpu": spec["is_gpu"],
            "enable_internet": True,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
        }

        meta_path = k_dir / "kernel-metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(
            "Configured [%s] -> Account: %s, Kernel ID: %s/%s",
            k_key,
            config_dir.name if config_dir else "default",
            username,
            spec["slug"],
        )


def push_all_kernels(accounts_pool: Dict[str, Any], single_user: Optional[str] = None, target_keys: Optional[List[str]] = None):
    """Dispatches kernels in parallel on Kaggle using separate accounts."""
    prepare_kernels(accounts_pool, single_user, target_keys)
    specs = get_target_specs(target_keys)
    logger.info("=" * 65)
    logger.info("DISPATCHING MULTI-ACCOUNT MULTI-KAGGLE RUNS IN PARALLEL (%d kernels)", len(specs))
    logger.info("=" * 65)

    for k_key, spec in specs.items():
        k_dir = KERNELS_DIR / k_key
        username, config_dir = get_kernel_account_info(k_key, accounts_pool, single_user)
        acc_label = config_dir.name if config_dir else "default"
        logger.info("Pushing [%s] via Account '%s' (%s)...", spec['slug'], acc_label, username)
        out = run_command(["kaggle", "kernels", "push", "-p", str(k_dir)], config_dir=config_dir)
        print(f"[{acc_label} | {username}] {out}")

    print("\nKernels pushed across separate accounts! Use '--action status' to monitor execution in parallel.")


def check_status(accounts_pool: Dict[str, Any], single_user: Optional[str] = None, target_keys: Optional[List[str]] = None):
    """Checks the live execution status of dispatched kernels across their respective accounts."""
    specs = get_target_specs(target_keys)
    logger.info("Checking status of %d Multi-Kaggle baseline kernels...", len(specs))
    print("\n" + "=" * 75)
    print(f"{'ACCOUNT':<10} | {'USERNAME':<15} | {'KERNEL SLUG':<32} | {'STATUS'}")
    print("=" * 75)
    for k_key, spec in specs.items():
        username, config_dir = get_kernel_account_info(k_key, accounts_pool, single_user)
        acc_label = config_dir.name if config_dir else "default"
        kernel_id = f"{username}/{spec['slug']}"
        status_out = run_command(["kaggle", "kernels", "status", kernel_id], config_dir=config_dir)
        # Clean warning messages from output
        lines = [line for line in status_out.splitlines() if not line.startswith("Warning:")]
        clean_status = lines[-1] if lines else status_out
        print(f"{acc_label:<10} | {username:<15} | {spec['slug']:<32} | {clean_status}")
    print("=" * 75 + "\n")


def pull_all_results(accounts_pool: Dict[str, Any], single_user: Optional[str] = None, target_keys: Optional[List[str]] = None):
    """Downloads outputs from finished baseline kernels across their respective accounts."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    specs = get_target_specs(target_keys)
    logger.info("Pulling outputs from finished Kaggle kernels into: %s", OUTPUTS_DIR)

    for k_key, spec in specs.items():
        username, config_dir = get_kernel_account_info(k_key, accounts_pool, single_user)
        acc_label = config_dir.name if config_dir else "default"
        kernel_id = f"{username}/{spec['slug']}"
        dest_sub = OUTPUTS_DIR / k_key
        dest_sub.mkdir(parents=True, exist_ok=True)
        logger.info("Pulling output for %s (Account: %s)...", kernel_id, acc_label)
        out = run_command(["kaggle", "kernels", "output", kernel_id, "-p", str(dest_sub)], config_dir=config_dir)
        print(f"[{acc_label} | {k_key}] {out}")

    logger.info("All outputs pulled successfully!")


def main():
    parser = argparse.ArgumentParser(description="Multi-Kaggle Multi-Account Benchmark Dispatcher")
    parser.add_argument(
        "--action",
        type=str,
        default="status",
        choices=["prepare", "push_all", "push", "status", "pull_results"],
        help="Action to perform across the multi-kernel fleet",
    )
    parser.add_argument(
        "--kernels",
        type=str,
        default=None,
        help="Comma-separated kernel keys to target (e.g. kernel_d1_finetune,kernel_qwen_finetune)",
    )
    parser.add_argument(
        "--username",
        type=str,
        default=None,
        help="Force a single Kaggle username instead of multi-account pool",
    )
    args = parser.parse_args()

    target_keys = [k.strip() for k in args.kernels.split(",")] if args.kernels else None
    accounts_pool = load_accounts_pool() if not args.username else {}
    if args.action == "prepare":
        prepare_kernels(accounts_pool, args.username, target_keys)
    elif args.action in ["push_all", "push"]:
        push_all_kernels(accounts_pool, args.username, target_keys)
    elif args.action == "status":
        check_status(accounts_pool, args.username, target_keys)
    elif args.action == "pull_results":
        pull_all_results(accounts_pool, args.username, target_keys)


if __name__ == "__main__":
    main()
