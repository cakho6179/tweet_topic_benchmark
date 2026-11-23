"""Generates the V4 strict one-factor-at-a-time BERT experiment configs (10 epochs + Early Stopping).

Usage:
    python scripts/make_v4_configs.py
"""

from __future__ import annotations

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "configs" / "v4"

BASE_MODEL = {
    "model_type": "bert_encoder",
    "model_name": "bert-base-uncased",
    "max_length": 128,
    "batch_size": 32,
    "eval_batch_size": 64,
    "epochs": 10,
    "lr": 2.0e-5,
    "head_lr": 5.0e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "scheduler": "cosine",
    "grad_clip": 1.0,
    "seed": 42,
    "save_best": True,
    "select_metric": "val_f1_balanced",
    "early_stopping_patience": 3,
    "msd_p": 0.2,
}

DATA = {
    "dataset_name": "cardiffnlp/tweet_topic_multi",
    "train_split": "train_2020",
    "val_split": "validation_2020",
    "test_splits": ["test_2020", "test_2021"],
}

POSTPROCESS = {
    "threshold_mode": "global",
    "criterion": "micro_f1",
    "shrinkage_k": 20.0,
    "calibration_report": True,
    "selective": {"mode": "margin", "coverages": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]},
}


def bert_v4_config(
    name: str,
    *,
    loss: str = "bce",
    loss_params: dict | None = None,
    pooling: str = "cls",
    msd_samples: int = 1,
    force_custom_head: bool = True,
    head_lr: float = 5.0e-4,
    attn_dim: int = 256,
    preprocessor: str | None = None,
    threshold_mode: str = "global",
    seed: int = 42,
) -> dict:
    model = dict(BASE_MODEL)
    model.update(
        loss_type=loss,
        loss_params=loss_params or {},
        pooling=pooling,
        msd_samples=msd_samples,
        force_custom_head=force_custom_head,
        head_lr=head_lr,
        attn_dim=attn_dim,
        seed=seed,
    )
    pp = dict(POSTPROCESS)
    pp["threshold_mode"] = threshold_mode
    cfg = {
        "experiment_name": name,
        "output_dir": f"./outputs/{name}",
        **pp,
        "data": DATA,
        "model": model,
    }
    if preprocessor:
        cfg["preprocessor"] = preprocessor
    return cfg


CONFIGS = {
    # ---- Progressive One-Factor-at-a-Time Ladder (S0 -> S4; S5 is Shrinkage post-hoc on S4) ----
    "v4_s0_stock_bce": bert_v4_config(
        "v4_s0_stock_bce",
        loss="bce",
        pooling="cls",
        msd_samples=1,
        force_custom_head=False,
        head_lr=2.0e-5,
        preprocessor=None,
    ),
    "v4_s1_custom_head_bce": bert_v4_config(
        "v4_s1_custom_head_bce",
        loss="bce",
        pooling="cls",
        msd_samples=1,
        force_custom_head=True,
        head_lr=5.0e-4,
        preprocessor=None,
    ),
    "v4_s2_cleaner_bce": bert_v4_config(
        "v4_s2_cleaner_bce",
        loss="bce",
        pooling="cls",
        msd_samples=1,
        force_custom_head=True,
        head_lr=5.0e-4,
        preprocessor="tweet_text_cleaner",
    ),
    "v4_s3_hybrid_pool_msd_bce": bert_v4_config(
        "v4_s3_hybrid_pool_msd_bce",
        loss="bce",
        pooling="hybrid",
        msd_samples=5,
        force_custom_head=True,
        head_lr=5.0e-4,
        preprocessor="tweet_text_cleaner",
    ),
    "v4_s4_damped_wbce": bert_v4_config(
        "v4_s4_damped_wbce",
        loss="weighted_bce",
        loss_params={"power": 0.5, "pos_weight_max": 6.0},
        pooling="hybrid",
        msd_samples=5,
        force_custom_head=True,
        head_lr=5.0e-4,
        preprocessor="tweet_text_cleaner",
    ),
    # ---- Controlled Loss Ablation on the S3 Architecture (Cleaner + Hybrid Pooling + MSD) ----
    "v4_loss_naive_wbce": bert_v4_config(
        "v4_loss_naive_wbce",
        loss="weighted_bce",
        loss_params={"power": 1.0, "pos_weight_max": 30.0},
        pooling="hybrid",
        msd_samples=5,
        force_custom_head=True,
        head_lr=5.0e-4,
        preprocessor="tweet_text_cleaner",
    ),
    "v4_loss_focal_noalpha": bert_v4_config(
        "v4_loss_focal_noalpha",
        loss="focal",
        loss_params={"gamma": 2.0, "alpha": None},
        pooling="hybrid",
        msd_samples=5,
        force_custom_head=True,
        head_lr=5.0e-4,
        preprocessor="tweet_text_cleaner",
    ),
    "v4_loss_asl": bert_v4_config(
        "v4_loss_asl",
        loss="asymmetric",
        loss_params={"gamma_neg": 2.0, "gamma_pos": 0.0, "clip": 0.05},
        pooling="hybrid",
        msd_samples=5,
        force_custom_head=True,
        head_lr=5.0e-4,
        preprocessor="tweet_text_cleaner",
    ),
    # ---- Multi-Seed Reproducibility for the Final BERT Configuration (S4) ----
    "v4_s4_damped_wbce_s1337": bert_v4_config(
        "v4_s4_damped_wbce_s1337",
        loss="weighted_bce",
        loss_params={"power": 0.5, "pos_weight_max": 6.0},
        pooling="hybrid",
        msd_samples=5,
        force_custom_head=True,
        head_lr=5.0e-4,
        preprocessor="tweet_text_cleaner",
        seed=1337,
    ),
    "v4_s4_damped_wbce_s2024": bert_v4_config(
        "v4_s4_damped_wbce_s2024",
        loss="weighted_bce",
        loss_params={"power": 0.5, "pos_weight_max": 6.0},
        pooling="hybrid",
        msd_samples=5,
        force_custom_head=True,
        head_lr=5.0e-4,
        preprocessor="tweet_text_cleaner",
        seed=2024,
    ),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, cfg in CONFIGS.items():
        path = OUT / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
