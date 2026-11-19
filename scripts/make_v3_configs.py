"""Generates the v3 experiment configs (BERT loss + progressive technique ablation, Liquid d1 native).

Run once after changing the design:
    python scripts/make_v3_configs.py

The generated YAMLs are local artifacts (``/configs`` is gitignored); this script exists so the whole ablation
matrix is visible in one place and every run shares identical training hyper-parameters except for the factor
under test.
"""

from __future__ import annotations

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "configs" / "v3"

BASE_MODEL = {
    "model_type": "bert_encoder",
    "model_name": "bert-base-uncased",
    "max_length": 128,
    "batch_size": 32,
    "eval_batch_size": 64,
    "epochs": 4,
    "lr": 2.0e-5,
    "head_lr": 1.0e-3,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "scheduler": "linear",
    "grad_clip": 1.0,
    "seed": 42,
    "save_best": True,
    "select_metric": "val_macro_f1_tuned",
    "early_stopping_patience": 2,
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
    "calibration_report": True,
    "selective": {"mode": "margin", "coverages": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]},
}


def bert_config(name: str, *, loss: str, loss_params: dict | None = None, pooling: str = "cls",
                msd_samples: int = 1, force_custom_head: bool = False,
                attn_dim: int = 256, preprocessor: str | None = None) -> dict:
    model = dict(BASE_MODEL)
    model.update(
        loss_type=loss,
        loss_params=loss_params or {},
        pooling=pooling,
        msd_samples=msd_samples,
        force_custom_head=force_custom_head,
        attn_dim=attn_dim,
    )
    cfg = {
        "experiment_name": name,
        "output_dir": f"./outputs/{name}",
        **POSTPROCESS,
        "data": DATA,
        "model": model,
    }
    if preprocessor:
        cfg["preprocessor"] = preprocessor
    return cfg


CONFIGS = {
    # ---- Loss ablation: one variable (the loss) on a stock BERT classification head --------------------
    "v3_l1_bce": bert_config("v3_l1_bce", loss="bce"),
    "v3_l2_weighted_bce": bert_config(
        "v3_l2_weighted_bce", loss="weighted_bce", loss_params={"pos_weight_max": 30.0}
    ),
    "v3_l3_focal_g2": bert_config(
        "v3_l3_focal_g2", loss="focal", loss_params={"gamma": 2.0, "alpha": 0.25}
    ),
    "v3_l4_focal_g1": bert_config(
        "v3_l4_focal_g1", loss="focal", loss_params={"gamma": 1.0, "alpha": 0.25}
    ),
    "v3_l5_focal_g2_noalpha": bert_config(
        "v3_l5_focal_g2_noalpha", loss="focal", loss_params={"gamma": 2.0, "alpha": None}
    ),
    # ---- Progressive technique ablation ---------------------------------------------------------------
    # P0 uses the same custom head as P4/P5/P6 so that pooling is a controlled comparison.
    "v3_p0_custom_cls_bce": bert_config(
        "v3_p0_custom_cls_bce", loss="bce", force_custom_head=True
    ),
    "v3_p2_cleaner_bce": bert_config(
        "v3_p2_cleaner_bce", loss="bce", force_custom_head=True,
        preprocessor="tweet_text_cleaner",
    ),
    "v3_p3_cleaner_weighted_bce": bert_config(
        "v3_p3_cleaner_weighted_bce", loss="weighted_bce", loss_params={"pos_weight_max": 30.0},
        force_custom_head=True, preprocessor="tweet_text_cleaner",
    ),
    "v3_p4_cleaner_weighted_mean": bert_config(
        "v3_p4_cleaner_weighted_mean", loss="weighted_bce", loss_params={"pos_weight_max": 30.0},
        pooling="mean", force_custom_head=True, preprocessor="tweet_text_cleaner",
    ),
    "v3_p5_cleaner_weighted_attention": bert_config(
        "v3_p5_cleaner_weighted_attention", loss="weighted_bce", loss_params={"pos_weight_max": 30.0},
        pooling="attention", attn_dim=256, force_custom_head=True, preprocessor="tweet_text_cleaner",
    ),
    "v3_p6_cleaner_weighted_attn_msd": bert_config(
        "v3_p6_cleaner_weighted_attn_msd", loss="weighted_bce", loss_params={"pos_weight_max": 30.0},
        pooling="attention", attn_dim=256, msd_samples=5, force_custom_head=True,
        preprocessor="tweet_text_cleaner",
    ),
    # ---- Liquid d1-2B native System One API (zero-shot, no training) ----------------------------------
    "v3_d1_system_one": {
        "experiment_name": "v3_d1_system_one",
        "output_dir": "./outputs/v3_d1_system_one",
        **POSTPROCESS,
        "data": DATA,
        "model": {
            "model_type": "decision_d1_system_one",
            "model_name": "LiquidAI/d1-2B",
            "batch_size": 8,
            "question_mode": "choice_multiple",
        },
    },
    "v3_d1_system_one_noul": {
        "experiment_name": "v3_d1_system_one_noul",
        "output_dir": "./outputs/v3_d1_system_one_noul",
        **POSTPROCESS,
        "data": DATA,
        "model": {
            "model_type": "decision_d1_system_one",
            "model_name": "LiquidAI/d1-2B",
            "batch_size": 8,
            "question_mode": "noul",
        },
    },
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, cfg in CONFIGS.items():
        path = OUT / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
