"""Hyperparameter tuning with Optuna.

Usage examples:
    # Tune RandomForest
    python tune.py model=random_forest feature_extraction=mfcc tuning.n_trials=30

    # Tune MLP
    python tune.py model=mlp feature_extraction=mfcc tuning.n_trials=20

    # Tune CNN
    python tune.py model=cnn feature_extraction=melspec tuning.n_trials=15
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import hydra
import optuna
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from src.data.datamodule import EmotionDataModule
from src.data.download import ensure_dataset
from src.features.extractor import FeatureExtractor
from src.models.cnn import CNNModule
from src.models.classical import build_sklearn_pipeline, extract_split
from src.models.mlp import MLPModule
from src.evaluation.metrics import compute_metrics
from src.data.dataset import AudioPreprocessor, load_dataframe, split_dataframe

_CLASSICAL = {"random_forest", "svm", "logistic_regression"}


# ------------------------------------------------------------------
# Objective functions
# ------------------------------------------------------------------

def _objective_classical(base_cfg: DictConfig, trial: optuna.Trial) -> float:
    cfg_dict = OmegaConf.to_container(base_cfg, resolve=True)
    model_type = cfg_dict["model"]["type"]

    if model_type == "random_forest":
        cfg_dict["model"]["n_estimators"] = trial.suggest_int("n_estimators", 50, 500)
        cfg_dict["model"]["max_depth"] = trial.suggest_int("max_depth", 3, 30)
        cfg_dict["model"]["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 10)
    elif model_type == "svm":
        cfg_dict["model"]["C"] = trial.suggest_float("C", 0.01, 100.0, log=True)
        cfg_dict["model"]["kernel"] = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])
    elif model_type == "logistic_regression":
        cfg_dict["model"]["C"] = trial.suggest_float("C", 0.01, 100.0, log=True)

    cfg = OmegaConf.create(cfg_dict)
    extractor = FeatureExtractor(cfg.feature_extraction, cfg.preprocessing)
    preprocessor = AudioPreprocessor(cfg.preprocessing)

    df, label_encoder = load_dataframe(cfg.data)
    train_df, val_df, _ = split_dataframe(
        df,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        stratify=cfg.data.stratify,
        seed=cfg.seed,
    )
    X_train, y_train = extract_split(train_df, preprocessor, extractor, label_encoder)
    X_val, y_val = extract_split(val_df, preprocessor, extractor, label_encoder)

    model = build_sklearn_pipeline(cfg)
    model.fit(X_train, y_train)
    metrics = compute_metrics(y_val, model.predict(X_val), prefix="val")
    return metrics["val/f1"]


def _objective_dl(base_cfg: DictConfig, trial: optuna.Trial) -> float:
    cfg_dict = OmegaConf.to_container(base_cfg, resolve=True)
    model_type = cfg_dict["model"]["type"]

    cfg_dict["training"]["lr"] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    cfg_dict["training"]["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64])

    if model_type == "mlp":
        h = trial.suggest_categorical("hidden_size", [128, 256, 512])
        cfg_dict["model"]["hidden_sizes"] = [h, h // 2]
        cfg_dict["model"]["dropout"] = trial.suggest_float("dropout", 0.1, 0.5)
    elif model_type == "cnn":
        cfg_dict["model"]["backbone"] = trial.suggest_categorical(
            "backbone", ["resnet18", "efficientnet_b0"]
        )

    cfg = OmegaConf.create(cfg_dict)
    extractor = FeatureExtractor(cfg.feature_extraction, cfg.preprocessing)
    datamodule = EmotionDataModule(cfg, extractor)
    datamodule.setup()
    num_classes = datamodule.num_classes

    if model_type == "mlp":
        model = MLPModule(cfg, input_dim=extractor.get_feature_dim(), num_classes=num_classes)
    else:
        model = CNNModule(cfg, num_classes=num_classes, in_chans=extractor.get_output_channels())

    trainer = pl.Trainer(
        max_epochs=cfg.tuning.tune_epochs,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="val/f1", patience=5, mode="max")
        ],
        accelerator="auto",
    )
    trainer.fit(model, datamodule)
    return float(trainer.callback_metrics.get("val/f1", 0.0))


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)
    ensure_dataset(cfg.data)

    model_type = cfg.model.type
    feat_method = cfg.feature_extraction.method
    study_name = f"{model_type}_{feat_method}"

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
    )

    objective = (
        _objective_classical if model_type in _CLASSICAL else _objective_dl
    )
    study.optimize(lambda t: objective(cfg, t), n_trials=cfg.tuning.n_trials)

    print(f"\nBest val/f1: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    os.makedirs("outputs/best_params", exist_ok=True)
    out = {
        "study_name": study_name,
        "best_val_f1": study.best_value,
        "best_params": study.best_params,
    }
    path = f"outputs/best_params/{study_name}.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Best params saved -> {path}")


if __name__ == "__main__":
    main()
