"""Re-evaluate a trained model on val + test set and save results to CSV.

Loads existing checkpoint (.ckpt for DL, .pkl for classical) — no retraining.
Use this to fix/update results in the CSV without re-running training.

Usage examples
--------------
    # Classical ML (loads artifacts/models/<pipeline_name>.pkl)
    python eval.py model=random_forest feature_extraction=mfcc pipeline_name=mfcc_random_forest

    # DL — checkpoint inferred from pipeline_name
    python eval.py model=mlp feature_extraction=mfcc pipeline_name=mfcc_mlp
    python eval.py model=cnn feature_extraction=logmel pipeline_name=logmel_cnn

    # Override checkpoint path explicitly
    python eval.py model=mlp feature_extraction=mfcc pipeline_name=mfcc_mlp \
                   eval.ckpt_path=outputs/checkpoints/mfcc_mlp/best.ckpt
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_tmp = os.environ.get("TMPDIR") or "/tmp"
if not os.path.isdir(_tmp):
    _tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp")
    os.makedirs(_tmp, exist_ok=True)
    os.environ["TMPDIR"] = _tmp
    tempfile.tempdir = _tmp

import torch
_orig_torch_load = torch.load
def _torch_load_unsafe(f, map_location=None, pickle_module=None, *, weights_only=False, mmap=None, **kw):
    return _orig_torch_load(f, map_location=map_location, pickle_module=pickle_module,
                            weights_only=False, mmap=mmap, **kw)
torch.load = _torch_load_unsafe

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from src.data.download import ensure_dataset
from src.features.extractor import FeatureExtractor
from src.utils.logger import ExperimentLogger

_CLASSICAL = {"random_forest", "svm", "logistic_regression"}
_DL        = {"mlp", "cnn"}


def _eval_classical(cfg: DictConfig, extractor: FeatureExtractor) -> dict:
    from joblib import load as jload
    from src.data.dataset import AudioPreprocessor, load_dataframe, get_or_create_splits
    from src.evaluation.metrics import compute_metrics
    from src.models.classical import extract_split

    pipeline_name = cfg.get("pipeline_name", cfg.model.type)
    artifacts_dir = cfg.training.get("model_artifacts_dir", "artifacts/models")
    pkl_path = os.path.join(artifacts_dir, f"{pipeline_name}.pkl")

    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(
            f"No saved model at {pkl_path!r}. "
            "Train first with train.py or run_pipelines.py."
        )

    print(f"Loading classical model: {pkl_path}")
    artifact      = jload(pkl_path)
    model         = artifact["model"]
    label_encoder = artifact["label_encoder"]

    preprocessor = AudioPreprocessor(cfg.preprocessing)
    df, _        = load_dataframe(cfg.data)
    cache_dir    = cfg.data.get("feature_cache_dir", "data/processed")
    _, val_df, test_df = get_or_create_splits(df, cfg.data, cfg.seed, cache_dir=cache_dir)

    cache = None
    if cfg.data.get("use_feature_cache", True):
        from src.features.cache import FeatureCache
        cache = FeatureCache(
            cfg.data.get("feature_cache_dir", "data/processed"),
            extractor, preprocessor,
        )

    print("Extracting val features ...")
    X_val, y_val   = extract_split(val_df,  preprocessor, extractor, label_encoder, "val",  cache=cache)
    print("Extracting test features ...")
    X_test, y_test = extract_split(test_df, preprocessor, extractor, label_encoder, "test", cache=cache)

    val_metrics  = compute_metrics(y_val,  model.predict(X_val),  prefix="val")
    test_metrics = compute_metrics(y_test, model.predict(X_test), prefix="test")

    print(f"val_acc={val_metrics['val/acc']:.4f}  val_f1={val_metrics['val/f1']:.4f}")
    print(f"test_acc={test_metrics['test/acc']:.4f}  test_f1={test_metrics['test/f1']:.4f}")
    return {**val_metrics, **test_metrics}


def _eval_dl(cfg: DictConfig, extractor: FeatureExtractor) -> dict:
    from src.data.datamodule import EmotionDataModule

    pipeline_name = cfg.get("pipeline_name", cfg.model.type)
    model_type    = cfg.model.type

    # Resolve checkpoint path
    eval_cfg  = cfg.get("eval", {})
    ckpt_path = eval_cfg.get("ckpt_path", None) or \
                os.path.join("outputs", "checkpoints", pipeline_name, "best.ckpt")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path!r}. "
            "Train first or pass eval.ckpt_path=<path>."
        )

    print(f"Loading checkpoint: {ckpt_path}")

    datamodule = EmotionDataModule(cfg, extractor)
    datamodule.setup()
    num_classes = datamodule.num_classes

    if model_type == "mlp":
        from src.models.mlp import MLPModule
        model = MLPModule.load_from_checkpoint(
            ckpt_path,
            cfg=cfg,
            input_dim=extractor.get_feature_dim(),
            num_classes=num_classes,
        )
    elif model_type == "cnn":
        from src.models.cnn import CNNModule
        model = CNNModule.load_from_checkpoint(
            ckpt_path,
            cfg=cfg,
            num_classes=num_classes,
            in_chans=extractor.get_output_channels(),
        )
    else:
        raise ValueError(f"Unknown DL model: {model_type!r}")

    trainer = pl.Trainer(
        accelerator="auto",
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    # Val metrics
    val_out  = trainer.validate(model, datamodule, verbose=False)
    val_acc  = round(float(val_out[0].get("val/acc", 0.0)), 4) if val_out else ""
    val_f1   = round(float(val_out[0].get("val/f1",  0.0)), 4) if val_out else ""

    # Test metrics
    test_out  = trainer.test(model, datamodule, verbose=False)
    test_acc  = round(float(test_out[0].get("test/acc", 0.0)), 4) if test_out else ""
    test_f1   = round(float(test_out[0].get("test/f1",  0.0)), 4) if test_out else ""

    print(f"val_acc={val_acc}  val_f1={val_f1}")
    print(f"test_acc={test_acc}  test_f1={test_f1}")
    return {"val/acc": val_acc, "val/f1": val_f1,
            "test/acc": test_acc, "test/f1": test_f1}


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)
    ensure_dataset(cfg.data)

    extractor  = FeatureExtractor(cfg.feature_extraction, cfg.preprocessing)
    model_type = cfg.model.type

    if model_type in _CLASSICAL:
        results = _eval_classical(cfg, extractor)
    elif model_type in _DL:
        results = _eval_dl(cfg, extractor)
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")

    # Log with train_time_sec=0 (no training happened)
    exp_logger = ExperimentLogger(cfg)
    exp_logger.log_run(results, train_time=0.0)


if __name__ == "__main__":
    main()
