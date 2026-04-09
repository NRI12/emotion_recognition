"""Classical ML training pipeline.

Orchestrates: data loading → feature extraction → sklearn fit → eval → save.
Feature extraction and model building are delegated to src.models.classical.
"""
from __future__ import annotations

import time
from typing import Any, Dict

from omegaconf import DictConfig

from src.data.dataset import AudioPreprocessor, load_dataframe, split_dataframe
from src.evaluation.metrics import compute_metrics
from src.features.extractor import FeatureExtractor
from src.models.classical import build_sklearn_pipeline, extract_split, save_classical_model
from src.pipelines.base import BasePipeline


class ClassicalPipeline(BasePipeline):
    """End-to-end classical ML pipeline: extract → fit → evaluate → save."""

    def run(self) -> Dict[str, Any]:
        cfg = self.cfg
        preprocessor = AudioPreprocessor(cfg.preprocessing)
        df, label_encoder = load_dataframe(cfg.data)

        train_df, val_df, test_df = split_dataframe(
            df,
            train_ratio=cfg.data.train_ratio,
            val_ratio=cfg.data.val_ratio,
            stratify=cfg.data.stratify,
            seed=cfg.seed,
        )

        # Feature cache — skip recomputation on subsequent runs
        cache = None
        if cfg.data.get("use_feature_cache", True):
            from src.features.cache import FeatureCache
            cache = FeatureCache(
                cfg.data.get("feature_cache_dir", "data/processed"),
                self.extractor,
                preprocessor,
            )
            print(f"Feature cache: {cache.cache_dir}")

        n_workers = int(cfg.training.get("extraction_workers", 1))
        print(
            f"Feature extraction  [{self.extractor.method}]"
            f"  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}"
            f"  workers={n_workers}"
        )

        X_train, y_train = extract_split(
            train_df, preprocessor, self.extractor, label_encoder,
            split_name="train", n_workers=n_workers, cache=cache,
        )
        X_val, y_val = extract_split(
            val_df, preprocessor, self.extractor, label_encoder,
            split_name="val", n_workers=n_workers, cache=cache,
        )
        X_test, y_test = extract_split(
            test_df, preprocessor, self.extractor, label_encoder,
            split_name="test", n_workers=n_workers, cache=cache,
        )

        model = build_sklearn_pipeline(cfg)
        print(f"Fitting {cfg.model.type} ...")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        print(f"Fit done in {train_time:.1f}s")

        val_metrics  = compute_metrics(y_val,  model.predict(X_val),  prefix="val")
        test_metrics = compute_metrics(y_test, model.predict(X_test), prefix="test")

        print(f"val_acc={val_metrics['val/acc']:.4f}  val_f1={val_metrics['val/f1']:.4f}")
        print(f"test_acc={test_metrics['test/acc']:.4f}  test_f1={test_metrics['test/f1']:.4f}")

        pipeline_name = cfg.get("pipeline_name", f"{cfg.model.type}_{self.extractor.method}")
        save_classical_model(
            model, label_encoder, pipeline_name,
            cfg.training.get("model_artifacts_dir", "artifacts/models"),
        )

        return {
            "model": model,
            "label_encoder": label_encoder,
            "train_time": train_time,
            **val_metrics,
            **test_metrics,
        }
