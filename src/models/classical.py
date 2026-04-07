from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from omegaconf import DictConfig

from src.data.dataset import AudioPreprocessor, load_dataframe, split_dataframe
from src.evaluation.metrics import compute_metrics
from src.features.extractor import FeatureExtractor


def build_sklearn_pipeline(cfg: DictConfig) -> Pipeline:
    """Construct an sklearn Pipeline (StandardScaler + classifier)."""
    model_type: str = cfg.model.type

    if model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=cfg.model.n_estimators,
            max_depth=cfg.model.get("max_depth", None),
            min_samples_split=cfg.model.get("min_samples_split", 2),
            random_state=cfg.seed,
            n_jobs=-1,
        )
    elif model_type == "svm":
        clf = SVC(
            C=cfg.model.C,
            kernel=cfg.model.get("kernel", "rbf"),
            gamma=cfg.model.get("gamma", "scale"),
            probability=True,
            random_state=cfg.seed,
        )
    elif model_type == "logistic_regression":
        clf = LogisticRegression(
            C=cfg.model.C,
            max_iter=cfg.model.get("max_iter", 1000),
            random_state=cfg.seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown classical model type: {model_type}")

    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def extract_split(
    df: pd.DataFrame,
    preprocessor: AudioPreprocessor,
    extractor: FeatureExtractor,
    label_encoder,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract flat features for all rows in a DataFrame."""
    features, labels = [], []
    for _, row in df.iterrows():
        waveform = preprocessor.load(row["path"])
        feat = extractor.extract(waveform).numpy()
        features.append(feat)
        labels.append(int(label_encoder.transform([row["emotion"]])[0]))
    return np.array(features), np.array(labels)


def train_classical_model(
    cfg: DictConfig, extractor: FeatureExtractor
) -> Dict[str, Any]:
    """End-to-end classical ML training and evaluation. Returns metrics dict."""
    preprocessor = AudioPreprocessor(cfg.preprocessing)
    df, label_encoder = load_dataframe(cfg.data)

    train_df, val_df, test_df = split_dataframe(
        df,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        stratify=cfg.data.stratify,
        seed=cfg.seed,
    )

    print(f"Extracting features: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    X_train, y_train = extract_split(train_df, preprocessor, extractor, label_encoder)
    X_val, y_val = extract_split(val_df, preprocessor, extractor, label_encoder)
    X_test, y_test = extract_split(test_df, preprocessor, extractor, label_encoder)

    model = build_sklearn_pipeline(cfg)

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    val_metrics = compute_metrics(y_val, model.predict(X_val), prefix="val")
    test_metrics = compute_metrics(y_test, model.predict(X_test), prefix="test")

    print(f"val_acc={val_metrics['val/acc']:.4f}  val_f1={val_metrics['val/f1']:.4f}")
    print(f"test_acc={test_metrics['test/acc']:.4f}  test_f1={test_metrics['test/f1']:.4f}")

    return {
        "model": model,
        "label_encoder": label_encoder,
        "train_time": train_time,
        **val_metrics,
        **test_metrics,
    }
