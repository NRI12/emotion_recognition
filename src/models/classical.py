from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import AudioPreprocessor, load_dataframe, split_dataframe
from src.evaluation.metrics import compute_metrics
from src.features.extractor import FeatureExtractor

# ---------------------------------------------------------------------------
# Per-process worker state — set once by _worker_init, reused for every task.
# Module-level globals are required for ProcessPoolExecutor on Windows (spawn).
# ---------------------------------------------------------------------------
_wp: AudioPreprocessor = None   # type: ignore[assignment]
_we: FeatureExtractor = None    # type: ignore[assignment]
_wl: LabelEncoder = None        # type: ignore[assignment]
_wcache_dir: Optional[str] = None


def _worker_init(
    prep_dict: dict,
    feat_dict: dict,
    label_classes: List[str],
    cache_dir: Optional[str] = None,
) -> None:
    """Initializer called once per worker process.

    Reconstructs AudioPreprocessor and FeatureExtractor from plain dicts
    (picklable) so they are ready for every _extract_one call in this process.
    cache_dir: if set, workers read/write per-sample .npy cache files.
    """
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    global _wp, _we, _wl, _wcache_dir

    prep_cfg = OmegaConf.create(prep_dict)
    feat_cfg = OmegaConf.create(feat_dict)

    _wp = AudioPreprocessor(prep_cfg)
    _we = FeatureExtractor(feat_cfg, prep_cfg)
    _wl = LabelEncoder()
    _wl.classes_ = np.array(label_classes)
    _wcache_dir = cache_dir


def _extract_one(args: Tuple[str, str]) -> Tuple[np.ndarray, int]:
    """Worker: load one audio file and return (feature_vector, label_int).

    Checks the on-disk cache first; writes to cache after computing.
    """
    from src.features.cache import worker_get_numpy, worker_save_numpy

    path, emotion = args
    label = int(_wl.transform([emotion])[0])

    # Cache hit
    cached = worker_get_numpy(path, _wcache_dir)
    if cached is not None:
        return cached, label

    # Compute
    waveform = _wp.load(path)
    feat = _we.extract(waveform).numpy()

    # Cache write
    worker_save_numpy(path, feat, _wcache_dir)

    return feat, label


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_split(
    df: pd.DataFrame,
    preprocessor: AudioPreprocessor,
    extractor: FeatureExtractor,
    label_encoder,
    split_name: str = "split",
    log_every: int = 100,
    n_workers: int = 1,
    cache=None,  # Optional[FeatureCache]
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract flat features for all rows in a DataFrame.

    Args:
        df:          Rows to process.
        preprocessor: AudioPreprocessor instance.
        extractor:   FeatureExtractor instance.
        label_encoder: Fitted LabelEncoder.
        split_name:  Label shown in the progress bar (e.g. "train").
        log_every:   Print a debug line every N samples (0 = disabled).
        n_workers:   Number of worker processes (1 = sequential).
        cache:       Optional FeatureCache; used in sequential path and passed
                     to workers via cache_dir string.
    """
    t_start = time.time()
    args = list(zip(df["path"].tolist(), df["emotion"].tolist()))
    cache_dir = str(cache.cache_dir) if cache is not None else None

    if n_workers > 1:
        prep_dict = {
            "sample_rate": preprocessor.sample_rate,
            "duration": preprocessor.duration,
            "normalize": preprocessor.normalize,
        }
        feat_dict = OmegaConf.to_container(extractor.cfg, resolve=True)
        label_classes = list(label_encoder.classes_)

        results = [None] * len(args)
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(prep_dict, feat_dict, label_classes, cache_dir),
        ) as executor:
            future_to_idx = {executor.submit(_extract_one, arg): i for i, arg in enumerate(args)}
            with tqdm(total=len(args), desc=f"  {split_name:5s}", unit="sample", dynamic_ncols=True) as bar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    results[idx] = future.result()
                    bar.update(1)

        features = [r[0] for r in results]
        labels = [r[1] for r in results]
    else:
        features, labels = [], []
        with tqdm(total=len(df), desc=f"  {split_name:5s}", unit="sample", dynamic_ncols=True) as bar:
            for i, (path, emotion) in enumerate(args):
                if cache is not None:
                    feat = cache.get_numpy(path)
                else:
                    waveform = preprocessor.load(path)
                    feat = extractor.extract(waveform).numpy()
                features.append(feat)
                labels.append(int(label_encoder.transform([emotion])[0]))
                bar.update(1)

                if log_every > 0 and (i + 1) % log_every == 0:
                    elapsed = time.time() - t_start
                    rate = (i + 1) / elapsed
                    bar.write(
                        f"    [{split_name}] {i + 1}/{len(df)} samples"
                        f"  ({rate:.1f} samples/s)"
                    )

    elapsed = time.time() - t_start
    print(f"  {split_name}: {len(df)} samples in {elapsed:.1f}s  ({len(df)/elapsed:.1f} samples/s)")
    return np.array(features), np.array(labels)


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_classical_model(
    model: Pipeline,
    label_encoder: LabelEncoder,
    pipeline_name: str,
    artifacts_dir: str = "artifacts/models",
) -> Path:
    """Persist a fitted sklearn Pipeline + LabelEncoder to disk."""
    from joblib import dump

    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{pipeline_name}.pkl"
    dump({"model": model, "label_encoder": label_encoder}, model_path)
    print(f"Model saved -> {model_path}")
    return model_path


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

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

    # Feature cache
    cache = None
    if cfg.data.get("use_feature_cache", True):
        from src.features.cache import FeatureCache
        cache_dir = cfg.data.get("feature_cache_dir", "data/processed")
        cache = FeatureCache(cache_dir, extractor, preprocessor)
        print(f"Feature cache: {cache.cache_dir}")

    n_workers = int(cfg.training.get("extraction_workers", 1))
    print(
        f"Feature extraction  [{extractor.method}]"
        f"  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}"
        f"  workers={n_workers}"
    )
    X_train, y_train = extract_split(train_df, preprocessor, extractor, label_encoder, "train", n_workers=n_workers, cache=cache)
    X_val,   y_val   = extract_split(val_df,   preprocessor, extractor, label_encoder, "val",   n_workers=n_workers, cache=cache)
    X_test,  y_test  = extract_split(test_df,  preprocessor, extractor, label_encoder, "test",  n_workers=n_workers, cache=cache)

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

    # Persist model artifact
    pipeline_name = cfg.get("pipeline_name", f"{cfg.model.type}_{extractor.method}")
    artifacts_dir = cfg.training.get("model_artifacts_dir", "artifacts/models")
    save_classical_model(model, label_encoder, pipeline_name, artifacts_dir)

    return {
        "model": model,
        "label_encoder": label_encoder,
        "train_time": train_time,
        **val_metrics,
        **test_metrics,
    }
