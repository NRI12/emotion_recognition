from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from tqdm import tqdm
from omegaconf import DictConfig

from src.data.dataset import AudioPreprocessor, load_dataframe
from src.features.extractor import FeatureExtractor


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
    label_encoder: LabelEncoder,
    split_name: str = "split",
    log_every: int = 200,
    n_workers: int = 1,
    cache=None,  # Optional[FeatureCache]
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract flat features for all rows in a DataFrame.

    Uses ThreadPoolExecutor for parallel extraction when n_workers > 1.
    Threads are safe here because:
      - soundfile.read() releases the GIL (C extension)
      - torchaudio transforms release the GIL (PyTorch C++ backend)
      - numpy operations release the GIL (C extension)
      - LabelEncoder.transform() is read-only after fit (stateless)
      - FeatureCache uses atomic file writes (process + thread safe)

    This avoids the fork/spawn issues that plague ProcessPoolExecutor
    when PyTorch's C++ thread pool is already initialised in the parent.
    """
    t_start = time.time()
    rows: List[Tuple[str, str]] = list(
        zip(df["path"].tolist(), df["emotion"].tolist())
    )
    n = len(rows)

    def _do_one(path: str, emotion: str) -> Tuple[np.ndarray, int]:
        """Extract one sample (thread-safe, runs in worker thread)."""
        if cache is not None:
            feat = cache.get_numpy(path)
        else:
            waveform = preprocessor.load(path)
            feat = extractor.extract(waveform).numpy()
        label = int(label_encoder.transform([emotion])[0])
        return feat, label

    features: List[Optional[np.ndarray]] = [None] * n
    labels:   List[Optional[int]]        = [None] * n

    n_threads = min(max(n_workers, 1), n)

    with tqdm(total=n, desc=f"  {split_name:5s}", unit="sample", dynamic_ncols=True) as bar:
        if n_threads > 1:
            with ThreadPoolExecutor(max_workers=n_threads) as pool:
                future_to_idx = {
                    pool.submit(_do_one, path, emotion): i
                    for i, (path, emotion) in enumerate(rows)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    features[idx], labels[idx] = future.result()
                    bar.update(1)
        else:
            for i, (path, emotion) in enumerate(rows):
                features[i], labels[i] = _do_one(path, emotion)
                bar.update(1)
                if log_every > 0 and (i + 1) % log_every == 0:
                    elapsed = time.time() - t_start
                    bar.write(
                        f"    [{split_name}] {i + 1}/{n}"
                        f"  ({(i + 1) / elapsed:.1f} samples/s)"
                    )

    elapsed = time.time() - t_start
    rate = n / elapsed if elapsed > 0 else 0
    print(f"  {split_name}: {n} samples in {elapsed:.1f}s  ({rate:.1f} samples/s)")
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
