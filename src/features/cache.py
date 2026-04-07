"""On-disk feature cache.

Maps audio_path → extracted feature array, keyed by a short hash of the
feature + preprocessing configuration. Stale cache entries are bypassed
automatically when the config changes (different hash → different folder).

Cache layout:
    <cache_root>/<method>_<cfg_hash>/<md5(audio_path)>.npy

Using .npy (numpy) instead of .pt (torch):
  - No CUDA/torch initialisation in spawned worker processes
  - Read by both classical (numpy) and DL (torch.from_numpy) paths
  - Each file is uniquely named → workers write in parallel without locks

Public API:
    cache = FeatureCache(cache_root, extractor, preprocessor)
    tensor  = cache.get(audio_path)   # torch.Tensor, for DL path
    ndarray = cache.get_numpy(audio_path)   # np.ndarray, for classical path
    params  = cache.worker_params()   # picklable dict for ProcessPoolExecutor
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_hash(feat_cfg: DictConfig, prep_cfg: DictConfig) -> str:
    """8-char MD5 of the combined feature + preprocessing config."""
    payload = json.dumps(
        {
            "feat": OmegaConf.to_container(feat_cfg, resolve=True),
            "prep": OmegaConf.to_container(prep_cfg, resolve=True),
        },
        sort_keys=True,
    )
    return hashlib.md5(payload.encode()).hexdigest()[:8]


def _path_key(audio_path: str) -> str:
    """MD5 of the absolute audio path → unique filename."""
    return hashlib.md5(os.path.abspath(audio_path).encode()).hexdigest()


# ---------------------------------------------------------------------------
# FeatureCache
# ---------------------------------------------------------------------------

class FeatureCache:
    """Transparent disk cache wrapping FeatureExtractor + AudioPreprocessor.

    Thread/process safe: each sample maps to a unique file, so concurrent
    workers never contend on the same path.
    """

    def __init__(
        self,
        cache_root: str | Path,
        extractor,       # FeatureExtractor  (avoid circular import at module level)
        preprocessor,    # AudioPreprocessor
    ) -> None:
        self._extractor = extractor
        self._preprocessor = preprocessor

        prep_cfg = OmegaConf.create({
            "sample_rate": preprocessor.sample_rate,
            "duration": preprocessor.duration,
            "normalize": preprocessor.normalize,
        })
        cfg_hash = _config_hash(extractor.cfg, prep_cfg)
        self._dir = Path(cache_root) / f"{extractor.method}_{cfg_hash}"
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, audio_path: str) -> torch.Tensor:
        """Return cached features as a torch.Tensor (compute + save if missing)."""
        arr = self.get_numpy(audio_path)
        return torch.from_numpy(arr)

    def get_numpy(self, audio_path: str) -> np.ndarray:
        """Return cached features as a numpy array (compute + save if missing)."""
        cache_file = self._cache_path(audio_path)
        if cache_file.exists():
            return np.load(cache_file)
        waveform = self._preprocessor.load(audio_path)
        features = self._extractor.extract(waveform).numpy()
        # Write atomically: temp file + rename avoids partial reads in workers
        tmp = cache_file.with_suffix(f".tmp{os.getpid()}")
        np.save(tmp, features)
        try:
            tmp.rename(cache_file)
        except OSError:
            tmp.unlink(missing_ok=True)  # another worker already wrote it
        return features

    def worker_params(self) -> dict:
        """Picklable dict passed to ProcessPoolExecutor worker initializer."""
        return {"cache_dir": str(self._dir)}

    @property
    def cache_dir(self) -> Path:
        return self._dir

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cache_path(self, audio_path: str) -> Path:
        return self._dir / f"{_path_key(audio_path)}.npy"


# ---------------------------------------------------------------------------
# Standalone worker helper (used by classical.py multiprocessing workers)
# ---------------------------------------------------------------------------

def worker_get_numpy(path: str, cache_dir: Optional[str]) -> Optional[np.ndarray]:
    """Load a cached feature array inside a worker process.

    Returns the array if the cache file exists, else None (caller must compute).
    Does NOT compute or save — keeps worker logic in classical.py.
    """
    if cache_dir is None:
        return None
    cache_file = Path(cache_dir) / f"{_path_key(path)}.npy"
    if cache_file.exists():
        return np.load(cache_file)
    return None


def worker_save_numpy(path: str, arr: np.ndarray, cache_dir: Optional[str]) -> None:
    """Save a feature array from inside a worker process (atomic write)."""
    if cache_dir is None:
        return
    cache_dir_p = Path(cache_dir)
    cache_file = cache_dir_p / f"{_path_key(path)}.npy"
    if cache_file.exists():
        return  # already written by another worker
    tmp = cache_dir_p / f"{_path_key(path)}.tmp{os.getpid()}"
    np.save(tmp, arr)
    try:
        tmp.rename(cache_file)
    except OSError:
        tmp.unlink(missing_ok=True)
