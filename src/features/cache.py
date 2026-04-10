"""Feature caches.

FeatureCache   — on-disk (.npy per sample).  Fast after first epoch.
                 Use for spectral features (MFCC, logmel, melspec).

InMemoryCache  — RAM only, no disk writes.  Ideal for SSL models (HuBERT /
                 WavLM) whose temporal embeddings are too large to store on
                 disk but benefit from being extracted once and reused across
                 epochs.  Stores float16 internally to halve RAM usage;
                 returns float32 to callers.
                 Requires num_workers=0 (dict lives in main process).

Both share the same public interface:
    cache.get(audio_path)        → torch.Tensor
    cache.get_numpy(audio_path)  → np.ndarray
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
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

    def warm(self, paths: list, n_workers: int = 1) -> None:
        """Pre-extract and cache features for every path in the list.

        After all entries are cached the SSL model (if any) is offloaded from
        memory — CNN / LSTM training reads only from .npy files from this point.

        Parameters
        ----------
        paths     : list of audio file paths
        n_workers : number of parallel threads for extraction
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm

        missing = [p for p in paths if not self._cache_path(p).exists()]
        if not missing:
            print(f"[cache] All {len(paths)} entries already cached — skipping extraction.")
        else:
            print(f"[cache] Warming {len(missing)}/{len(paths)} missing entries "
                  f"({n_workers} worker(s)) ...")
            def _do(path: str) -> None:
                self.get_numpy(path)

            with tqdm(total=len(missing), unit="sample", dynamic_ncols=True) as bar:
                if n_workers > 1:
                    with ThreadPoolExecutor(max_workers=n_workers) as pool:
                        futures = {pool.submit(_do, p): p for p in missing}
                        for _ in as_completed(futures):
                            bar.update(1)
                else:
                    for p in missing:
                        _do(p)
                        bar.update(1)

        # Offload SSL model — no longer needed once cache is warm
        if hasattr(self._extractor, "_transform") and hasattr(
            self._extractor._transform, "offload"
        ):
            self._extractor._transform.offload()

    @property
    def cache_dir(self) -> Path:
        return self._dir

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cache_path(self, audio_path: str) -> Path:
        return self._dir / f"{_path_key(audio_path)}.npy"


# ---------------------------------------------------------------------------
# In-memory cache  (SSL models: no disk writes)
# ---------------------------------------------------------------------------

class InMemoryCache:
    """RAM-only feature cache for SSL extractors (HuBERT / WavLM).

    Stores embeddings as float16 to halve memory usage; returns float32.

    Memory estimate:
        HuBERT/WavLM base, 3 s audio, 50 fps:
        (1, 768, 150) × 2 bytes ≈ 230 KB / sample
        31 K samples             ≈ 7 GB RAM total

    Thread safety:
        Uses a per-path lock so concurrent callers for the *same* path
        don't double-extract.  Requires num_workers=0 in DataLoader
        (subprocess workers get isolated copies of this dict).
    """

    def __init__(self, extractor, preprocessor) -> None:
        self._extractor    = extractor
        self._preprocessor = preprocessor
        self._store: dict  = {}          # path → float16 ndarray
        self._lock         = threading.Lock()
        self._path_locks: dict = {}      # path → Lock (fine-grained)

    # ------------------------------------------------------------------
    # Public API  (same interface as FeatureCache)
    # ------------------------------------------------------------------

    def get(self, audio_path: str) -> torch.Tensor:
        return torch.from_numpy(self.get_numpy(audio_path))

    def get_numpy(self, audio_path: str) -> np.ndarray:
        if audio_path in self._store:
            return self._store[audio_path].astype(np.float32)

        # Ensure only one thread extracts each path
        with self._lock:
            if audio_path not in self._path_locks:
                self._path_locks[audio_path] = threading.Lock()
            path_lock = self._path_locks[audio_path]

        with path_lock:
            if audio_path in self._store:          # double-check
                return self._store[audio_path].astype(np.float32)
            waveform = self._preprocessor.load(audio_path)
            feat_f32 = self._extractor.extract(waveform).numpy()
            feat_f16 = feat_f32.astype(np.float16)  # halve RAM
            self._store[audio_path] = feat_f16
            return feat_f32

    @property
    def size(self) -> int:
        """Number of cached entries."""
        return len(self._store)

    @property
    def memory_mb(self) -> float:
        """Approximate RAM used in MB."""
        return sum(v.nbytes for v in self._store.values()) / 1024 ** 2
