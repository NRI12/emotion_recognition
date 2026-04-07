"""Dataset download utility.

Completely decoupled from preprocessing and training logic.
Knows only about: URLs, file paths, and archives.

Public API:
    ensure_dataset(data_cfg)  -- call this from any entry point before training
"""
from __future__ import annotations

import os
import re
import zipfile
import tarfile
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

_GDRIVE_PATTERNS = [
    r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
    r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
    r"drive\.google\.com/uc\?.*id=([a-zA-Z0-9_-]+)",
]


def extract_gdrive_id(url: str) -> str:
    """Return the file ID embedded in any standard Google Drive share URL."""
    for pattern in _GDRIVE_PATTERNS:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    raise ValueError(
        f"Cannot extract a Google Drive file ID from URL: {url!r}\n"
        "Expected a URL like https://drive.google.com/file/d/<ID>/view"
    )


# ---------------------------------------------------------------------------
# Download backends
# ---------------------------------------------------------------------------

def _download_with_gdown(file_id: str, output_path: Path) -> None:
    """Download a Google Drive file using gdown (handles large-file warnings)."""
    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "gdown is required for Google Drive downloads. "
            "Install it with: pip install gdown"
        ) from exc

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(output_path), quiet=False, fuzzy=True)


def _download_with_requests(file_id: str, output_path: Path) -> None:
    """Fallback: direct requests download (may fail on large files with GDrive warning)."""
    import requests

    session = requests.Session()
    gdrive_url = "https://drive.google.com/uc"

    # First request: get confirmation token for large files
    resp = session.get(gdrive_url, params={"id": file_id}, stream=True)
    token = _get_confirm_token(resp)
    if token:
        resp = session.get(
            gdrive_url, params={"id": file_id, "confirm": token}, stream=True
        )

    _save_response(resp, output_path)


def _get_confirm_token(response) -> Optional[str]:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def _save_response(response, output_path: Path, chunk_size: int = 1 << 20) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as fh:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                fh.write(chunk)


# ---------------------------------------------------------------------------
# Archive extraction
# ---------------------------------------------------------------------------

def _extract_archive(archive_path: Path, extract_to: Path) -> None:
    """Extract a zip or tar archive to extract_to/."""
    extract_to.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive_path):
        print(f"Extracting zip: {archive_path} -> {extract_to}")
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_to)
    elif tarfile.is_tarfile(archive_path):
        print(f"Extracting tar: {archive_path} -> {extract_to}")
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(extract_to)
    else:
        raise RuntimeError(
            f"Downloaded file {archive_path} is not a recognised archive (zip/tar). "
            "If it is already extracted, set data.auto_download=false and point "
            "data.csv_path to the correct label CSV."
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ensure_dataset(data_cfg: DictConfig) -> None:
    """Ensure the dataset is available locally.

    Behaviour:
    - csv_path exists → nothing to do, return immediately.
    - csv_path missing AND auto_download=false → raise a clear error.
    - csv_path missing AND auto_download=true → download, extract, verify.

    This function is intentionally side-effect-free with respect to the rest
    of the pipeline: it only touches the filesystem under raw_dir and the
    configured data directories.
    """
    csv_path = Path(data_cfg.csv_path)
    if csv_path.exists():
        return  # already have the data

    auto_download: bool = data_cfg.get("auto_download", False)
    if not auto_download:
        raise FileNotFoundError(
            f"Dataset CSV not found at '{csv_path}'.\n"
            "Either:\n"
            "  1. Point data.csv_path to your existing dataset, or\n"
            "  2. Enable automatic download with data.auto_download=true\n"
            "     (requires data.download_url to be set in configs/data/default.yaml)"
        )

    download_url: str = data_cfg.get("download_url", "")
    if not download_url:
        raise ValueError(
            "data.auto_download=true but data.download_url is not set. "
            "Add the Google Drive URL to configs/data/default.yaml."
        )

    raw_dir = Path(data_cfg.get("raw_dir", "data/raw"))
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Derive archive name from file ID so re-runs reuse the cached file
    file_id = extract_gdrive_id(download_url)
    archive_path = raw_dir / f"{file_id}.zip"

    if not archive_path.exists():
        print(f"Downloading dataset from Google Drive (id={file_id}) ...")
        try:
            _download_with_gdown(file_id, archive_path)
        except ImportError:
            print("gdown not available, falling back to requests ...")
            _download_with_requests(file_id, archive_path)
    else:
        print(f"Archive already cached at {archive_path}, skipping download.")

    extract_to = Path(data_cfg.get("extract_to", "data"))
    _extract_archive(archive_path, extract_to)

    # Verify the CSV is now reachable
    if not csv_path.exists():
        raise RuntimeError(
            f"Download and extraction completed but '{csv_path}' still not found.\n"
            "The archive may use a different internal directory structure.\n"
            f"Archive extracted to: {extract_to}\n"
            "Check the extracted layout and update data.csv_path / data.audio_dir accordingly."
        )

    print(f"Dataset ready: {csv_path}")
