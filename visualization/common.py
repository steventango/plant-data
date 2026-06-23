"""Shared data-loading and CLI plumbing for all visualization scripts."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from config import VERSION

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("results")


def default_parquet(version: str = VERSION) -> str:
    """Return the default mixed-parquet path for *version*."""
    return f"/data/plant-rl/offline/{version}/mixed-{version}.parquet"


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def prepare_out(path: Path | str) -> Path:
    """Create parent directories for *path* and return it as a Path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging() -> None:
    """Configure root logger (INFO level, simple format)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Parquet loading
# ---------------------------------------------------------------------------
def read_parquet(path: Path | str, columns: list[str] | None = None) -> pl.DataFrame:
    """Log and read a parquet file, optionally selecting *columns*."""
    path = Path(path)
    logging.info(f"Reading parquet: {path}")
    return pl.read_parquet(path, columns=columns)
