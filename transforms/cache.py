import logging
from pathlib import Path
from typing import Optional
import polars as pl

logger = logging.getLogger(__name__)


class DiskCache:
    def __init__(self, base_dir: Path, version: str):
        self.cache_dir = base_dir / "cache" / version
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized DiskCache at {self.cache_dir}")

    def get_path(self, experiment: int, zone: int, num_images: int) -> Path:
        return self.cache_dir / f"E{experiment}_Z{zone:02d}_n{num_images}_cache.parquet"

    def load(
        self, experiment: int, zone: int, num_images: int
    ) -> Optional[pl.DataFrame]:
        """Load cached results for a zone as a DataFrame."""
        path = self.get_path(experiment, zone, num_images)
        if not path.exists():
            return None

        try:
            df = pl.read_parquet(path)
            logger.info(f"Loaded {len(df)} cached results from {path}")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache from {path}: {e}")
            return None

    def save(
        self,
        experiment: int,
        zone: int,
        data: pl.DataFrame,
        num_images: int,
    ) -> bool:
        """Save results to cache."""
        if data.is_empty():
            return False

        path = self.get_path(experiment, zone, num_images)
        try:
            data.write_parquet(path)
            logger.info(f"Saved {len(data)} results to cache at {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save cache to {path}: {e}")
            return False
