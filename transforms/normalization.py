import json
import logging
from pathlib import Path
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "wall_time",
    "clean_area",
    "clean_convex_hull_area",
    "clean_solidity",
    "clean_perimeter",
    "clean_width",
    "clean_height",
    "clean_longest_path",
    "clean_center_of_mass_x",
    "clean_center_of_mass_y",
    "clean_convex_hull_vertices",
    "clean_ellipse_center_x",
    "clean_ellipse_center_y",
    "clean_ellipse_major_axis",
    "clean_ellipse_minor_axis",
    "clean_ellipse_angle",
    "clean_ellipse_eccentricity",
]


def get_feature_columns(df: pl.DataFrame) -> list[str]:
    """Get all feature columns present in the DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of column names that are feature columns
    """
    feature_cols = []
    
    # Add explicitly named feature columns
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            feature_cols.append(col)

    return feature_cols


def compute_normalization_stats(df: pl.DataFrame) -> dict:
    """Compute min and max values for all feature columns.
    
    Args:
        df: Input DataFrame with feature columns
        
    Returns:
        Dictionary mapping column names to {"min": value, "max": value}
    """
    feature_cols = get_feature_columns(df)
    stats = {}
    
    for col in feature_cols:
        if col not in df.columns:
            continue
            
        # Skip non-numeric columns
        dtype = df.schema[col]
        if dtype not in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]:
            logger.warning(f"Skipping non-numeric column: {col} (type: {dtype})")
            continue
        
        col_stats = df.select([
            pl.col(col).min().alias("min"),
            pl.col(col).max().alias("max"),
        ]).row(0)
        
        min_val, max_val = col_stats
        
        # Handle null values
        if min_val is None or max_val is None:
            logger.warning(f"Column {col} has all null values, skipping")
            continue
            
        stats[col] = {
            "min": float(min_val),
            "max": float(max_val),
        }
        
    logger.info(f"Computed normalization stats for {len(stats)} columns")
    return stats


def save_normalization_stats(stats: dict, output_path: Path) -> None:
    """Save normalization statistics to a JSON file.
    
    Args:
        stats: Dictionary of column stats from compute_normalization_stats
        output_path: Path to save the JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved normalization stats to {output_path}")


def load_normalization_stats(stats_path: Path) -> dict:
    """Load normalization statistics from a JSON file.
    
    Args:
        stats_path: Path to the JSON file
        
    Returns:
        Dictionary of column stats
    """
    with open(stats_path, "r") as f:
        return json.load(f)


def normalize_dataframe(
    df: pl.DataFrame, 
    stats: Optional[dict] = None,
    stats_path: Optional[Path] = None,
) -> pl.DataFrame:
    """Apply min-max normalization to feature columns.
    
    Args:
        df: Input DataFrame
        stats: Dictionary of normalization stats (if None, will load from stats_path)
        stats_path: Path to load stats from (if stats is None)
        
    Returns:
        DataFrame with normalized feature columns
    """
    if stats is None:
        if stats_path is None:
            raise ValueError("Either stats or stats_path must be provided")
        stats = load_normalization_stats(stats_path)
    
    # Apply normalization to each column
    for col, col_stats in stats.items():
        if col not in df.columns:
            continue
            
        min_val = col_stats["min"]
        max_val = col_stats["max"]
        
        # Avoid division by zero
        if max_val == min_val:
            logger.warning(f"Column {col} has constant value {min_val}, setting to 0")
            df = df.with_columns(pl.lit(0.0).alias(col))
        else:
            df = df.with_columns(
                ((pl.col(col) - min_val) / (max_val - min_val)).alias(col)
            )
    
    return df