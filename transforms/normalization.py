import numpy as np
import json
import logging
from pathlib import Path

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
        if dtype not in [
            pl.Float32,
            pl.Float64,
            pl.Int32,
            pl.Int64,
            pl.UInt32,
            pl.UInt64,
        ]:
            logger.warning(f"Skipping non-numeric column: {col} (type: {dtype})")
            continue

        col_stats = df.select(
            [
                pl.col(col).min().alias("min"),
                pl.col(col).max().alias("max"),
                pl.col(col).mean().alias("mean"),
                pl.col(col).std().alias("std"),
            ]
        ).row(0)

        min_val, max_val, mean_val, std_val = col_stats

        # Handle null values
        if min_val is None or max_val is None:
            logger.warning(f"Column {col} has all null values, skipping")
            continue

        stats[col] = {
            "min": float(min_val),
            "max": float(max_val),
            "mean": float(mean_val),
            "std": float(std_val),
        }

    for col in ["red_coef_trace_0.9", "white_coef_trace_0.9", "blue_coef_trace_0.9"]:
        if col in df.columns:
            stats[col] = {
                "min": 0.0,
                "max": 1.0,
                "mean": 0.0,
                "std": 1.0,
            }

    if "cls_token" in df.columns:
        cls_tokens = np.stack(df["cls_token"].to_numpy())
        stats["cls_token"] = {
            "min": np.min(cls_tokens, axis=0).tolist(),
            "max": np.max(cls_tokens, axis=0).tolist(),
            "mean": np.mean(cls_tokens, axis=0).tolist(),
            "std": np.std(cls_tokens, axis=0).tolist(),
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
