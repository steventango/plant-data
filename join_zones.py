import argparse
import logging
from datetime import time
from pathlib import Path

import polars as pl

from config import VERSION, tzinfo
from transforms import (
    transform_action_traces,
    transform_outlier_detection,
    # transform_reward,
    # transform_state,
)
from transforms.normalization import (
    compute_normalization_stats,
    save_normalization_stats,
)

logging.basicConfig(level=logging.INFO)


def subsample(df: pl.DataFrame, type: str = "daily") -> pl.DataFrame:
    if type == "daily":
        # TODO: transform action by averaging over the day between 9:30 and 20:30
        # df = df.filter(
        #     (pl.col("time").dt.time() >= time(9, 30, tzinfo=tzinfo))
        #     & (pl.col("time").dt.time() < time(20, 30, tzinfo=tzinfo))
        # )
        # df = df.with_columns(
        #     pl.col("action")
        #     .mean()
        #     .over("experiment", "zone", "plant_id", "day")
        #     .alias("action"),
        # )
        df_daily = df.filter(pl.col("time").dt.time() == time(9, 30, tzinfo=tzinfo))
        df_daily = df_daily.with_columns(
            pl.col("clean_area")
            .shift(1)
            .over("experiment", "zone", "plant_id")
            .alias("prev_clean_area"),
        )
        df_daily = df_daily.with_columns(
            (
                (pl.col("clean_area") - pl.col("prev_clean_area"))
                / pl.col("prev_clean_area")
            ).alias("reward"),
        )
        df_daily = df_daily.drop("prev_clean_area")
        df_daily = transform_action_traces(df_daily)
        df_daily = transform_outlier_detection(df_daily, q1=0.01, q2=0.99)
        df_daily = df_daily.with_columns(
            pl.col("time")
            .shift(-1)
            .over("experiment", "zone", "plant_id")
            .alias("next_time"),
        )
        df_daily = df_daily.with_columns(
            (
                pl.col("next_time").is_not_null()
                & (pl.col("next_time") != pl.col("time") + pl.duration(days=1))
            ).alias("truncated")
        )
        df_daily = df_daily.drop("next_time")
        return df_daily
    else:
        raise ValueError(f"Unknown subsample type: {type}")


def main():
    parser = argparse.ArgumentParser(description="Join processed zone data.")
    parser.add_argument(
        "--root-dir",
        type=str,
        default="/data/plant-rl/online",
        help="Root directory to search for processed files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"/data/plant-rl/offline/{VERSION}/",
        help="Directory to save output files",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)

    # Load all intermediate zone files recursively
    logging.info(f"Searching for processed files in {root_dir}...")
    files = sorted(root_dir.rglob(f"processed/{VERSION}/*.parquet"))

    if not files:
        logging.error(f"No processed files found in {root_dir}")
        return

    logging.info(
        f"Found {len(files)} processed files. Concatenating with lazy loading..."
    )

    # Use lazy loading to reduce memory usage during concatenation
    lazy_frames = []
    for file in files:
        print(file)
        lf = pl.scan_parquet(file)
        # TODO: fix later drop all columns starting with agent_state
        lf = lf.drop(
            [col for col in lf.columns if col.startswith("agent_state")] + ["patch_features"] if "patch_features" in lf.columns else []
        )

        # Make image_path absolute and point to segment images
        prefix = str(file.parent)
        lf = lf.with_columns(
            (prefix + "/" + pl.col("image_path"))
            .alias("image_path")
        )

        lazy_frames.append(lf)

    df = (
        pl.concat(lazy_frames, how="diagonal_relaxed")
        .sort("experiment", "zone", "plant_id", "time")
        .collect(engine="streaming")
    )

    # Apply outlier detection on full dataset before saving
    df = transform_outlier_detection(df, q1=0.01, q2=0.99)

    df = df.with_columns(
        pl.col("time")
        .shift(-1)
        .over("experiment", "zone", "plant_id")
        .alias("next_time"),
    )
    df = df.with_columns(
        (
            (pl.col("next_time").is_null() & ~pl.col("terminal") & (pl.col("wall_time") < 13))
            | (pl.col("next_time") != pl.col("time") + pl.duration(days=1))
        ).alias("truncated")
    )
    df = df.drop("next_time")

    pl.Config.set_tbl_rows(20)
    print(df.filter((pl.col("experiment") == 11) & (pl.col("zone") == 1) & (pl.col("plant_id") == 0)).select(
        "wall_time",
        "time",
        "clean_area",
        "reward",
        "nodata",
        "outlier",
        "valid"
    ))

    # Save to parquet
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the full continuous dataset
    path = output_dir / f"mixed-{VERSION}.parquet"
    logging.info(f"Saving full dataset to {path}")
    df.write_parquet(path)

    # Compute and save normalization stats for the full dataset
    logging.info("Computing normalization statistics for full dataset...")
    full_stats = compute_normalization_stats(df)
    full_stats_path = output_dir / f"normalization-stats-{VERSION}.json"
    save_normalization_stats(full_stats, full_stats_path)


if __name__ == "__main__":
    main()
