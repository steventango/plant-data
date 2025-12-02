import logging
import argparse
from pathlib import Path
from datetime import time

import polars as pl
from config import tzinfo, VERSION
from transforms import (
    transform_action_traces,
    transform_outlier_detection,
)

logging.basicConfig(level=logging.INFO)

def subsample(df: pl.DataFrame, type: str = "daily") -> pl.DataFrame:
    if type == "daily":
        df_daily = df.filter(pl.col("time").dt.time() == time(9, 30, tzinfo=tzinfo))
        df_daily = df_daily.with_columns(
            pl.col("clean_area")
            .shift(1)
            .over("experiment", "zone", "plant_id")
            .alias("prev_clean_area"),
        )
        df_daily = df_daily.with_columns(
            (
                (pl.col("clean_area") - pl.col("prev_clean_area")) / pl.col("prev_clean_area")
            ).alias("reward"),
        )
        df_daily = df_daily.drop("prev_clean_area")
        df_daily = transform_action_traces(df_daily)
        df_daily = transform_outlier_detection(df_daily, q1=0.01, q2=0.99)
        df_daily = df_daily.with_columns(
            pl.col("time").shift(-1).over("experiment", "zone", "plant_id").alias("next_time"),
        )
        df_daily = df_daily.with_columns(
            (
                pl.col("next_time").is_not_null()
                & (pl.col("next_time") != pl.col("time") + pl.duration(days=1))
            ).alias("truncated")
        )
        df_daily = df_daily.drop("next_time")
        # TODO: is this the correct terminal?
        df_daily = df_daily.with_columns(
            pl.col("time")
            .shift(-1)
            .over("experiment", "zone", "plant_id")
            .is_null()
            .alias("terminal"),
        )
        return df_daily
    else:
        raise ValueError(f"Unknown subsample type: {type}")

def main():
    parser = argparse.ArgumentParser(description="Join processed zone data.")
    parser.add_argument("--root-dir", type=str, default="/data/plant-rl/online", help="Root directory to search for processed files")
    parser.add_argument("--output-dir", type=str, default=f"/data/plant-rl/offline/{VERSION}/", help="Directory to save output files")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)
    
    # Load all intermediate zone files recursively
    logging.info(f"Searching for processed files in {root_dir}...")
    files = list(root_dir.rglob(f"processed/{VERSION}/*.parquet"))
    
    if not files:
        logging.error(f"No processed files found in {root_dir}")
        return

    logging.info(f"Found {len(files)} processed files. Concatenating...")
    
    dfs = [pl.read_parquet(f) for f in files]
    df = pl.concat(dfs, how="diagonal_relaxed").sort(
        "experiment", "zone", "plant_id", "time"
    )
    
    print(df.head())
    
    # Create daily dataset
    df_daily = subsample(df, "daily")
    
    print(df_daily.select("reward").describe())
    print(df_daily[["time", "day", "red_coef", "white_coef", "blue_coef", "reward"]])    

    # Save to parquet
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the full continuous dataset  
    path = output_dir / f"mixed-{VERSION}.parquet"
    logging.info(f"Saving full continuous dataset to {path}")
    df.write_parquet(path)
    
    # Save the daily continuous dataset
    path = output_dir / f"mixed-daily-{VERSION}.parquet"
    logging.info(f"Saving daily continuous dataset to {path}")
    df_daily.write_parquet(path)

if __name__ == "__main__":
    main()
