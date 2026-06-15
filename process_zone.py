import argparse
import json
import logging
import re
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from config import GOOD_ZONE_DAYS, TIMEZONE, VERSION
from transforms import (
    transform_action,
    transform_action_traces,
    transform_experiment_attributes,
    transform_image_embeddings,
    transform_reward,
    transform_state,
    transform_terminal,
)

logging.basicConfig(level=logging.INFO)


def process_zone(data_path, output_path, exp_id, zone_id, good_days, subsampling="daily"):
    logging.info(f"Processing Experiment {exp_id}, Zone {zone_id} from {data_path}")

    # Check if raw.csv exists in the data_path
    raw_csv_paths = list(Path(data_path).glob("raw*.csv"))
    raw_csv_path = Path(data_path) / "raw.csv"
    if raw_csv_path.exists():
        raw_csv_paths.append(raw_csv_path)
    config_path = Path(data_path) / "config.json"
    local_tz = TIMEZONE
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
            local_tz = config.get("timezone", TIMEZONE)

    local_tzinfo = ZoneInfo(local_tz)
    logging.info(f"Using timezone {local_tz}")

    dfs = [
        pl.read_csv(
            path,
            try_parse_dates=True,
            infer_schema_length=10000,
            columns=[
                "time",
                "image_name",
                "action.0",
                "action.1",
                "action.2",
                "action.3",
                "action.4",
                "action.5",
            ],
        )
        for path in raw_csv_paths
    ]
    df = pl.concat(dfs, how="diagonal_relaxed")
    df = df.unique(subset=["time"])
    df = df.with_columns(pl.col("time").dt.convert_time_zone(local_tz))
    df = df.with_columns(
        pl.col("time").dt.replace(second=0, microsecond=0, ambiguous="earliest")
    )
    assert df.filter((pl.col("time").dt.minute() % 5 != 0)).is_empty()
    # fill in missing time steps
    min_time: datetime = df["time"].min()  # type: ignore
    max_time: datetime = df["time"].max()  # type: ignore
    all_times = pl.datetime_range(min_time, max_time, interval="5m", eager=True)
    times_df = pl.DataFrame(data={"time": all_times})
    df = times_df.join(df, on=["time"], how="left")
    df = df.sort("time")
    df = transform_experiment_attributes(df, exp_id, zone_id)
    df = df.filter(
        pl.col("time")
        .dt.time()
        .is_between(time(9, tzinfo=local_tzinfo), time(21, tzinfo=local_tzinfo))
    )
    df = df.sort("time")
    print(
        f"E{exp_id}/zone{zone_id}: missing {df['action.0'].is_null().sum()} time steps"
    )

    df = df.with_columns(
        ((pl.col("time").dt.date() - df["time"].dt.date().min()).dt.total_days()).alias(
            "day"
        ),
    )

    df = df.with_columns(
        pl.col("image_name").fill_null(strategy="forward"),
        pl.col("action.0").fill_null(strategy="forward"),
        pl.col("action.1").fill_null(strategy="forward"),
        pl.col("action.2").fill_null(strategy="forward"),
        pl.col("action.3").fill_null(strategy="forward"),
        pl.col("action.4").fill_null(strategy="forward"),
        pl.col("action.5").fill_null(strategy="forward"),
    )
    df = transform_action(df)
    if exp_id == 18:
        df = df.filter(
            (pl.col("time").dt.time() >= time(9, 0, tzinfo=local_tzinfo))
            & (pl.col("time").dt.time() <= time(21, 0, tzinfo=local_tzinfo))
        )
    else:
        df = df.filter(
            (pl.col("time").dt.time() >= time(9, 30, tzinfo=local_tzinfo))
            & (pl.col("time").dt.time() < time(20, 30, tzinfo=local_tzinfo))
        )
    # transform action by averaging over the day
    df = df.with_columns(
        pl.col("red_coef").mean().over("experiment", "zone", "day").alias("red_coef"),
        pl.col("white_coef")
        .mean()
        .over("experiment", "zone", "day")
        .alias("white_coef"),
        pl.col("blue_coef").mean().over("experiment", "zone", "day").alias("blue_coef"),
    )
    if subsampling == "daily":
        # daily subsampling
        df = df.filter(pl.col("time").dt.time() == time(9, 30, tzinfo=local_tzinfo))
    elif subsampling == "hourly":
        # hourly subsampling
        df = df.filter(pl.col("time").dt.minute() == 30)
    elif subsampling == "4hourly":
        if exp_id == 18:
            # For E18, daytime is 9:00 to 21:00, but 21:00 is dark.
            # So sample at 9:00, 13:00, 17:00 (minutes = 00)
            df = df.filter(
                (pl.col("time").dt.minute() == 0) &
                (pl.col("time").dt.hour().is_in([9, 13, 17]))
            )
        else:
            # 4-hourly subsampling (e.g. 9:30, 13:30, 17:30)
            df = df.filter(
                (pl.col("time").dt.minute() == 30) &
                (pl.col("time").dt.hour().is_in([9, 13, 17]))
            )
    else:
        raise ValueError(f"Unknown subsampling: {subsampling}")

    df = transform_action_traces(df)

    # Pass output directory for image embeddings (same directory as parquet output)
    output_dir = Path(output_path).parent
    df = transform_image_embeddings(df, output_dir=output_dir)

    df = transform_state(df)
    df = transform_terminal(df, 13)
    df = transform_reward(df)

    df = df.with_columns(pl.col("day").is_in(good_days).alias("is_good_day"))
    if exp_id == 12:
        # experiment 12: day 9 has is filled with invalid observation
        df = df.filter(pl.col("wall_time") < 9)
    df = df.with_columns(
        pl.col("time")
        .shift(-1)
        .over("experiment", "zone", "plant_id")
        .alias("next_time"),
        pl.col("time")
        .shift(-2)
        .over("experiment", "zone", "plant_id")
        .alias("next_next_time"),
    )
    if subsampling == "daily":
        expected_next = pl.col("time") + pl.duration(days=1)
    elif subsampling == "4hourly":
        is_overnight = (pl.col("time").dt.hour() == 17)
        expected_next = pl.when(is_overnight).then(pl.col("time") + pl.duration(hours=16)).otherwise(pl.col("time") + pl.duration(hours=4))
    else:
        expected_next = pl.col("time") + pl.duration(hours=1)

    df = df.with_columns(
        (
            (
                pl.col("next_next_time").is_null()
                & ~pl.col("terminal")
                & (pl.col("wall_time") < 13)
            )
            | (
                (pl.col("next_time") != expected_next)
                & pl.col("next_time").is_not_null()
            )
        ).alias("truncated")
    )
    df = df.drop("next_time", "next_next_time")
    pl.Config.set_tbl_rows(20)
    print(
        df.filter(pl.col("plant_id") == pl.col("plant_id").first()).select(
            "time",
            "wall_time",
            "clean_area",
            "red_coef",
            "white_coef",
            "blue_coef",
            "reward",
            "terminal",
            "truncated",
        )
    )
    print(
        df.select(
            "time", "mean_clean_area", "red_coef", "white_coef", "blue_coef", "reward"
        ).describe()
    )
    print(f"E{exp_id}/zone{zone_id}: day min {df['day'].min()}, max {df['day'].max()}")

    # Save intermediate result
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    logging.info(f"Saved intermediate file to {output_path}")
    return df


def infer_experiment_zone(path_str):
    # Regex to extract E{exp_id} and zone{zone_id} or alliance-zone{zone_id}
    # Example path: /data/plant-rl/online/E14/P1/Constant1/alliance-zone01/

    exp_match = re.search(r"E(\d+)", path_str)
    zone_match = re.search(r"zone(\d+)", path_str)

    if exp_match and zone_match:
        return int(exp_match.group(1)), int(zone_match.group(1))
    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Process plant data for a specific zone."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the directory containing raw.csv",
    )
    parser.add_argument(
        "--subsampling",
        choices=["daily", "hourly", "4hourly"],
        default="daily",
        help="Time-of-day sampling cadence",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional suffix on output parquet filename (e.g. '_hourly')",
    )
    args = parser.parse_args()

    data_path = args.data_path.rstrip("/")
    exp_id, zone_id = infer_experiment_zone(data_path)

    if exp_id is None or zone_id is None:
        logging.error(f"Could not infer experiment and zone ID from path: {data_path}")
        return

    logging.info(f"Inferred Experiment: {exp_id}, Zone: {zone_id} from path")

    # Determine good days
    key_candidates = [f"E{exp_id}/zone{zone_id}", f"E{exp_id}/zone{zone_id:02}"]
    good_days = None
    for k in key_candidates:
        if k in GOOD_ZONE_DAYS:
            good_days = GOOD_ZONE_DAYS[k]
            break

    if good_days is None:
        logging.warning(
            f"Experiment {exp_id} Zone {zone_id} not found in GOOD_ZONE_DAYS. Using empty good_days list."
        )
        good_days = []

    # Construct output path
    output_dir = Path(data_path) / "processed" / VERSION
    output_path = output_dir / f"E{exp_id}_Z{zone_id}{args.output_suffix}.parquet"

    process_zone(
        data_path,
        str(output_path),
        exp_id,
        zone_id,
        good_days,
        subsampling=args.subsampling,
    )


if __name__ == "__main__":
    main()
