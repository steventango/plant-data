import argparse
import logging
import math
import re
from pathlib import Path

import polars as pl

from config import VERSION, TIMEZONE
from transforms import (
    transform_action_traces,
    transform_outlier_detection,
    transform_pca,
    transform_umap,
)
from transforms.normalization import (
    compute_normalization_stats,
    save_normalization_stats,
)
from transforms.states import transform_energy, transform_log_clean_area

logging.basicConfig(level=logging.INFO)


def _parse_exp_zone(path: Path) -> tuple[int, int] | tuple[None, None]:
    """Extract (experiment, zone) from a processed parquet filename like E18_Z11.parquet."""
    m = re.match(r"E(\d+)_Z(\d+)", path.stem)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def subsample_daily(df: pl.DataFrame) -> pl.DataFrame:
    """
    Recompute daily-cadence derived columns after the per-file 9:00-local filter.

    The processed parquets carry 4-hourly energy and reward values; after
    restricting to the single 9:00-local row per day we need to:
      - recompute energy  as cumulative_energy diff over the daily grid
      - recompute log_clean_area
      - recompute reward  as Δ log_clean_area over experiment/zone/plant_id
      - recompute action traces at the daily cadence
      - recompute outlier flags
      - recompute truncated flag (next row not exactly +1 day)
    """
    df = df.sort("experiment", "zone", "plant_id", "time")

    # Per-day energy: difference of cumulative_energy on the daily grid
    df = transform_energy(df)

    # Null out energy when the next sample is not exactly +1 day (cross-gap interval).
    # imaging_gap anchor rows are still present here so that day1's next is day2_ghost
    # (1-day diff) rather than day3 (2-day diff), giving the correct energy for day1.
    if "energy" in df.columns:
        df = df.with_columns(
            pl.col("time")
            .shift(-1)
            .over("experiment", "zone", "plant_id")
            .alias("_next_time"),
        )
        df = df.with_columns(
            pl.when(
                pl.col("_next_time").is_null()
                | (pl.col("_next_time") - pl.col("time") != pl.duration(days=1))
            )
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col("energy"))
            .alias("energy"),
        )
        df = df.drop("_next_time")

    # Drop imaging-gap anchor rows: they served their purpose for energy anchoring.
    if "imaging_gap" in df.columns:
        df = df.filter(~pl.col("imaging_gap")).drop("imaging_gap")

    # log_clean_area
    df = transform_log_clean_area(df)

    # reward = Δ log_clean_area per plant trajectory
    df = df.with_columns(
        pl.col("log_clean_area")
        .shift(1)
        .over("experiment", "zone", "plant_id")
        .alias("prev_log_clean_area"),
        pl.col("time")
        .shift(1)
        .over("experiment", "zone", "plant_id")
        .alias("_prev_time"),
    )
    df = df.with_columns(
        pl.when(
            pl.col("_prev_time").is_null()
            | (pl.col("time") - pl.col("_prev_time") != pl.duration(days=1))
        )
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("log_clean_area") - pl.col("prev_log_clean_area"))
        .alias("reward"),
    )
    df = df.drop("prev_log_clean_area", "_prev_time")

    # Recompute action traces at daily cadence
    df = transform_action_traces(df)

    # Outlier detection on the new daily values
    df = transform_outlier_detection(df, q1=0.01, q2=0.99)

    # truncated: next row is not exactly +1 day
    df = df.with_columns(
        pl.col("time")
        .shift(-1)
        .over("experiment", "zone", "plant_id")
        .alias("next_time"),
    )
    df = df.with_columns(
        (
            pl.col("next_time").is_not_null()
            & (pl.col("next_time") != pl.col("time") + pl.duration(days=1))
        ).alias("truncated")
    )
    df = df.drop("next_time")

    return df


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
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="Comma-separated experiment IDs to include (e.g. '18'). Default: all.",
    )
    parser.add_argument(
        "--zones",
        type=str,
        default=None,
        help="Comma-separated zone IDs to include (e.g. '1,3,4,11'). Default: all.",
    )
    parser.add_argument(
        "--subsample",
        choices=["none", "daily"],
        default="none",
        help="Temporal subsampling to apply after joining. "
        "'daily' retains the 9:00 AM local-time row per day and recomputes "
        "energy/reward/traces/truncated for the daily cadence.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help=f"Base name for output files (default: mixed-{{VERSION}}). "
        f"Files written: <name>.parquet, normalization-stats-<name>.json.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)

    # Parse experiment/zone filters
    exp_filter: set[int] | None = None
    zone_filter: set[int] | None = None
    if args.experiments:
        exp_filter = {int(e) for e in args.experiments.split(",")}
    if args.zones:
        zone_filter = {int(z) for z in args.zones.split(",")}

    output_name = args.output_name if args.output_name else f"mixed-{VERSION}"

    # Load all intermediate zone files recursively
    logging.info(f"Searching for processed files in {root_dir}...")
    files = sorted(root_dir.rglob(f"processed/{VERSION}/*.parquet"))

    # Apply experiment/zone filter by parsing filenames
    if exp_filter is not None or zone_filter is not None:
        filtered = []
        for f in files:
            exp, zone = _parse_exp_zone(f)
            if exp is None:
                continue
            if exp_filter is not None and exp not in exp_filter:
                continue
            if zone_filter is not None and zone not in zone_filter:
                continue
            filtered.append(f)
        files = filtered

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
            [col for col in lf.columns if col.startswith("agent_state")]
            + (["patch_features"] if "patch_features" in lf.columns else [])
        )

        if args.subsample == "daily":
            # Filter to 9:00 AM local-time rows (before Edmonton tz conversion).
            # E18 local tz is Etc/GMT-2; 9:00 local = 01:00 Edmonton.
            # Filtering on local hour/minute is robust across all zones.
            lf = lf.filter(
                (pl.col("time").dt.hour() == 9) & (pl.col("time").dt.minute() == 0)
            )

        # Make image_path absolute and point to segment images
        prefix = str(file.parent)
        lf = lf.with_columns(
            (prefix + "/" + pl.col("image_path")).alias("image_path"),
            pl.col("time").dt.convert_time_zone(TIMEZONE),
        )

        lazy_frames.append(lf)

    df = (
        pl.concat(lazy_frames, how="diagonal_relaxed")
        .sort("experiment", "zone", "plant_id", "time")
        .collect(engine="streaming")
    )

    if args.subsample == "daily":
        logging.info("Recomputing daily-cadence derived columns...")
        df = subsample_daily(df)

        # Compute scalar intensity: a = sum(action channels) / 100
        # Follows plant-rl IntensityAction: commanded = BALANCED_ACTION_100 * a
        action_cols = [f"action.{i}" for i in range(6)]
        available_action_cols = [c for c in action_cols if c in df.columns]
        if available_action_cols:
            df = df.with_columns(
                (sum(pl.col(c) for c in available_action_cols) / 100.0).alias(
                    "intensity"
                )
            )
            # Drop rows where lights were off (intensity == 0 at 9:00 = bad data)
            before = df.height
            df = df.filter(pl.col("intensity") > 0)
            dropped = before - df.height
            if dropped:
                logging.warning(f"Dropped {dropped} zero-intensity rows (lights off at 9:00 AM)")

        # ----- Energy-reward columns -----
        # Reference constants derived from Zone 11 (Constant baseline)
        z11_df = df.filter(pl.col("zone") == 11)
        if z11_df.is_empty() or z11_df["energy"].drop_nulls().len() == 0:
            logging.warning("Zone 11 not in dataset; using dataset-wide energy/reward baselines")
            e_const = float(df["energy"].drop_nulls().mean())
            r_const = float(df["reward"].drop_nulls().mean())
        else:
            e_const = float(z11_df["energy"].drop_nulls().mean())
            r_const = float(z11_df["reward"].drop_nulls().mean())
        max_energy = float(df["energy"].drop_nulls().max())
        r_gate = 0.95 * r_const
        fail_penalty = 2.0 * max_energy
        log_e_const = math.log(e_const)
        logging.info(
            f"Schema constants: e_const={e_const:.1f} Wh, r_const={r_const:.5f}, "
            f"r_gate={r_gate:.5f}, max_energy={max_energy:.1f} Wh, fail_penalty={fail_penalty:.1f}"
        )

        # Rename daily Δ log_clean_area: reward → area_reward
        df = df.rename({"reward": "area_reward"})

        # Schema A: energy_reward_schema_a_β = (β / N_STEPS) · (log(energy) − log(e_const))
        # Divided by N_STEPS=14 so that summing over an episode gives the same
        # magnitude as the episode-level formula β · log(E_total / E_const_total).
        N_STEPS = 14
        for beta in [0.5, 1.0, 2.0, 4.0, 8.0]:
            col_name = f"energy_reward_schema_a_{beta:g}"
            df = df.with_columns(
                pl.when(pl.col("energy").is_not_null() & (pl.col("energy") > 0))
                .then((pl.col("energy").log() - log_e_const) * (beta / N_STEPS))
                .otherwise(None)
                .alias(col_name)
            )

        # Schema B: gate on each step's area_reward vs Zone 11 baseline
        # passed (area_reward ≥ 0.95 · r_const): pay energy; failed: pay 2 · max_energy
        df = df.with_columns(
            pl.when(pl.col("area_reward").is_null())
            .then(None)
            .when(pl.col("area_reward") >= r_gate)
            .then(pl.col("energy"))
            .otherwise(pl.lit(fail_penalty))
            .alias("energy_reward_schema_b")
        )

        # Combined RL reward: area growth minus energy cost (Schema A, β=1)
        df = df.with_columns(
            (pl.col("area_reward") - pl.col("energy_reward_schema_a_1")).alias("reward")
        )
    else:
        # Apply outlier detection on full dataset before saving (original behaviour)
        df = transform_outlier_detection(df, q1=0.01, q2=0.99)

    pl.Config.set_tbl_rows(20)
    print(
        df.filter(
            (pl.col("experiment") == df["experiment"].min())
            & (pl.col("zone") == df["zone"].min())
            & (pl.col("plant_id") == df["plant_id"].min())
        ).select(
            "wall_time", "time", "clean_area", "reward", "nodata", "outlier", "valid"
        )
    )

    df = transform_pca(df, K=10, output_path=output_dir / "pca_model.joblib")
    df = transform_umap(df, K=2, output_path=output_dir / "umap_model.joblib")

    # Save to parquet
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"{output_name}.parquet"
    logging.info(f"Saving dataset to {path}")
    df.write_parquet(path)

    stats_path = output_dir / f"normalization-stats-{output_name}.json"
    full_stats = compute_normalization_stats(df)

    # Add stats for any action columns not in COLS (e.g. intensity, energy_reward_*)
    extra_reward_cols = [f"energy_reward_schema_a_{b:g}" for b in [0.5, 1.0, 2.0, 4.0, 8.0]]
    for extra_col in ["intensity", "area_reward", "energy_reward_schema_b"] + extra_reward_cols:
        if extra_col in df.columns and extra_col not in full_stats:
            col_series = df[extra_col].drop_nulls().drop_nans()
            if col_series.len() > 0:
                full_stats[extra_col] = {
                    "min": float(col_series.min()),
                    "max": float(col_series.max()),
                    "mean": float(col_series.mean()),
                    "std": float(col_series.std()),
                }

    save_normalization_stats(full_stats, stats_path)


if __name__ == "__main__":
    main()
