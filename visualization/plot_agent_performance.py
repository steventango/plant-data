import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Plot 95% bootstrap average rewards for each experiment zone."
    )
    parser.add_argument(
        "--parquet",
        "-p",
        default="/data/plant-rl/offline/cleaned_offline_dataset_daily_continuous_v15.parquet",
        help="Path to parquet file",
    )
    parser.add_argument(
        "--out",
        "-o",
        default="results/return_plots",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--show", action="store_true", help="Show the plots interactively"
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Do not filter by good days and outliers (default: filter)",
    )
    # TODO: migrate to dataset generation
    parser.add_argument(
        "--lower-quantile",
        type=float,
        default=0.2,
        help="Lower quantile for filtering returns (default: 0.0 - no filtering)",
    )
    parser.add_argument(
        "--upper-quantile",
        type=float,
        default=0.8,
        help="Upper quantile for filtering area (default: 1.0 - no filtering)",
    )
    args = parser.parse_args()

    try:
        logging.info(f"Reading parquet: {args.parquet}")
        df = pl.read_parquet(args.parquet)
        logging.info(f"Rows after reading: {df.shape[0]}")
    except Exception as e:
        logging.error(f"Failed to read parquet: {e}")
        sys.exit(1)

    # Check if required columns exist
    required_cols = ["experiment", "zone", "plant_id", "reward", "day"]
    # Use clean_area if area is not available
    if "area" in df.columns:
        area_col = "area"
    elif "clean_area" in df.columns:
        area_col = "clean_area"
        required_cols.append("clean_area")
    else:
        logging.error("Neither 'area' nor 'clean_area' column found")
        logging.info("Available columns: %s", df.columns)
        sys.exit(1)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error("Missing required columns: %s", missing_cols)
        logging.info("Available columns: %s", df.columns)
        sys.exit(1)

    # Filter data
    if not args.no_filter:
        logging.info("Filtering data (day <= 13, is_good_day, not outlier)...")
        filter_expr = pl.col("day") <= 13

        if "is_good_day" in df.columns:
            filter_expr = filter_expr & pl.col("is_good_day")
        else:
            logging.warning("'is_good_day' column not found, skipping this filter")

        if "outlier" in df.columns:
            filter_expr = filter_expr & ~pl.col("outlier")
        else:
            logging.warning("'outlier' column not found, skipping this filter")

        df = df.filter(filter_expr)
        logging.info(f"Rows after filtering: {df.shape[0]}")

    # Filter data by area quantile if specified
    if args.lower_quantile > 0.0 or args.upper_quantile < 1.0:
        logging.info(
            f"Filtering data to keep only {args.lower_quantile}-{args.upper_quantile} area quantile range per experiment/zone/day..."
        )

        # Sort to ensure correct shifting for next-step validation
        df = df.sort(["experiment", "zone", "plant_id", "day"])

        df = df.with_columns(
            [
                pl.col(area_col)
                .quantile(args.lower_quantile)
                .over(["experiment", "zone", "day"])
                .alias("lower_q"),
                pl.col(area_col)
                .quantile(args.upper_quantile)
                .over(["experiment", "zone", "day"])
                .alias("upper_q"),
            ]
        )

        original_count = df.shape[0]

        # Mark rows as valid if they are within the quantile range
        df = df.with_columns(
            (
                (pl.col(area_col) >= pl.col("lower_q"))
                & (pl.col(area_col) <= pl.col("upper_q"))
            ).alias("is_valid_area")
        )

        # Filter: keep row if it is valid AND the next row (for same plant) is valid
        # This is because reward depends on next_area / area.
        df = df.filter(
            pl.col("is_valid_area")
            & pl.col("is_valid_area")
            .shift(-1)
            .over(["experiment", "zone", "plant_id"])
            .fill_null(True)
        ).drop(["lower_q", "upper_q", "is_valid_area"])

        logging.info(
            f"Rows after area quantile filtering: {df.shape[0]} (removed {original_count - df.shape[0]})"
        )

    # Map experiment/zone to Agent
    df = df.with_columns(
        pl.when(pl.col("experiment").is_in([11, 12]))
        .then(pl.lit("Uniform_Discrete"))
        .when(pl.col("experiment") == 13)
        .then(pl.lit("Uniform_Dirichlet"))
        .when(pl.col("experiment") == 14)
        .then(
            pl.when(pl.col("zone") == 1)
            .then(pl.lit("Constant_White"))
            .when(pl.col("zone") == 2)
            .then(pl.lit("InAC_Data_Det"))
            .when(pl.col("zone").is_in([3, 4, 5, 6]))
            .then(pl.lit("InAC_Data_Sto"))
            .when(pl.col("zone") == 8)
            .then(pl.lit("InAC_GP_Det_Opt0"))
            .when(pl.col("zone") == 9)
            .then(pl.lit("InAC_GP_Det_Opt0.25"))
            .when(pl.col("zone") == 10)
            .then(pl.lit("InAC_GP_Det_Opt0.5"))
            .when(pl.col("zone") == 11)
            .then(pl.lit("InAC_GP_Det_Opt0.75"))
            .when(pl.col("zone") == 12)
            .then(pl.lit("InAC_GP_Det_Opt1"))
            .otherwise(pl.lit("Other_E14"))
        )
        .otherwise(pl.lit("Other"))
        .alias("agent")
    )

    # Filter out "Other" agents if they exist, to keep the plot clean based on the request
    df = df.filter(~pl.col("agent").str.contains("Other"))

    logging.info(f"Number of plants (episodes): {df.shape[0]}")

    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert to pandas for seaborn
    pdf = df.to_pandas()

    sns.set(style="whitegrid")

    # Plot 1: All experiments together
    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=pdf,
        x="agent",
        y="reward",
        errorbar=("ci", 95),
        capsize=0.1,
        palette="tab10",
    )

    plt.title("Reward by Agent (95% CI)")
    plt.xlabel("Agent")
    plt.ylabel("Reward")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out_file = out_dir / "rewards_by_zone_experiment.png"
    plt.savefig(out_file, dpi=200)
    logging.info("Saved plot to %s", out_file)

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
