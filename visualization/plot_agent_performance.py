import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from visualization.common import default_parquet, setup_logging


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Plot 95% bootstrap average rewards for each experiment zone."
    )
    parser.add_argument(
        "--parquet",
        "-p",
        default=default_parquet(),
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
    args = parser.parse_args()

    try:
        logging.info(f"Reading parquet: {args.parquet}")
        df = pl.read_parquet(args.parquet)
        logging.info(f"Rows after reading: {df.shape[0]}")
    except Exception as e:
        logging.error(f"Failed to read parquet: {e}")
        sys.exit(1)

    # Check if required columns exist
    required_cols = ["experiment", "zone", "plant_id", "reward", "day", "clean_area"]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error("Missing required columns: %s", missing_cols)
        logging.info("Available columns: %s", df.columns)
        sys.exit(1)

    # Calculate returns over exp, zone, plant_id
    df = df.with_columns(
        pl.col("reward").sum().over(["experiment", "zone", "plant_id"]).alias("return")
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

    # filter by day < 14
    df = df.filter(pl.col("day") < 14)

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
        y="return",
        errorbar=("ci", 95),
        capsize=0.1,
        palette="tab10",
    )

    plt.title("Return by Agent (95% CI)")
    plt.xlabel("Agent")
    plt.ylabel("Return")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out_file = out_dir / "returns_by_agent.png"
    plt.savefig(out_file, dpi=200)
    logging.info("Saved plot to %s", out_file)

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
