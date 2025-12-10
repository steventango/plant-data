import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

VERSION = "v17"

def main():
    parser = argparse.ArgumentParser(
        description="Plot distribution of plant growth ratio (final area / initial area)."
    )
    parser.add_argument(
        "--parquet",
        "-p",
        default=f"/data/plant-rl/offline/{VERSION}/mixed-{VERSION}.parquet",
        help="Path to parquet file",
    )
    parser.add_argument(
        "--out",
        "-o",
        default="results/plant_area_plots",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--show", action="store_true", help="Show the plots interactively"
    )
    parser.add_argument(
        "--initial-day",
        type=float,
        default=0.0,
        help="Day to use for initial area (default: 0)",
    )
    parser.add_argument(
        "--final-day",
        type=float,
        default=13.0,
        help="Day to use for final area (default: 13)",
    )
    parser.add_argument(
        "--day-tolerance",
        type=float,
        default=0.5,
        help="Tolerance in days for matching initial/final day (default: 0.5)",
    )
    parser.add_argument(
        "--group-by-agent",
        action="store_true",
        help="Group by agent instead of experiment-zone",
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
    area_col = "clean_area"
    if area_col not in df.columns:
        if "area" in df.columns:
            area_col = "area"
        else:
            logging.error("Neither 'clean_area' nor 'area' column found")
            logging.info("Available columns: %s", df.columns)
            sys.exit(1)

    required_cols = ["experiment", "zone", "plant_id", "wall_time"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error("Missing required columns: %s", missing_cols)
        logging.info("Available columns: %s", df.columns)
        sys.exit(1)

    # Sort data
    logging.info("Sorting data...")
    df = df.sort(["experiment", "zone", "plant_id", "wall_time"])

    # Handle truncation - filter out post-truncation data
    if "truncated" in df.columns:
        logging.info("Filtering truncated trajectories...")
        mask = (
            pl.col("truncated")
            .fill_null(False)
            .shift(1, fill_value=False)
            .cum_max()
            .not_()
            .over(["experiment", "zone", "plant_id"])
        )
        original_count = df.shape[0]
        df = df.filter(mask)
        filtered_count = df.shape[0]
        logging.info(
            f"Filtered {original_count - filtered_count} rows due to truncation"
        )

    # Get initial area (near day 0)
    logging.info(f"Extracting initial area near day {args.initial_day}...")
    initial_areas = (
        df.filter(
            (pl.col("wall_time") >= args.initial_day - args.day_tolerance)
            & (pl.col("wall_time") <= args.initial_day + args.day_tolerance)
        )
        .sort("wall_time")
        .group_by(["experiment", "zone", "plant_id"])
        .first()
        .select(["experiment", "zone", "plant_id", pl.col(area_col).alias("initial_area")])
    )
    logging.info(f"Found {initial_areas.shape[0]} plants with initial area data")

    # Get final area (near day 13)
    logging.info(f"Extracting final area near day {args.final_day}...")
    final_areas = (
        df.filter(
            (pl.col("wall_time") >= args.final_day - args.day_tolerance)
            & (pl.col("wall_time") <= args.final_day + args.day_tolerance)
        )
        .sort("wall_time", descending=True)
        .group_by(["experiment", "zone", "plant_id"])
        .first()
        .select(["experiment", "zone", "plant_id", pl.col(area_col).alias("final_area")])
    )
    logging.info(f"Found {final_areas.shape[0]} plants with final area data")

    # Join initial and final areas
    growth_df = initial_areas.join(
        final_areas,
        on=["experiment", "zone", "plant_id"],
        how="inner",
    )
    logging.info(f"Found {growth_df.shape[0]} plants with both initial and final area")

    # Filter out plants with zero or very small initial area to avoid division issues
    min_initial_area = 1.0  # Minimum initial area to consider
    growth_df = growth_df.filter(pl.col("initial_area") > min_initial_area)
    logging.info(f"After filtering small initial areas: {growth_df.shape[0]} plants")

    # Calculate growth ratio
    growth_df = growth_df.with_columns(
        (pl.col("final_area") / pl.col("initial_area")).alias("growth_ratio")
    )

    # Filter out extreme outliers (growth ratio > 100 is likely an error)
    max_ratio = 100.0
    growth_df = growth_df.filter(pl.col("growth_ratio") <= max_ratio)
    logging.info(f"After filtering extreme ratios: {growth_df.shape[0]} plants")

    if growth_df.shape[0] == 0:
        logging.error("No valid data after filtering. Check your parameters.")
        sys.exit(1)

    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Add group key for visualization
    if args.group_by_agent:
        growth_df = growth_df.with_columns(
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
            .alias("group_key")
        )
        # Filter out "Other" groups
        growth_df = growth_df.filter(~pl.col("group_key").str.contains("Other"))
    else:
        growth_df = growth_df.with_columns(
            (
                pl.lit("E")
                + pl.col("experiment").cast(pl.Utf8)
                + pl.lit("_Z")
                + pl.col("zone").cast(pl.Utf8)
            ).alias("group_key")
        )

    # Convert to pandas for plotting
    pdf = growth_df.to_pandas()

    # Plotting
    sns.set(style="whitegrid")

    # Plot 1: Histogram of growth ratios
    plt.figure(figsize=(12, 6))
    
    unique_groups = sorted(pdf["group_key"].unique())
    
    if len(unique_groups) <= 10:
        # Use different colors for each group
        for group in unique_groups:
            group_data = pdf[pdf["group_key"] == group]
            sns.histplot(
                data=group_data,
                x="growth_ratio",
                kde=True,
                label=group,
                alpha=0.5,
                bins=30,
            )
        plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        # Too many groups, plot all together
        sns.histplot(data=pdf, x="growth_ratio", kde=True, bins=50)

    plt.title(
        f"Distribution of Plant Growth Ratio\n(Final Area @ Day {args.final_day} / Initial Area @ Day {args.initial_day})"
    )
    plt.xlabel("Growth Ratio")
    plt.ylabel("Count")

    # Add summary stats
    mean_ratio = pdf["growth_ratio"].mean()
    median_ratio = pdf["growth_ratio"].median()
    std_ratio = pdf["growth_ratio"].std()

    stats_text = f"Mean: {mean_ratio:.2f}\nMedian: {median_ratio:.2f}\nStd: {std_ratio:.2f}\nN: {len(pdf)}"
    plt.text(
        0.95,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    out_file = out_dir / "growth_ratio_histogram.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    logging.info("Saved histogram to %s", out_file)

    if args.show:
        plt.show()
    else:
        plt.close()

    # Plot 2: Box plot by group
    plt.figure(figsize=(14, 6))

    if args.group_by_agent:
        order = sorted(unique_groups)
    else:
        # Sort by experiment-zone
        order = sorted(unique_groups, key=lambda x: (int(x.split("_")[0][1:]), int(x.split("_")[1][1:])))

    sns.boxplot(data=pdf, x="group_key", y="growth_ratio", order=order)
    plt.xticks(rotation=45, ha="right")
    plt.title(
        f"Growth Ratio by Group\n(Final Area @ Day {args.final_day} / Initial Area @ Day {args.initial_day})"
    )
    plt.xlabel("Group")
    plt.ylabel("Growth Ratio")

    plt.tight_layout()

    out_file = out_dir / "growth_ratio_boxplot.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    logging.info("Saved boxplot to %s", out_file)

    if args.show:
        plt.show()
    else:
        plt.close()

    # Plot 3: Violin plot by group
    plt.figure(figsize=(14, 6))

    sns.violinplot(data=pdf, x="group_key", y="growth_ratio", order=order)
    plt.xticks(rotation=45, ha="right")
    plt.title(
        f"Growth Ratio Distribution by Group\n(Final Area @ Day {args.final_day} / Initial Area @ Day {args.initial_day})"
    )
    plt.xlabel("Group")
    plt.ylabel("Growth Ratio")

    plt.tight_layout()

    out_file = out_dir / "growth_ratio_violin.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    logging.info("Saved violin plot to %s", out_file)

    if args.show:
        plt.show()
    else:
        plt.close()

    # Print summary statistics per group
    logging.info("\n=== Growth Ratio Summary Statistics ===")
    for group in order:
        group_data = pdf[pdf["group_key"] == group]["growth_ratio"]
        logging.info(
            f"{group}: N={len(group_data)}, "
            f"Mean={group_data.mean():.2f}, "
            f"Median={group_data.median():.2f}, "
            f"Std={group_data.std():.2f}, "
            f"Min={group_data.min():.2f}, "
            f"Max={group_data.max():.2f}"
        )


if __name__ == "__main__":
    main()