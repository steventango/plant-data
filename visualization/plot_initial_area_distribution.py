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
        description="Plot initial plant area distribution for a specific experiment and zone."
    )
    parser.add_argument(
        "--parquet",
        "-p",
        default=default_parquet(),
        help="Path to parquet file",
    )
    parser.add_argument(
        "--experiment",
        "-e",
        required=True,
        help="Experiment identifier (e.g., E11)",
    )
    parser.add_argument(
        "--zone",
        "-z",
        required=True,
        help="Zone identifier (e.g., zone1)",
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

    args = parser.parse_args()

    try:
        logging.info(f"Reading parquet: {args.parquet}")
        df = pl.read_parquet(args.parquet)
        logging.info(f"Rows after reading: {df.shape[0]}")
    except Exception as e:
        logging.error(f"Failed to read parquet: {e}")
        sys.exit(1)

    # Check if required columns exist
    required_cols = ["experiment", "zone", "time", "plant_id"]
    if "area" in df.columns:
        area_col = "area"
    elif "clean_area" in df.columns:
        area_col = "clean_area"
        required_cols.append("clean_area")
    else:
        logging.error("Neither 'area' nor 'clean_area' column found")
        sys.exit(1)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error("Missing required columns: %s", missing_cols)
        sys.exit(1)

    # Parse experiment and zone arguments
    try:
        # Extract number from experiment string (e.g., "E11" -> 11)
        exp_str = args.experiment
        if exp_str.upper().startswith("E"):
            exp_val = int(exp_str[1:])
        else:
            exp_val = int(exp_str)

        # Extract number from zone string (e.g., "zone1" -> 1)
        zone_str = args.zone
        if zone_str.lower().startswith("zone"):
            zone_val = int(zone_str[4:])
        else:
            zone_val = int(zone_str)
    except ValueError:
        logging.error(
            f"Could not parse experiment '{args.experiment}' or zone '{args.zone}' to integers."
        )
        sys.exit(1)

    # Filter by experiment and zone
    logging.info(f"Filtering for Experiment: {exp_val}, Zone: {zone_val}")
    df_filtered = df.filter(
        (pl.col("experiment") == exp_val) & (pl.col("zone") == zone_val)
    )

    if df_filtered.shape[0] == 0:
        logging.error("No data found for the specified experiment and zone.")
        sys.exit(1)

    logging.info(f"Rows after filtering: {df_filtered.shape[0]}")

    # Get initial area for each plant
    # We define initial area as the area at the earliest timestamp for each plant
    logging.info("Extracting initial area for each plant...")

    # Sort by time to ensure we get the first observation
    initial_areas = (
        df_filtered.sort("time")
        .group_by("plant_id")
        .first()  # Takes the first row for each group (which corresponds to earliest time due to sort)
        .select(["plant_id", area_col])
    )

    logging.info(f"Found {initial_areas.shape[0]} unique plants.")

    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plotting
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    data_to_plot = initial_areas.to_pandas()

    sns.histplot(data=data_to_plot, x=area_col, kde=True, binwidth=0.1)

    plt.title(
        f"Distribution of Initial Plant Area\nExperiment: {args.experiment}, Zone: {args.zone}"
    )
    plt.xlabel("Initial Plant Area (cm²)")
    plt.ylabel("Count")

    # Add summary stats to the plot
    mean_area = data_to_plot[area_col].mean()
    median_area = data_to_plot[area_col].median()
    std_area = data_to_plot[area_col].std()

    stats_text = f"Mean: {mean_area:.2f}\nMedian: {median_area:.2f}\nStd: {std_area:.2f}\nN: {len(data_to_plot)}"
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

    out_file = out_dir / f"initial_area_dist_{args.experiment}_{args.zone}.png"
    plt.savefig(out_file, dpi=200)
    logging.info("Saved plot to %s", out_file)

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
