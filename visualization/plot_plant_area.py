import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
import numpy as np
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

VERSION = "v16"

def main():
    parser = argparse.ArgumentParser(
        description="Plot plant area vs time for each experiment and zone."
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
        "--no-filter",
        action="store_true",
        help="Do not filter by good days and outliers (default: filter)",
    )
    parser.add_argument(
        "--mean-per-zone",
        action="store_true",
        help="Plot mean plant area per zone (all experiments on one plot)",
    )
    parser.add_argument(
        "--group-by-agent",
        action="store_true",
        help="Group by agent instead of experiment-zone (only for --mean-per-zone)",
    )
    # TODO: migrate to dataset generation
    parser.add_argument(
        "--lower-quantile",
        type=float,
        default=0.2,
        help="Lower quantile for filtering (default: 0.2)",
    )
    parser.add_argument(
        "--upper-quantile",
        type=float,
        default=0.8,
        help="Upper quantile for filtering (default: 0.8)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize each plant's area by its initial area",
    )
    args = parser.parse_args()

    try:
        logging.info(f"Reading parquet: {args.parquet}")
        df = pl.read_parquet(args.parquet)
        logging.info(f"Rows after reading: {df.shape[0]}")
        if df.shape[0] > 0:
            logging.info(
                f"Time range in data: {df['time'].min()} to {df['time'].max()}"
            )
    except Exception as e:
        logging.error(f"Failed to read parquet: {e}")
        sys.exit(1)

    # Check if required columns exist
    required_cols = ["experiment", "zone", "time", "plant_id"]
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

    # TODO: migrate to dataset generation
    # Calculate wall_time
    # Reference time: 9:30 AM on the first day of each experiment/zone
    logging.info("Calculating wall_time...")
    df = df.with_columns(
        pl.col("time")
        .min()
        .over(["experiment", "zone"])
        .dt.truncate("1d")
        .alias("first_day_midnight")
    )
    df = df.with_columns(
        (pl.col("first_day_midnight") + pl.duration(hours=9, minutes=30)).alias(
            "ref_time"
        )
    )
    df = df.with_columns(
        ((pl.col("time") - pl.col("ref_time")) / pl.duration(days=1)).alias("wall_time")
    )
    df = df.drop(["first_day_midnight", "ref_time"])

    # Filter data
    if not args.no_filter:
        logging.info("Filtering data (day <= 13, is_good_day, not outlier)...")
        # Hard filter for day range
        df = df.filter(pl.col("day") <= 13)

        # Soft filter (set to null) for quality flags to break lines
        if "is_good_day" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("is_good_day"))
                .then(pl.col(area_col))
                .otherwise(None)
                .alias(area_col)
            )
        else:
            logging.warning("'is_good_day' column not found, skipping this filter")

        if "outlier" in df.columns:
            df = df.with_columns(
                pl.when(~pl.col("outlier"))
                .then(pl.col(area_col))
                .otherwise(None)
                .alias(area_col)
            )
        else:
            logging.warning("'outlier' column not found, skipping this filter")

    # Sort data
    logging.info("Sorting data...")
    df = df.sort(["experiment", "zone", "plant_id", "time"])

    # Handle truncation
    if "truncated" in df.columns:
        logging.info("Filtering truncated trajectories...")
        # Keep rows up to and including the first truncation event for each plant
        # We shift by 1 so that the truncation event itself is kept (cum_max becomes True on the NEXT row)
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

    # Filter data to keep only points within specified quantiles for each timestep
    logging.info(
        f"Filtering data to keep only {args.lower_quantile}-{args.upper_quantile} quantile range per experiment/zone/wall_time..."
    )
    df = df.with_columns(
        [
            pl.col(area_col)
            .quantile(args.lower_quantile)
            .over([pl.col("wall_time").round(4)])
            .alias("lower_q"),
            pl.col(area_col)
            .quantile(args.upper_quantile)
            .over([pl.col("wall_time").round(4)])
            .alias("upper_q"),
        ]
    )

    # Soft filter for quantiles
    df = df.with_columns(
        pl.when(
            (pl.col(area_col) >= pl.col("lower_q"))
            & (pl.col(area_col) <= pl.col("upper_q"))
        )
        .then(pl.col(area_col))
        .otherwise(None)
        .alias(area_col)
    ).drop(["lower_q", "upper_q"])

    # Normalize area if requested
    if args.normalize:
        logging.info("Normalizing area by initial area for each plant...")
        df = df.with_columns(
            (
                pl.col(area_col)
                / pl.col(area_col).first().over(["experiment", "zone", "plant_id"])
            ).alias(area_col)
        )

    # Get unique experiments
    experiments = sorted(df["experiment"].unique())

    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set(style="whitegrid")

    if args.mean_per_zone:
        # Plot mean plant area per zone, all experiments on one plot
        group_label = "agent" if args.group_by_agent else "experiment-zone"
        logging.info(f"Plotting IQM plant area per {group_label} (all experiments)")

        if args.group_by_agent:
            # Map experiment/zone to Agent
            df_plot = df.with_columns(
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
            # Filter out "Other" agents
            df_plot = df_plot.filter(~pl.col("group_key").str.contains("Other"))
        else:
            # Create a combined experiment-zone identifier
            df_plot = df.with_columns(
                (
                    pl.col("experiment").cast(pl.Utf8)
                    + "-"
                    + pl.col("zone").cast(pl.Utf8)
                ).alias("group_key")
            )

        # Convert to pandas for plotting
        pdf = df_plot.to_pandas()

        # Define IQM (Interquartile Mean) estimator function
        def iqm(x):
            """Calculate the interquartile mean (mean of values between Q1 and Q3)"""
            import numpy as np

            x = np.asarray(x)
            x = x[~np.isnan(x)]
            if len(x) < 4:
                # Not enough data points for IQM, fall back to mean
                return np.mean(x) if len(x) > 0 else np.nan
            q1, q3 = np.percentile(
                x, [args.lower_quantile * 100, args.upper_quantile * 100]
            )
            iqr_values = x[(x >= q1) & (x <= q3)]
            if len(iqr_values) == 0:
                # Edge case: all values are the same or very few unique values
                return np.mean(x)
            return np.mean(iqr_values)

        # Pre-calculate IQM and CI for each timestamp
        logging.info("Calculating IQM and CI for each timestamp...")
        results = []
        # Group by exp_zone and wall_time
        # Note: wall_time was rounded to 4 decimals in filtering, but here it might have more precision?
        # The filtering used: .over([pl.col("wall_time").round(4)])
        # But the column "wall_time" itself wasn't rounded in the dataframe.
        # Let's round it for grouping to ensure alignment
        pdf["wall_time_rounded"] = pdf["wall_time"].round(4)

        for (group_key, time), group in pdf.groupby(["group_key", "wall_time_rounded"]):
            vals = group[area_col].dropna().values
            if len(vals) == 0:
                continue

            val_iqm = iqm(vals)

            # Bootstrap CI (95%)
            ci_low, ci_high = val_iqm, val_iqm
            if len(vals) >= 4:
                n_boot = 1000
                try:
                    # Vectorized bootstrap
                    resamples = np.random.choice(
                        vals, (n_boot, len(vals)), replace=True
                    )
                    # Calculate IQM for each resample
                    q1 = np.percentile(resamples, 25, axis=1, keepdims=True)
                    q3 = np.percentile(resamples, 75, axis=1, keepdims=True)
                    mask = (resamples >= q1) & (resamples <= q3)
                    # Handle case where mask sum is 0 (should be rare/impossible with sufficient data)
                    counts = np.sum(mask, axis=1)
                    # Avoid division by zero
                    counts[counts == 0] = 1
                    means = np.sum(resamples * mask, axis=1) / counts
                    ci_low, ci_high = np.percentile(means, [2.5, 97.5])
                except Exception:
                    # Fallback if bootstrap fails
                    pass

            results.append(
                {
                    "group_key": group_key,
                    "wall_time": time,
                    "iqm": val_iqm,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )

        agg_df = pd.DataFrame(results)

        plt.figure(figsize=(14, 8))

        if not agg_df.empty:
            unique_groups = sorted(agg_df["group_key"].unique())
            colors = sns.color_palette("tab20", n_colors=len(unique_groups))
            labeled_groups = set()

            for i, group_key in enumerate(unique_groups):
                group = agg_df[agg_df["group_key"] == group_key].sort_values(
                    "wall_time"
                )
                color = colors[i]

                times = group["wall_time"].values
                iqms = group["iqm"].values
                lows = group["ci_low"].values
                highs = group["ci_high"].values

                # Find gaps in time
                gap_threshold = 1.5 / 24
                gap_indices = np.where(np.diff(times) > gap_threshold)[0] + 1

                start_idx = 0
                for end_idx in list(gap_indices) + [len(times)]:
                    t_seg = times[start_idx:end_idx]
                    y_seg = iqms[start_idx:end_idx]
                    l_seg = lows[start_idx:end_idx]
                    h_seg = highs[start_idx:end_idx]

                    start_idx = end_idx

                    if len(t_seg) == 0:
                        continue

                    label = None
                    if group_key not in labeled_groups:
                        label = group_key
                        labeled_groups.add(group_key)

                    plt.plot(t_seg, y_seg, color=color, label=label)
                    plt.fill_between(t_seg, l_seg, h_seg, color=color, alpha=0.2)

        legend_title = "Agent" if args.group_by_agent else "Experiment-Zone"
        plot_title = f"IQM Plant Area vs Wall Time by {legend_title} (95% CI)"
        ylabel = "IQM Normalized Plant Area" if args.normalize else "IQM Plant Area"

        plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        plt.title(plot_title)
        plt.xlabel("Wall Time (days)")
        plt.ylabel(ylabel)
        plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc="upper left")

        filename = (
            "plant_area_mean_per_agent.png"
            if args.group_by_agent
            else "plant_area_mean_per_zone.png"
        )
        out_file = out_dir / filename
        plt.savefig(out_file, dpi=200, bbox_inches="tight")
        logging.info("Saved plot to %s", out_file)

        if args.show:
            plt.show()
        else:
            plt.close()
    else:
        # Original plotting: separate plot for each experiment
        for exp in experiments:
            logging.info(f"Plotting for experiment {exp}")
            exp_data = df.filter(pl.col("experiment") == exp)

            plt.figure(figsize=(12, 6))
            # Use matplotlib directly to handle NaNs correctly (break lines)
            pdf = exp_data.to_pandas()
            zones = sorted(pdf["zone"].unique())
            colors = sns.color_palette("tab10")

            labeled_zones = set()
            for i, zone in enumerate(zones):
                zone_data = pdf[pdf["zone"] == zone]
                color = colors[i % len(colors)]

                for pid, pdata in zone_data.groupby("plant_id"):
                    # Check for large time gaps (e.g. > 1.5 hours) to break lines
                    gap_threshold = 1.5 / 24
                    times = pdata["wall_time"].values

                    # Find indices where the gap to the *next* point is large
                    # np.diff(times) gives t[i+1] - t[i]
                    # We want to split *after* i if diff > threshold, so split index is i+1
                    gap_indices = np.where(np.diff(times) > gap_threshold)[0] + 1

                    start_idx = 0
                    # Iterate through segments
                    for end_idx in list(gap_indices) + [len(pdata)]:
                        segment = pdata.iloc[start_idx:end_idx]
                        start_idx = end_idx

                        if segment.empty:
                            continue

                        # Determine label
                        # We only want one legend entry per zone.
                        # If we haven't labeled this zone yet, do it now.
                        seg_label = None
                        if zone not in labeled_zones:
                            seg_label = zone
                            labeled_zones.add(zone)

                        plt.plot(
                            segment["wall_time"],
                            segment[area_col],
                            color=color,
                            alpha=0.5,
                            label=seg_label,
                        )

            plt.legend(title="Zone")

            plt.title(f"Plant Area vs Wall Time for Experiment {exp}")
            plt.xlabel("Wall Time (days)")
            plt.ylabel("Plant Area")
            plt.xticks(rotation=45)
            plt.tight_layout()

            out_file = out_dir / f"plant_area_E{exp}.png"
            plt.savefig(out_file, dpi=200)
            logging.info("Saved plot to %s", out_file)

            if args.show:
                plt.show()
            else:
                plt.close()


if __name__ == "__main__":
    main()
