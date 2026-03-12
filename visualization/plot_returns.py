import argparse
import sys
from pathlib import Path

# Add project root to sys.path to import config.py
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from config import VERSION
from transforms.attributes import get_agent_name


def plot_returns(parquet_path: Path):
    print(f"Loading parquet: {parquet_path}")
    df = pl.read_parquet(parquet_path)

    print("Processing data...")
    # filter for experiment 16
    df = df.filter(pl.col("experiment") == 16)
    # Filter for day 0 and day 14
    df_day0 = df.filter(pl.col("wall_time") == 0).select(
        ["experiment", "zone", "plant_id", pl.col("log_clean_area").alias("clean_area_0")]
    )
    df_day14 = (
        df.filter(pl.col("terminal"))
        .unique(subset=["experiment", "zone", "plant_id"], keep="first")
        .select(["experiment", "zone", "plant_id", pl.col("log_clean_area").alias("clean_area_14")])
    )

    # Join to get plants that exist on both days
    df_returns = df_day0.join(
        df_day14, on=["experiment", "zone", "plant_id"], how="inner"
    )

    if df_returns.is_empty():
        print("No plants found with both day 0 and day 14 data.")
        return

    # Filter out plants with zero or very small initial area to avoid division issues
    initial_count = df_returns.height
    df_returns = df_returns.filter(pl.col("clean_area_0") > 1.0)
    if df_returns.height < initial_count:
        print(
            f"Filtered {initial_count - df_returns.height} plants with zero/small initial area."
        )

    # Calculate return: day 14 - day 0
    df_returns = df_returns.with_columns(
        (
            (pl.col("clean_area_14") - pl.col("clean_area_0"))
        ).alias("return")
    )

    # Assign agent names
    df_returns = get_agent_name(df_returns)

    # Filter out "Other" if necessary, or just keep it
    df_returns = df_returns.filter(pl.col("agent") != "Other")

    if df_returns.is_empty():
        print("No episodes matched agent criteria.")
        return

    print(f"Total plants included: {df_returns.height}")

    # Statistics per agent
    print("\nStats per Agent:")
    stats_df = (
        df_returns.group_by("agent")
        .agg(
            [
                pl.col("return").mean().alias("mean"),
                pl.col("return").median().alias("median"),
                pl.col("return").count().alias("count"),
                pl.col("return").std().alias("std"),
                pl.col("return").min().alias("min"),
                pl.col("return").max().alias("max"),
            ]
        )
        .sort("mean", descending=True)
    )
    print(stats_df)

    # Convert to pandas for plotting
    pdf = df_returns.to_pandas()

    # Plot
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(16, 8))

    # Order agents by mean return
    agent_order = stats_df["agent"].to_list()

    # Violinplot per agent
    sns.violinplot(
        data=pdf,
        x="agent",
        y="return",
        hue="agent",
        legend=False,
        palette="pastel",
        inner="box",
        order=agent_order,
    )

    # Pointplot for mean
    sns.pointplot(
        data=pdf,
        x="agent",
        y="return",
        hue="agent",
        palette="dark",
        legend=False,
        order=agent_order,
        markers="_",
        capsize=0.2,
        alpha=0.5,
    )

    plt.xticks(rotation=45, ha="right")
    plt.title("Violin Plot of Returns by Agent")
    plt.xlabel("Agent")
    plt.ylabel("Return")

    plt.tight_layout()
    output_path = "return_violinplot_by_agent.png"
    plt.savefig(output_path)
    print(f"Plot saved to '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot returns by agent from parquet data."
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default=f"/data/plant-rl/offline/{VERSION}/mixed-{VERSION}.parquet",
        help="Path to the parquet file",
    )
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        print(f"Error: Parquet file not found at {parquet_path}")
    else:
        plot_returns(parquet_path)
