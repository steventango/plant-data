"""
Plot action coefficient weights over time for E16.
4×3 stacked bar chart grid (agents × zones): x = day, y = coefficient value, colour = R/W/B.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import polars as pl

from config import VERSION
from visualization.plot_e16_metrics import (
    AGENT_COLORS,
    AGENT_ORDER,
    ZONE_COLORS,
    ZONE_MAP,
)

PARQUET_PATH = Path(f"/data/plant-rl/offline/{VERSION}/mixed-{VERSION}.parquet")
OUTPUT_DIR = Path("results/action_coef_weights")

COEF_COLORS = {"red_coef": "#e63946", "white_coef": "#adb5bd", "blue_coef": "#457b9d"}
COEF_LABELS = {"red_coef": "Red", "white_coef": "White", "blue_coef": "Blue"}


def load_daily_action_coefficients(parquet_path: Path):
    df = pl.read_parquet(
        parquet_path,
        columns=[
            "experiment",
            "zone",
            "plant_id",
            "wall_time",
            "red_coef",
            "white_coef",
            "blue_coef",
        ],
    )
    df = df.filter(pl.col("experiment") == 16)

    MAX_ACTIONS = 13
    df = df.with_columns(
        pl.col("wall_time").rank("ordinal").over("zone", "plant_id").alias("_step"),
    )
    df = df.filter(pl.col("_step") <= MAX_ACTIONS)
    df = df.with_columns((pl.col("_step") - 1).alias("day"))

    result = (
        df.group_by(["zone", "day"])
        .agg(
            pl.col("red_coef").mean(),
            pl.col("white_coef").mean(),
            pl.col("blue_coef").mean(),
        )
        .sort("zone", "day")
    )
    return result.to_pandas()


def draw(daily_coefs, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif"})

    fig, axes = plt.subplots(4, 3, figsize=(14, 10), sharex=True, sharey=True)

    coef_cols = ["red_coef", "white_coef", "blue_coef"]
    legend_handles = []

    for row_i, agent in enumerate(AGENT_ORDER):
        zones = ZONE_MAP[agent]
        for col_i, zone in enumerate(zones):
            ax = axes[row_i, col_i]
            zone_df = daily_coefs[daily_coefs["zone"] == zone].sort_values("day")
            days = zone_df["day"].values

            n_days = len(days)
            left = None
            for coef in coef_cols:
                vals = zone_df[coef].values
                bars = ax.barh(
                    days,
                    vals,
                    left=left if left is not None else 0,
                    color=COEF_COLORS[coef],
                    label=COEF_LABELS[coef],
                    height=0.8,
                )
                if row_i == 0 and col_i == 0:
                    legend_handles.append(bars)
                left = (left + vals) if left is not None else vals

            ax.set_ylim(n_days - 0.5, -0.5)  # day 0 at top
            ax.set_yticks(range(n_days))
            ax.set_xlim(0, 1.05)

            ax.set_title(
                f"Zone {zone}",
                fontsize=10,
                fontweight="bold",
                color=ZONE_COLORS[zone],
            )
            if col_i == 0:
                ax.set_ylabel("Day", fontsize=9)
                ax.annotate(
                    agent.replace("_", " "),
                    xy=(0, 0.5),
                    xytext=(-55, 0),
                    xycoords="axes fraction",
                    textcoords="offset points",
                    fontsize=10,
                    fontweight="bold",
                    color=AGENT_COLORS[agent],
                    ha="center",
                    va="center",
                    rotation=90,
                )

            ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel("Coefficient", fontsize=9)

    fig.suptitle(
        "E16 — Action Coefficient Weights Over Time",
        fontsize=13,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.08, right=0.88, hspace=0.25, wspace=0.15)

    fig.legend(
        handles=legend_handles,
        labels=[COEF_LABELS[c] for c in coef_cols],
        loc="center right",
        bbox_to_anchor=(0.99, 0.5),
        fontsize=9,
        title="Channel",
        title_fontsize=9,
        frameon=False,
    )

    out = output_dir / "e16_action_coef_weights.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, default=str(PARQUET_PATH))
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    daily_coefs = load_daily_action_coefficients(Path(args.parquet))
    print(f"Loaded daily coefficients: {len(daily_coefs)} zone-day rows")

    draw(daily_coefs, Path(args.output))
