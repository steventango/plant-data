#!/usr/bin/env python3
"""
Plot E18 power use and cumulative energy over time, one line per agent/policy.

Data are at 4-hourly subsample timesteps (~3 per photoperiod day). Uses
per-interval ``energy`` (Wh) and ``cumulative_energy`` from the offline
parquet.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from config import E18_POLICY_MAP
from visualization.common import default_parquet

# Slide-style palette (slides/scripts/make_slide_figures.py / make_film_strip.py).
BG = "#FAFAF7"
MUTED = "#9A9A93"
# E18 photoperiod 09:00–21:00; wall_time 0 = 09:30 on day 0.
PHOTO_START = -0.5 / 24
PHOTO_END = 11.5 / 24


def add_night_shading(ax, max_day: int) -> None:
    """Gray bands for dark hours; photoperiod stays on the off-white background."""
    for d in range(max_day + 2):
        ax.axvspan(
            d + PHOTO_END,
            d + 1 + PHOTO_START,
            color=MUTED,
            alpha=0.12,
            linewidth=0,
            zorder=0,
        )
    ax.set_facecolor(BG)
    ax.grid(False)


def main():
    parser = argparse.ArgumentParser(
        description="Plot E18 power use and cumulative energy over time by agent."
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
        default="results/e18_power_energy_over_time.png",
        help="Output file path for the plot",
    )
    parser.add_argument(
        "--max-day",
        type=int,
        default=13,
        help="Only plot days <= this index (day 13 is terminal)",
    )
    args = parser.parse_args()

    print(f"Reading parquet: {args.parquet}")
    df = pl.read_parquet(args.parquet).filter(pl.col("experiment") == 18)
    for col in ("energy", "cumulative_energy"):
        if col not in df.columns:
            print(f"No `{col}` column found -- regenerate the dataset (generate.sh).")
            sys.exit(1)

    # One row per (zone, time) at 4-hourly resolution.
    steps = (
        df.select("zone", "time", "day", "wall_time", "energy", "cumulative_energy")
        .unique(subset=["zone", "time"])
        .filter(pl.col("day") <= args.max_day)
        .sort("zone", "time")
        .with_columns(
            pl.col("zone")
            .replace_strict(E18_POLICY_MAP, default="Unknown")
            .alias("policy"),
        )
        .filter(pl.col("policy") != "Unknown")
        .filter(pl.col("energy").is_not_null())
        .sort("zone", "wall_time")
    )

    pdf = steps.to_pandas()
    if pdf.empty:
        print("No E18 energy data found.")
        sys.exit(1)

    # Break lines at day boundaries (one photoperiod segment per day).
    pdf = pdf.sort_values(["zone", "wall_time"])
    pdf["segment"] = pdf.groupby("zone")["day"].transform(
        lambda d: d.ne(d.shift()).cumsum().fillna(0).astype(int)
    )
    pdf["line_id"] = pdf["zone"].astype(str) + "_" + pdf["segment"].astype(str)

    policy_order = [
        E18_POLICY_MAP[z]
        for z in [11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        if E18_POLICY_MAP[z] in set(pdf["policy"])
    ]
    palette = dict(zip(policy_order, sns.color_palette("husl", len(policy_order))))

    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)

    for ax in axes:
        add_night_shading(ax, args.max_day)

    sns.lineplot(
        data=pdf,
        x="wall_time",
        y="energy",
        hue="policy",
        units="line_id",
        hue_order=policy_order,
        palette=palette,
        estimator=None,
        ax=axes[0],
        linewidth=1.5,
        legend=False,
    )
    axes[0].set_title(
        "Interval Energy by Agent (4-hourly)", fontsize=13, fontweight="bold"
    )
    axes[0].set_xlabel("Wall time (days)", fontsize=11)
    axes[0].set_ylabel("Energy over interval (Wh)", fontsize=11)

    sns.lineplot(
        data=pdf,
        x="wall_time",
        y="cumulative_energy",
        hue="policy",
        units="line_id",
        hue_order=policy_order,
        palette=palette,
        estimator=None,
        ax=axes[1],
        linewidth=1.5,
        legend=False,
    )
    axes[1].set_title(
        "Cumulative Energy by Agent (4-hourly)", fontsize=13, fontweight="bold"
    )
    axes[1].set_xlabel("Wall time (days)", fontsize=11)
    axes[1].set_ylabel("Cumulative energy (Wh)", fontsize=11)

    for ax in axes:
        sns.despine(ax=ax)

    handles = [
        plt.Line2D([0], [0], color=palette[p], linewidth=1.5, label=p)
        for p in policy_order
    ]
    fig.legend(
        handles=handles,
        title="Policy / Agent",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=8,
    )

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved power/energy over time plot to {out_path}")


if __name__ == "__main__":
    main()
