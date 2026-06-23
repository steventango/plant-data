"""
Plot per-zone area transition scatter: x = current area (cm^2), y = next area (cm^2).
Input parquet stores clean_area in cm^2.
For each zone, fit a linear model y = m*x + b and annotate slope/intercept.
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from config import VERSION

PARQUET_PATH = Path(f"/data/plant-rl/offline/{VERSION}/mixed-{VERSION}.parquet")
OUTPUT_DIR = Path("results/zone_area_transition")
RED_ZONES = [1, 2, 5, 6, 9, 10]
BLUE_ZONES = [3, 4, 7, 8, 11, 12]


def load_transitions(path: Path, exp_id: int, max_steps: int) -> pl.DataFrame:
    df = pl.read_parquet(
        path,
        columns=["experiment", "zone", "plant_id", "wall_time", "clean_area"],
    )
    df = df.filter(pl.col("experiment") == exp_id)
    df = df.sort("zone", "plant_id", "wall_time")

    # Keep first max_steps rows per episode to match existing metric scripts.
    df = df.with_columns(
        pl.col("wall_time").rank("ordinal").over("zone", "plant_id").alias("_step")
    )
    df = df.filter(pl.col("_step") <= max_steps)

    # Build (current_area, next_area) pairs within each plant trajectory.
    df = df.with_columns(
        pl.col("clean_area").alias("current_area"),
        pl.col("clean_area").shift(-1).over(["zone", "plant_id"]).alias("next_area"),
    )

    df = df.filter(
        pl.col("current_area").is_not_null()
        & pl.col("next_area").is_not_null()
        & (pl.col("current_area") > 0)
        & (pl.col("next_area") > 0)
    )

    return df.select(["zone", "plant_id", "current_area", "next_area"])


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return slope, intercept, r2


def compute_grid(n: int) -> tuple[int, int]:
    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))
    return nrows, ncols


def draw_zone_plots(transitions: pl.DataFrame, output_dir: Path, exp_id: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf = transitions.to_pandas()
    zones = sorted(pdf["zone"].unique().tolist())

    if not zones:
        print(f"No transitions found for experiment {exp_id}.")
        return

    nrows, ncols = compute_grid(len(zones))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False
    )
    axes_flat = axes.flatten()

    for idx, zone in enumerate(zones):
        ax = axes_flat[idx]
        zone_df = pdf[pdf["zone"] == zone]

        x = zone_df["current_area"].to_numpy(dtype=float)
        y = zone_df["next_area"].to_numpy(dtype=float)

        ax.scatter(x, y, s=10, alpha=0.35, color="#1f77b4", edgecolors="none")

        slope, intercept, r2 = fit_line(x, y)
        if np.isfinite(slope) and np.isfinite(intercept):
            x_line = np.linspace(np.min(x), np.max(x), 200)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color="#d62728", linewidth=2.0)

            txt = (
                f"n={len(x)}\n"
                f"slope={slope:.4f}\n"
                f"intercept={intercept:.4f}\n"
                f"R^2={r2:.4f}"
            )
        else:
            txt = f"n={len(x)}\ninsufficient data"

        ax.text(
            0.03,
            0.97,
            txt,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.8", "lw": 0.8},
        )

        ax.set_title(f"Zone {zone}")
        ax.set_xlabel("Current Area (cm^2)")
        ax.set_ylabel("Next Area (cm^2)")
        ax.grid(True, alpha=0.25)

    for j in range(len(zones), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Experiment {exp_id}: Current vs Next Plant Area by Zone", fontsize=14
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    out_path = output_dir / f"e{exp_id}_zone_area_transition.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved -> {out_path}")


def draw_group_plots(transitions: pl.DataFrame, output_dir: Path, exp_id: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf = transitions.to_pandas()
    groups = [
        ("Red Zones", RED_ZONES, "#d62728"),
        ("Blue Zones", BLUE_ZONES, "#1f77b4"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
    axes_flat = axes.flatten()

    for idx, (name, zones, color) in enumerate(groups):
        ax = axes_flat[idx]
        group_df = pdf[pdf["zone"].isin(zones)]

        x = group_df["current_area"].to_numpy(dtype=float)
        y = group_df["next_area"].to_numpy(dtype=float)

        ax.scatter(x, y, s=9, alpha=0.28, color=color, edgecolors="none")

        slope, intercept, r2 = fit_line(x, y)
        if np.isfinite(slope) and np.isfinite(intercept):
            x_line = np.linspace(np.min(x), np.max(x), 200)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color="black", linewidth=2.0)
            txt = (
                f"zones={zones}\n"
                f"n={len(x)}\n"
                f"y = {slope:.4f}x + {intercept:.4f}\n"
                f"R^2={r2:.4f}"
            )
        else:
            txt = f"zones={zones}\nn={len(x)}\ninsufficient data"

        ax.text(
            0.03,
            0.97,
            txt,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.8", "lw": 0.8},
        )

        ax.set_title(name)
        ax.set_xlabel("Current Area (cm^2)")
        ax.set_ylabel("Next Area (cm^2)")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"Experiment {exp_id}: Current vs Next Plant Area (Aggregated Zone Groups)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    out_path = output_dir / f"e{exp_id}_zone_area_transition_red_blue.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-zone scatter of current area vs next area and linear fits."
        )
    )
    parser.add_argument("--parquet", type=str, default=str(PARQUET_PATH))
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    parser.add_argument(
        "--experiment",
        type=str,
        default="17",
        help="Experiment ID to plot, or 'all' to run all experiments in the dataset.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=14,
        help="Maximum number of steps per plant trajectory.",
    )
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    output_dir = Path(args.output)

    if args.experiment == "all":
        exp_df = pl.read_parquet(parquet_path, columns=["experiment"])
        exp_ids = sorted(exp_df["experiment"].unique().to_list())
        print(f"Found experiments: {exp_ids}")
    else:
        exp_ids = [int(args.experiment)]

    for exp_id in exp_ids:
        print(f"\n{'=' * 60}")
        print(f"Experiment {exp_id}")
        print(f"{'=' * 60}")

        transitions = load_transitions(parquet_path, exp_id, args.max_steps)
        print(f"Loaded {len(transitions)} transitions from E{exp_id}")

        if len(transitions) == 0:
            print(f"No data for E{exp_id}, skipping.")
            continue

        draw_zone_plots(transitions, output_dir, exp_id)
        draw_group_plots(transitions, output_dir, exp_id)


if __name__ == "__main__":
    main()
