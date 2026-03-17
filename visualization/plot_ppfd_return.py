"""
Plot correlation between zone-level PPFD and per-plant return for E16.
Scatter with regression line, Pearson r, and per-agent colouring.
Uses weighted average PPFD based on action coefficients from the dataset.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats as sp_stats

from config import VERSION
from visualization.plot_e16_metrics import (
    AGENT_COLORS,
    AGENT_ORDER,
    ZONE_MAP,
    load_e16_episode_metrics,
)

PARQUET_PATH = Path(f"/data/plant-rl/offline/{VERSION}/mixed-{VERSION}.parquet")
PPFD_PATH = Path(__file__).resolve().parent.parent / "data" / "ppfd.csv"
OUTPUT_DIR = Path("results/ppfd_return")


def load_zone_action_coefficients(parquet_path: Path) -> dict:
    """Load action coefficients from parquet and compute mean per zone.

    Filters to only include the first 13 actions (days 0-12).
    """
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

    # Filter to first 13 actions per episode
    MAX_ACTIONS = 13
    df = df.with_columns(
        pl.col("wall_time").rank("ordinal").over("zone", "plant_id").alias("_step"),
    )
    df = df.filter(pl.col("_step") <= MAX_ACTIONS)

    # Convert to pandas to avoid Polars aggregation issues
    df_pandas = df.to_pandas()

    # Compute mean coefficients per zone
    zone_means = df_pandas.groupby("zone")[
        ["red_coef", "white_coef", "blue_coef"]
    ].mean()

    # Convert to dictionary: zone -> {red_coef, white_coef, blue_coef}
    coef_dict = {}
    for zone, row in zone_means.iterrows():
        coef_dict[zone] = {
            "red_coef": row["red_coef"],
            "white_coef": row["white_coef"],
            "blue_coef": row["blue_coef"],
        }

    return coef_dict


def load_ppfd(path: Path, zone_coefs: dict) -> pd.DataFrame:
    """Load PPFD CSV and compute weighted average based on action coefficients.

    For each zone, calculates: weighted_ppfd = red_coef * ppfd_red + white_coef * ppfd_white + blue_coef * ppfd_blue
    """
    df = pd.read_csv(path)
    rows = []
    for zone, grp in df.groupby("ZONE"):
        # Get coefficients for this zone
        coefs = zone_coefs[zone]
        red_coef = coefs["red_coef"]
        white_coef = coefs["white_coef"]
        blue_coef = coefs["blue_coef"]

        # Extract PPFD values by color
        ppfd_data = {row["COLOR"]: row for _, row in grp.iterrows()}
        ppfd_elec = (
            ppfd_data["R"]["PPFD_ELEC"] * red_coef
            + ppfd_data["W"]["PPFD_ELEC"] * white_coef
            + ppfd_data["B"]["PPFD_ELEC"] * blue_coef
        )
        ppfd_sun = (
            ppfd_data["R"]["PPFD_SUN"] * red_coef
            + ppfd_data["W"]["PPFD_SUN"] * white_coef
            + ppfd_data["B"]["PPFD_SUN"] * blue_coef
        )

        rows.append(
            {
                "zone": zone,
                "ppfd_elec": ppfd_elec,
                "ppfd_sun": ppfd_sun,
            }
        )
    return pd.DataFrame(rows)


def draw(pdf: pd.DataFrame, ppfd: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = pdf.merge(ppfd, on="zone")

    fig, ax = plt.subplots(figsize=(6, 5))
    plt.rcParams.update({"font.family": "sans-serif"})

    zone_to_agent = {}
    for ag in AGENT_ORDER:
        for z in ZONE_MAP[ag]:
            zone_to_agent[z] = ag

    ppfd_col, ppfd_label = "ppfd_sun", "PPFD Sun (μmol/m²/s)"
    if True:
        # Scatter per agent
        for ag in AGENT_ORDER:
            mask = merged["agent"] == ag
            ax.scatter(
                merged.loc[mask, ppfd_col],
                merged.loc[mask, "return"],
                s=12,
                alpha=0.45,
                color=AGENT_COLORS[ag],
                label=ag.replace("_", " "),
                edgecolors="white",
                linewidths=0.2,
                zorder=3,
            )

        # Zone-level means for regression line
        zone_means = (
            merged.groupby("zone")
            .agg(
                ppfd_mean=(ppfd_col, "first"),
                return_mean=("return", "mean"),
            )
            .reset_index()
        )

        x_all = np.asarray(merged[ppfd_col].values, dtype=float)
        y_all = np.asarray(merged["return"].values, dtype=float)

        # Pearson correlation
        r, p = sp_stats.pearsonr(x_all, y_all)
        # Regression line
        slope, intercept = np.polyfit(x_all, y_all, 1)
        x_line = np.linspace(x_all.min(), x_all.max(), 100)
        ax.plot(
            x_line, slope * x_line + intercept, color="0.3", lw=1.5, ls="--", zorder=4
        )

        # Zone-mean markers
        for _, row in zone_means.iterrows():
            z = int(row["zone"])
            ag = zone_to_agent.get(z, "")
            ax.scatter(
                row["ppfd_mean"],
                row["return_mean"],
                s=60,
                marker="D",
                color=AGENT_COLORS.get(ag, "gray"),
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
            )
            ax.annotate(
                str(z),
                (row["ppfd_mean"], row["return_mean"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                color="0.3",
            )

        ax.set_xlabel(ppfd_label, fontsize=10)
        ax.set_ylabel("Return (Σ reward, days 0–13)", fontsize=10)
        ax.set_title(f"r = {r:.3f}, p = {p:.3f}", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="both", linewidth=0.3, alpha=0.45, zorder=0)
        ax.legend(fontsize=7.5, framealpha=0.8)

    fig.suptitle(
        "E16 — PPFD vs Return by Zone",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    out = output_dir / "ppfd_return_correlation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, default=str(PARQUET_PATH))
    parser.add_argument("--ppfd", type=str, default=str(PPFD_PATH))
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    parser.add_argument(
        "--min-return",
        type=float,
        default=2.0,
        metavar="R",
        help="Exclude plants whose total return is less than R.",
    )
    args = parser.parse_args()

    pdf = load_e16_episode_metrics(Path(args.parquet))
    print(f"Loaded {len(pdf)} plant episodes from E16")
    if args.min_return is not None:
        before = len(pdf)
        pdf = pdf.loc[pdf["return"] >= args.min_return].reset_index(drop=True)
        print(
            f"Excluded {before - len(pdf)} plants with return <{args.min_return} ({len(pdf)} remain)"
        )

    # Load action coefficients and compute weighted PPFD
    zone_coefs = load_zone_action_coefficients(Path(args.parquet))
    ppfd = load_ppfd(Path(args.ppfd), zone_coefs)
    print(f"Loaded PPFD data for {len(ppfd)} zones")
    draw(pdf, ppfd, Path(args.output))
