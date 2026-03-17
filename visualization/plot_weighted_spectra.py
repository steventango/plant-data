"""
Plot weighted light spectra over time for E16.
4×3 heatmap grid (agents × zones): x = wavelength, y = day, colour = intensity.
Spectra are weighted by daily action coefficients (red/white/blue mixing).
"""

import io
import re
import sys
import tarfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from config import VERSION
from visualization.plot_e16_metrics import (
    AGENT_COLORS,
    AGENT_ORDER,
    ZONE_COLORS,
    ZONE_MAP,
)

PARQUET_PATH = Path(f"/data/plant-rl/offline/{VERSION}/mixed-{VERSION}.parquet")
SPECTRA_PATH = Path(__file__).resolve().parent.parent / "data" / "spectra.tar.gz"
OUTPUT_DIR = Path("results/weighted_spectra")

COLOR_NAME = {"red": "R", "white": "W", "blue": "B"}


def load_spectra(tarball: Path) -> tuple[dict, np.ndarray]:
    """Load spectra from tarball. Returns ({zone: {color: intensities}}, wavelengths)."""
    spectra: dict[int, dict[str, np.ndarray]] = {}
    wavelengths = None

    with tarfile.open(tarball, "r:gz") as tar:
        for member in tar.getmembers():
            m = re.search(r"zone(\d+)_(red|white|blue)\.txt$", member.name)
            if not m:
                continue
            zone = int(m.group(1))
            color = m.group(2)

            data = np.loadtxt(
                io.BytesIO(tar.extractfile(member).read()),
                skiprows=1,
            )
            wl = data[:, 0]
            intensity = data[:, 1]

            if wavelengths is None:
                wavelengths = wl
            spectra.setdefault(zone, {})[color] = intensity

    return spectra, wavelengths


def load_daily_action_coefficients(parquet_path: Path) -> pd.DataFrame:
    """Load per-zone, per-day action coefficients from parquet for E16."""
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
        pl.col("wall_time")
        .rank("ordinal")
        .over("zone", "plant_id")
        .alias("_step"),
    )
    df = df.filter(pl.col("_step") <= MAX_ACTIONS)

    # Convert step to 0-indexed day
    df = df.with_columns((pl.col("_step") - 1).alias("day"))

    # Mean coefficients per zone per day (all plants in a zone get same light)
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


def compute_weighted_spectra(
    spectra: dict, daily_coefs: pd.DataFrame
) -> dict[int, np.ndarray]:
    """Compute weighted spectrum per zone per day. Returns {zone: array(days, wavelengths)}."""
    weighted = {}
    for zone in sorted(spectra.keys()):
        zone_coefs = daily_coefs[daily_coefs["zone"] == zone].sort_values("day")
        n_days = len(zone_coefs)
        n_wl = len(spectra[zone]["red"])
        arr = np.zeros((n_days, n_wl))

        for i, (_, row) in enumerate(zone_coefs.iterrows()):
            arr[i, :] = (
                row["red_coef"] * spectra[zone]["red"]
                + row["white_coef"] * spectra[zone]["white"]
                + row["blue_coef"] * spectra[zone]["blue"]
            )
        arr = np.clip(arr, 0, None)
        weighted[zone] = arr
    return weighted


def draw(
    weighted: dict[int, np.ndarray], wavelengths: np.ndarray, output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif"})

    fig, axes = plt.subplots(4, 3, figsize=(14, 10), sharex=True, sharey=True)

    # Global intensity range for shared colour scale
    vmin = min(w.min() for w in weighted.values())
    vmax = max(w.max() for w in weighted.values())

    for row_i, agent in enumerate(AGENT_ORDER):
        zones = ZONE_MAP[agent]
        for col_i, zone in enumerate(zones):
            ax = axes[row_i, col_i]
            data = weighted[zone]
            n_days = data.shape[0]

            im = ax.pcolormesh(
                wavelengths,
                np.arange(n_days),
                data,
                cmap="inferno",
                vmin=vmin,
                vmax=vmax,
                shading="nearest",
            )
            ax.set_ylim(n_days - 0.5, -0.5)  # day 0 at top
            ax.set_yticks(range(n_days))

            # Zone label
            ax.set_title(
                f"Zone {zone}",
                fontsize=10,
                fontweight="bold",
                color=ZONE_COLORS[zone],
            )
            # Row label — use text annotation to avoid collision with tick labels
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

    # Shared axis labels
    for ax in axes[-1, :]:
        ax.set_xlabel("Wavelength (nm)", fontsize=9)

    fig.suptitle(
        "E16 — Weighted Light Spectra Over Time",
        fontsize=13,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.08, right=0.88, hspace=0.25, wspace=0.15)

    # Shared colorbar in its own axes to avoid overlap
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Intensity (counts)", fontsize=9)
    out = output_dir / "e16_weighted_spectra_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, default=str(PARQUET_PATH))
    parser.add_argument("--spectra", type=str, default=str(SPECTRA_PATH))
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    spectra, wavelengths = load_spectra(Path(args.spectra))
    print(f"Loaded spectra for {len(spectra)} zones, {len(wavelengths)} wavelengths")

    daily_coefs = load_daily_action_coefficients(Path(args.parquet))
    print(f"Loaded daily coefficients: {len(daily_coefs)} zone-day rows")

    weighted = compute_weighted_spectra(spectra, daily_coefs)
    draw(weighted, wavelengths, Path(args.output))
