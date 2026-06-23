"""
Plot diff of weighted light spectra: each zone minus the per-agent average.
4×3 heatmap grid (agents × zones): x = wavelength, y = day, colour = diff intensity.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import VERSION
from visualization.plot_weighted_spectra import (
    SPECTRA_PATH,
    compute_weighted_spectra,
    load_daily_action_coefficients,
    load_spectra,
)

PARQUET_PATH = Path(f"/data/plant-rl/offline/{VERSION}/mixed-{VERSION}.parquet")
OUTPUT_DIR = Path("results/weighted_spectra")


def compute_diff(
    weighted: dict[int, np.ndarray], agent_order, zone_map
) -> dict[int, np.ndarray]:
    """Subtract per-agent mean spectrum (over zones) from each zone."""
    diff = {}
    for agent in agent_order:
        zones = zone_map[agent]
        # Align days: use min number of days across zones in this agent
        n_days = min(weighted[z].shape[0] for z in zones)
        stack = np.stack(
            [weighted[z][:n_days] for z in zones], axis=0
        )  # (n_zones, days, wl)
        mean = stack.mean(axis=0)  # (days, wl)
        for z in zones:
            diff[z] = weighted[z][:n_days] - mean
    return diff


def draw_diff(
    diff: dict[int, np.ndarray],
    wavelengths: np.ndarray,
    agent_order,
    zone_map,
    agent_colors,
    zone_colors,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif"})

    fig, axes = plt.subplots(4, 3, figsize=(14, 10), sharex=True, sharey=True)

    # Symmetric colour scale
    absmax = max(np.abs(d).max() for d in diff.values())
    vmin, vmax = -absmax, absmax

    for row_i, agent in enumerate(agent_order):
        zones = zone_map[agent]
        for col_i, zone in enumerate(zones):
            ax = axes[row_i, col_i]
            data = diff[zone]
            n_days = data.shape[0]

            im = ax.pcolormesh(
                wavelengths,
                np.arange(n_days),
                data,
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                shading="nearest",
            )
            ax.set_ylim(n_days - 0.5, -0.5)  # day 0 at top
            ax.set_yticks(range(n_days))

            ax.set_title(
                f"Zone {zone}",
                fontsize=10,
                fontweight="bold",
                color=zone_colors[zone],
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
                    color=agent_colors[agent],
                    ha="center",
                    va="center",
                    rotation=90,
                )

            ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel("Wavelength (nm)", fontsize=9)

    fig.suptitle(
        "E16 — Weighted Spectra: Zone − Agent Average",
        fontsize=13,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.08, right=0.88, hspace=0.25, wspace=0.15)

    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Δ Intensity (counts)", fontsize=9)

    out = output_dir / "e16_spectra_diff_heatmap.png"
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

    from visualization.plot_e16_metrics import build_layout, load_episode_metrics

    pdf = load_episode_metrics(Path(args.parquet), exp_id=16, max_steps=13)
    agent_order, zone_map, agent_colors, zone_colors, agent_bg = build_layout(pdf)
    diff = compute_diff(weighted, agent_order, zone_map)
    draw_diff(
        diff,
        wavelengths,
        agent_order,
        zone_map,
        agent_colors,
        zone_colors,
        Path(args.output),
    )
