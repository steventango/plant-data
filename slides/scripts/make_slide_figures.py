"""
Render minimalist E17 figures for the slides deck.

Outputs PNGs into slides/assets/figures/, styled to match the off-white
disco-design aesthetic used by slides/index.html and slides/css/theme.css.

Reuses data-loading and bootstrap helpers from visualization/plot_reward_over_time.py
and the red/blue zone split from visualization/plot_zone_area_transition.py.
"""

import io
import re
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from config import VERSION
from visualization.plot_reward_over_time import (
    bootstrap_mean_ci,
    load_reward_rows,
    summarize_bootstrap,
)
from visualization.plot_zone_area_transition import (
    BLUE_ZONES,
    RED_ZONES,
    fit_line,
    load_transitions,
)

PARQUET = Path(f"/data/plant-rl/offline/{VERSION}/mixed-{VERSION}.parquet")
SPECTRA = REPO_ROOT / "data" / "spectra.tar.gz"
OUT = REPO_ROOT / "slides" / "assets" / "figures"
EXPERIMENT = 17
MAX_STEPS = 14

BG = "#FAFAF7"
INK = "#111111"
MUTED = "#9A9A93"
GRID = "#E6E5DE"
ACCENT = "#2E5BFF"
RED = "#D04A3C"
BLUE = "#3D5CC8"
WHITE_BAR = "#B0AFA8"

AGENT_PALETTE = {
    "RedRed": "#C8493C",
    "RedBlue": "#A85FB2",
    "BlueRed": "#5F8DD3",
    "BlueBlue": "#2E5BFF",
}

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "savefig.edgecolor": BG,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "regular",
        "axes.titlepad": 14,
        "axes.labelsize": 11,
        "axes.labelpad": 8,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "font.family": ["Inter", "Helvetica Neue", "Arial", "sans-serif"],
        "font.size": 11,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    }
)


def _save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"saved -> {path}")


def fig_plant_area_trajectories():
    df = pl.read_parquet(
        PARQUET,
        columns=["experiment", "zone", "plant_id", "wall_time", "clean_area", "agent"],
    )
    df = df.filter(pl.col("experiment") == EXPERIMENT)
    df = df.filter(pl.col("wall_time").is_not_null() & pl.col("clean_area").is_not_null())
    df = df.filter((pl.col("wall_time") >= 0) & (pl.col("wall_time") <= MAX_STEPS))
    pdf = df.sort("zone", "plant_id", "wall_time").to_pandas()

    fig, ax = plt.subplots(figsize=(12, 6))

    for (_, _), g in pdf.groupby(["zone", "plant_id"]):
        t = g["wall_time"].to_numpy(float)
        a = g["clean_area"].to_numpy(float)
        gaps = np.where(np.diff(t) > 1.0)[0] + 1
        start = 0
        for end in list(gaps) + [len(t)]:
            ax.plot(t[start:end], a[start:end], color=MUTED, alpha=0.18, linewidth=0.6)
            start = end

    agents = ["BlueBlue", "BlueRed", "RedBlue", "RedRed"]
    for agent in agents:
        sub = pdf[pdf["agent"] == agent]
        if sub.empty:
            continue
        sub = sub.copy()
        sub["t_round"] = sub["wall_time"].round(2)
        mean = sub.groupby("t_round")["clean_area"].mean()
        ax.plot(
            mean.index.to_numpy(),
            mean.to_numpy(),
            color=AGENT_PALETTE[agent],
            linewidth=2.0,
            alpha=0.95,
            label=agent,
        )

    ax.set_xlabel("Wall time (days)")
    ax.set_ylabel("Plant area (cm²)")
    ax.set_xlim(0, MAX_STEPS)
    ax.legend(loc="upper left", ncol=4, columnspacing=1.6, handlelength=1.4)
    _save(fig, "e17_plant_area.png")


def fig_reward_over_time():
    pdf = load_reward_rows(PARQUET, exp_id=EXPERIMENT, max_steps=MAX_STEPS)
    if pdf.empty:
        print("no reward rows for E17")
        return

    pdf = pdf.sort_values(["zone", "agent", "plant_id", "wall_time"])
    pdf["cum_reward"] = pdf.groupby(["zone", "agent", "plant_id"])["reward"].cumsum()

    summary = summarize_bootstrap(
        pdf,
        group_cols=["agent"],
        metric_col="cum_reward",
        n_boot=3000,
        ci=0.95,
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    for agent in ["BlueBlue", "BlueRed", "RedBlue", "RedRed"]:
        g = summary[summary["agent"] == agent].sort_values("wall_time")
        if g.empty:
            continue
        x = g["wall_time"].to_numpy(float)
        y = g["mean"].to_numpy(float)
        lo = g["ci_low"].to_numpy(float)
        hi = g["ci_high"].to_numpy(float)
        c = AGENT_PALETTE[agent]
        ax.fill_between(x, lo, hi, color=c, alpha=0.14, linewidth=0)
        ax.plot(x, y, color=c, linewidth=2.0, label=agent)

    ax.set_xlabel("Wall time (days)")
    ax.set_ylabel("Cumulative reward")
    ax.set_xlim(0, MAX_STEPS)
    ax.legend(loc="upper left", ncol=4, columnspacing=1.6, handlelength=1.4)
    _save(fig, "e17_reward_over_time.png")


def fig_action_coef():
    df = pl.read_parquet(
        PARQUET,
        columns=[
            "experiment", "zone", "plant_id", "wall_time", "agent",
            "red_coef", "white_coef", "blue_coef",
        ],
    )
    df = df.filter(pl.col("experiment") == EXPERIMENT)
    df = df.with_columns(
        pl.col("wall_time").rank("ordinal").over("zone", "plant_id").alias("_step")
    )
    df = df.filter(pl.col("_step") <= MAX_STEPS)
    df = df.with_columns((pl.col("_step") - 1).alias("day"))

    agg = (
        df.group_by(["agent", "day"])
        .agg(
            pl.col("red_coef").mean().alias("red"),
            pl.col("white_coef").mean().alias("white"),
            pl.col("blue_coef").mean().alias("blue"),
        )
        .sort("agent", "day")
        .to_pandas()
    )

    agents = ["BlueBlue", "BlueRed", "RedBlue", "RedRed"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), sharey=True)
    for ax, agent in zip(axes, agents):
        g = agg[agg["agent"] == agent].sort_values("day")
        if g.empty:
            ax.set_visible(False)
            continue
        x = g["day"].to_numpy()
        r = g["red"].to_numpy()
        w = g["white"].to_numpy()
        b = g["blue"].to_numpy()
        ax.stackplot(
            x, r, w, b,
            colors=[RED, WHITE_BAR, BLUE],
            labels=["red", "white", "blue"],
            edgecolor=BG,
            linewidth=0.4,
        )
        ax.set_title(agent)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(0, 1.01)
        ax.set_xlabel("Day")
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_ylabel("Channel mix")
    axes[0].spines["left"].set_visible(True)
    axes[-1].legend(loc="center right", bbox_to_anchor=(1.35, 0.5))
    fig.subplots_adjust(wspace=0.18)
    _save(fig, "e17_action_coef.png")


def fig_zone_transition():
    transitions = load_transitions(PARQUET, EXPERIMENT, MAX_STEPS)
    pdf = transitions.to_pandas()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    groups = [
        ("Red zones", RED_ZONES, RED, axes[0]),
        ("Blue zones", BLUE_ZONES, BLUE, axes[1]),
    ]
    for name, zones, color, ax in groups:
        g = pdf[pdf["zone"].isin(zones)]
        x = g["current_area"].to_numpy(float)
        y = g["next_area"].to_numpy(float)
        ax.scatter(x, y, s=8, alpha=0.22, color=color, edgecolors="none")

        slope, intercept, r2 = fit_line(x, y)
        if np.isfinite(slope) and np.isfinite(intercept):
            xs = np.linspace(0, x.max(), 200)
            ax.plot(xs, slope * xs + intercept, color=INK, linewidth=1.4)
            ax.plot(xs, xs, color=MUTED, linewidth=0.8, linestyle=(0, (4, 4)))
            ax.text(
                0.04, 0.94,
                f"slope {slope:.2f}   R²  {r2:.2f}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=10, color=INK,
            )
        ax.set_title(name, color=color)
        ax.set_xlabel("Area at day t  (cm²)")
        ax.set_xlim(0, x.max() * 1.02)
        ax.set_ylim(0, max(y.max(), x.max()) * 1.02)
    axes[0].set_ylabel("Area at day t+1  (cm²)")
    fig.subplots_adjust(wspace=0.12)
    _save(fig, "e17_zone_transition.png")


def fig_spectrum():
    if not SPECTRA.exists():
        print(f"no spectra archive at {SPECTRA}; skipping spectrum figure")
        return

    zone_spectra: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    with tarfile.open(SPECTRA, "r:gz") as tar:
        for m in tar.getmembers():
            mm = re.search(r"zone(\d+)_(red|white|blue)\.txt$", m.name)
            if not mm:
                continue
            zone = int(mm.group(1))
            color = mm.group(2)
            data = np.loadtxt(io.BytesIO(tar.extractfile(m).read()), skiprows=1)
            zone_spectra.setdefault(zone, {})[color] = (data[:, 0], data[:, 1])

    if not zone_spectra:
        print("no zones parsed from spectra archive; skipping spectrum figure")
        return

    zone = sorted(zone_spectra.keys())[0]
    chans = zone_spectra[zone]

    fig, ax = plt.subplots(figsize=(12, 4.8))
    for color, hex_color, label in [
        ("blue", BLUE, "blue"),
        ("white", WHITE_BAR, "white"),
        ("red", RED, "red"),
    ]:
        if color not in chans:
            continue
        wl, intensity = chans[color]
        ax.fill_between(wl, 0, intensity, color=hex_color, alpha=0.22, linewidth=0)
        ax.plot(wl, intensity, color=hex_color, linewidth=1.6, label=label)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity")
    ax.set_yticks([])
    ax.legend(loc="upper right", ncol=3, columnspacing=1.6, handlelength=1.4)
    _save(fig, "e17_spectrum.png")


if __name__ == "__main__":
    print(f"output -> {OUT}")
    fig_plant_area_trajectories()
    fig_reward_over_time()
    fig_action_coef()
    fig_zone_transition()
    fig_spectrum()
    print("done")
