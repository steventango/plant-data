"""
Verification dashboard for the E18 daily dataset (mixed-e18-daily-v27.parquet).

Shows per-zone trajectories (mean ± 95% CI across plants) for:
  - Intensity (action)
  - Clean area
  - Daily reward
  - Daily energy (Wh)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import E18_POLICY_MAP, VERSION

PARQUET = Path(f"/data/plant-rl/offline/{VERSION}/mixed-e18-daily-v27.parquet")
OUTPUT = Path("results/e18_daily_dataset_verification.png")

ZONE_ORDER = [1, 3, 4, 11]
ZONE_COLORS = {1: "#e74c3c", 3: "#3498db", 4: "#2ecc71", 11: "#9b59b6"}


def bootstrap_ci(vals: np.ndarray, n_boot: int = 2000, ci: float = 0.95, seed: int = 0):
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    if len(vals) == 1:
        v = float(vals[0])
        return v, v, v
    rng = np.random.default_rng(seed)
    boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    alpha = (1 - ci) / 2
    return float(vals.mean()), float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha))


def summarize(pdf, metric: str):
    rows = []
    for (zone, day), g in pdf.groupby(["zone", "day"]):
        vals = g[metric].dropna().to_numpy(dtype=float)
        mean, lo, hi = bootstrap_ci(vals, seed=int(zone * 100 + day))
        rows.append({"zone": int(zone), "day": int(day), "mean": mean, "lo": lo, "hi": hi})
    import pandas as pd
    return pd.DataFrame(rows).sort_values(["zone", "day"])


def main():
    import pandas as pd
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default=str(PARQUET))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    path = Path(args.parquet)
    print(f"Loading {path} ...")
    df = pl.read_parquet(path, columns=[
        "zone", "day", "plant_id", "intensity", "clean_area",
        "area_reward", "reward", "energy", "agent",
    ])
    pdf = df.to_pandas()

    metrics = [
        ("intensity", "Intensity (a = Σchannel / 100)", False),
        ("clean_area", "Clean area (cm²)", False),
        ("area_reward", "Area reward (Δ log area)", True),
        ("energy", "Energy (Wh / day)", False),
        ("reward", "RL reward (area − β·ΔlogE, β=1)", True),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)), sharex=True)
    plt.rcParams.update({"font.family": "sans-serif"})

    for ax, (metric, ylabel, zero_line) in zip(axes, metrics):
        summ = summarize(pdf, metric)
        for zone in ZONE_ORDER:
            g = summ[summ["zone"] == zone]
            if g.empty:
                continue
            policy = E18_POLICY_MAP.get(zone, f"Z{zone}")
            label = f"Z{zone} {policy}"
            color = ZONE_COLORS[zone]
            x = g["day"].to_numpy()
            y = g["mean"].to_numpy()
            lo = g["lo"].to_numpy()
            hi = g["hi"].to_numpy()
            ax.plot(x, y, color=color, linewidth=2, label=label)
            ax.fill_between(x, lo, hi, color=color, alpha=0.18)
        if zero_line:
            ax.axhline(0, color="k", linewidth=0.7, linestyle="--", alpha=0.4)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9, ncol=2, framealpha=0.85)

    axes[-1].set_xlabel("Day")
    fig.suptitle(
        "E18 daily dataset — Zones 1, 3, 4, 11\n(mean ± 95% bootstrap CI across plants)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    out = Path(__file__).resolve().parent.parent / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
