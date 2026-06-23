#!/usr/bin/env python3
"""
Learn a scalar light-intensity → fixture power (W) function for E18.

Power source: raw ~10 s resolution smart-plug logs integrated over the
photoperiod via visualization/e18_power.daily_power_energy() — more accurate
than the parquet's cumulative_energy diffs.

Intensity source: mixed-e18-daily-v27.parquet, aggregated per (zone, day).

Fits:
  - Linear through origin:  P = k · a
  - Affine:                  P = P₀ + P₁ · a

Outputs:
  results/intensity_power_fit.png
  results/intensity_power_params.json

Usage:
  uv run python scripts/fit_intensity_power.py
  uv run python scripts/fit_intensity_power.py --zones 1 3 4 11
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from config import VERSION  # noqa: E402
from visualization.e18_power import PHOTOPERIOD_HOURS, daily_power_energy  # noqa: E402

PARQUET = ROOT / f"/data/plant-rl/offline/{VERSION}/mixed-e18-daily-v27.parquet"
ZONE_COLORS = {1: "#e74c3c", 3: "#3498db", 4: "#2ecc71", 11: "#9b59b6"}


def main():
    parser = argparse.ArgumentParser(description="Fit intensity → power model for E18.")
    parser.add_argument("--parquet", default=str(PARQUET))
    parser.add_argument("--zones", type=int, nargs="+", default=[1, 3, 4, 11])
    parser.add_argument("--output", default="results/intensity_power_fit.png")
    parser.add_argument("--save-params", default="results/intensity_power_params.json")
    args = parser.parse_args()

    # ---- power from raw logs (zone × date) ----
    print("Integrating raw power logs...")
    power_df = daily_power_energy(zones=args.zones)
    if power_df.is_empty():
        print("No raw power data found. Check E18_RAW_ROOT paths.")
        return
    print(f"  {power_df.height} zone-day power readings")

    # ---- intensity from parquet (zone × date) ----
    # The parquet time is America/Edmonton; the 9:00 AM local (Etc/GMT-2) row
    # appears as 01:00 AM Edmonton, so the "local date" is time.dt.date() + 1 day
    # in Edmonton. Recover local date by converting back to the zone's tz.
    df = pl.read_parquet(
        args.parquet,
        columns=["zone", "day", "time", "intensity", "outlier"],
    ).filter(pl.col("zone").is_in(args.zones))

    # Local date: convert Edmonton time → Etc/GMT-2 and take date
    intensity_by_zone_date = (
        df.with_columns(
            pl.col("time").dt.convert_time_zone("Etc/GMT-2").dt.date().alias("date")
        )
        .group_by(["zone", "date"])
        .agg(pl.col("intensity").mean())
        .sort("zone", "date")
    )

    # ---- join ----
    joined = intensity_by_zone_date.join(
        power_df.select(["zone", "date", "mean_power_W", "energy_Wh"]),
        on=["zone", "date"],
        how="inner",
    ).filter(pl.col("intensity").is_not_null() & pl.col("mean_power_W").is_not_null())

    if joined.is_empty():
        print(
            "Join produced no rows. Check date alignment between parquet and raw logs."
        )
        return

    print(f"  {joined.height} joined (zone, date) pairs after inner join")

    intensity = joined["intensity"].to_numpy()
    power_W = joined["mean_power_W"].to_numpy()
    zones_arr = joined["zone"].to_numpy()

    # ---- Model 1: linear through origin  P = k · a ----
    k_ols = float((power_W @ intensity) / (intensity @ intensity))
    pred_linear = k_ols * intensity
    r2_linear = r2_score(power_W, pred_linear)

    # Robust variant (Huber)
    hub = HuberRegressor(fit_intercept=False)
    hub.fit(intensity.reshape(-1, 1), power_W)
    k_huber = float(hub.coef_[0])
    pred_huber = k_huber * intensity
    r2_huber = r2_score(power_W, pred_huber)

    # ---- Model 2: affine  P = P₀ + P₁ · a ----
    X2 = np.column_stack([np.ones_like(intensity), intensity])
    coef_aff, *_ = np.linalg.lstsq(X2, power_W, rcond=None)
    P0_aff, P1_aff = coef_aff
    pred_affine = P0_aff + P1_aff * intensity
    r2_affine = r2_score(power_W, pred_affine)

    # Robust affine
    hub2 = HuberRegressor(fit_intercept=True)
    hub2.fit(intensity.reshape(-1, 1), power_W)
    P0_hub2, P1_hub2 = float(hub2.intercept_), float(hub2.coef_[0])
    pred_hub2 = P0_hub2 + P1_hub2 * intensity
    r2_hub2 = r2_score(power_W, pred_hub2)

    print(f"\nLinear  P = k · a  (OLS):    k = {k_ols:.3f} W   R² = {r2_linear:.4f}")
    print(f"Linear  P = k · a  (Huber):  k = {k_huber:.3f} W   R² = {r2_huber:.4f}")
    print(
        f"Affine  P = P₀ + P₁·a  (OLS):   P₀={P0_aff:.3f} W  P₁={P1_aff:.3f} W  R²={r2_affine:.4f}"
    )
    print(
        f"Affine  P = P₀ + P₁·a  (Huber): P₀={P0_hub2:.3f} W  P₁={P1_hub2:.3f} W  R²={r2_hub2:.4f}"
    )

    # ---- Per-zone linear slope (through origin) ----
    per_zone = {}
    print("\nPer-zone linear (P = k_z · a):")
    for zone in sorted(args.zones):
        mask = zones_arr == zone
        a_z, p_z = intensity[mask], power_W[mask]
        if len(a_z) < 2:
            continue
        k_z = float((p_z @ a_z) / (a_z @ a_z))
        r2_z = r2_score(p_z, k_z * a_z)
        per_zone[zone] = {"k_W": k_z, "r2": r2_z, "n": int(len(a_z))}
        print(f"  Z{zone}: k = {k_z:.3f} W  R² = {r2_z:.4f}  (n={len(a_z)})")

    # ---- Plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for zone in args.zones:
        mask = zones_arr == zone
        ax1.scatter(
            intensity[mask],
            power_W[mask],
            color=ZONE_COLORS.get(zone, "gray"),
            alpha=0.6,
            s=40,
            label=f"Z{zone}",
            zorder=3,
        )

    x_plot = np.linspace(intensity.min() * 0.9, intensity.max() * 1.05, 300)
    ax1.plot(
        x_plot,
        k_ols * x_plot,
        "k-",
        lw=2,
        label=f"Linear OLS  k={k_ols:.1f} W (R²={r2_linear:.3f})",
    )
    ax1.plot(
        x_plot,
        P0_aff + P1_aff * x_plot,
        color="steelblue",
        lw=1.5,
        linestyle="-.",
        label=f"Affine OLS P₀={P0_aff:.1f}+{P1_aff:.1f}·a (R²={r2_affine:.3f})",
    )
    for zone, info in per_zone.items():
        mask = zones_arr == zone
        a_range = np.linspace(intensity[mask].min(), intensity[mask].max(), 100)
        ax1.plot(
            a_range,
            info["k_W"] * a_range,
            color=ZONE_COLORS.get(zone, "gray"),
            lw=1,
            linestyle=":",
            label=f"Z{zone} k={info['k_W']:.1f} W",
        )
    ax1.set_xlabel("Intensity (a)")
    ax1.set_ylabel("Mean photoperiod power (W)")
    ax1.set_title("Intensity → Power")
    ax1.legend(fontsize=7.5)
    ax1.grid(True, alpha=0.25)
    ax1.spines[["top", "right"]].set_visible(False)

    # Right: residuals from global linear model, coloured by zone
    resid = power_W - k_ols * intensity
    for zone in args.zones:
        mask = zones_arr == zone
        ax2.scatter(
            intensity[mask],
            resid[mask],
            color=ZONE_COLORS.get(zone, "gray"),
            alpha=0.6,
            s=40,
            label=f"Z{zone}",
            zorder=3,
        )
    ax2.axhline(0, color="k", lw=0.8, linestyle="--")
    ax2.set_xlabel("Intensity (a)")
    ax2.set_ylabel("Residual (W)")
    ax2.set_title(f"Residuals — Global linear (k={k_ols:.2f} W)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"E18 intensity → power fit  ({PHOTOPERIOD_HOURS:.0f} h photoperiod, n={len(intensity)} zone-days)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()

    out_plot = ROOT / args.output
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_plot}")

    # ---- Save params ----
    params = {
        "photoperiod_h": PHOTOPERIOD_HOURS,
        "n_zone_days": int(len(intensity)),
        "linear_ols": {"k_W": k_ols, "r2": r2_linear},
        "linear_huber": {"k_W": k_huber, "r2": r2_huber},
        "affine_ols": {"P0_W": P0_aff, "P1_W": P1_aff, "r2": r2_affine},
        "affine_huber": {"P0_W": P0_hub2, "P1_W": P1_hub2, "r2": r2_hub2},
        "per_zone_linear": {f"zone_{z}": v for z, v in per_zone.items()},
    }
    out_params = ROOT / args.save_params
    out_params.parent.mkdir(parents=True, exist_ok=True)
    with open(out_params, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Params saved → {out_params}")


if __name__ == "__main__":
    main()
