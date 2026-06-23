#!/usr/bin/env python3
"""Compare E18 zone averages: biomass, final vision area, and return."""

import argparse
import re
from datetime import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from scipy import stats

from config import E18_POLICY_MAP, tzinfo
from visualization.common import default_parquet

BIOMASS_CSV = "Plant RL Schedule 2026 - Biomass (Fresh Weight) 2026.06.15.csv"
FINAL_TIME = time(1, 0, tzinfo=tzinfo)
POLICY_ORDER = [E18_POLICY_MAP[z] for z in [11, 3, 1, 2, 4, 5, 6, 7, 8, 9, 10]]


def load_biomass(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, header=None)
    headers = raw.iloc[1, 1:13].tolist()
    avg_row = raw.iloc[-1, 1:13]
    rows = []
    for j, h in enumerate(headers):
        m = re.match(r"zone (\d+)", str(h).strip().lower())
        if m:
            rows.append({"zone": int(m.group(1)), "biomass_g": float(avg_row.iloc[j])})
    return pd.DataFrame(rows).groupby("zone")["biomass_g"].mean().reset_index()


def load_metrics(parquet_path: Path, max_day: int) -> pd.DataFrame:
    df = pl.read_parquet(parquet_path).filter(pl.col("experiment") == 18)

    final = (
        df.filter((pl.col("day") == 14) & (pl.col("time").dt.time() == FINAL_TIME))
        .group_by("zone")
        .agg(pl.col("clean_area").mean().alias("final_area"))
    )
    returns = (
        df.filter(pl.col("day") <= max_day)
        .group_by(["zone", "plant_id"])
        .agg(pl.col("reward").sum().alias("return"))
        .group_by("zone")
        .agg(pl.col("return").mean().alias("return_mean"))
    )
    return returns.join(final, on="zone").sort("zone").to_pandas()


def normalize(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def corr_label(x: np.ndarray, y: np.ndarray) -> str:
    r, p = stats.pearsonr(x, y)
    rho, _ = stats.spearmanr(x, y)
    return f"Pearson r = {r:.2f} (p = {p:.2f})\nSpearman ρ = {rho:.2f}"


def main():
    parser = argparse.ArgumentParser(
        description="Plot E18 biomass vs final area vs return by zone."
    )
    parser.add_argument(
        "--parquet",
        "-p",
        default=default_parquet(),
    )
    parser.add_argument(
        "--biomass-csv",
        default=BIOMASS_CSV,
        help="Biomass harvest CSV with AVG row",
    )
    parser.add_argument(
        "--out",
        "-o",
        default="results/e18_biomass_validation.png",
    )
    parser.add_argument("--max-day", type=int, default=13)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    biomass = load_biomass(root / args.biomass_csv)
    metrics = load_metrics(args.parquet, args.max_day)
    df = metrics.merge(biomass, on="zone")
    df["policy"] = df["zone"].map(E18_POLICY_MAP)
    df["label"] = df.apply(lambda r: f"Z{int(r.zone)}\n{r.policy[:18]}", axis=1)
    df = df.set_index("policy").loc[POLICY_ORDER].reset_index()
    df["norm_return"] = normalize(df["return_mean"])
    df["norm_area"] = normalize(df["final_area"])
    df["norm_biomass"] = normalize(df["biomass_g"])

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # 1. Normalized grouped bars
    ax = axes[0]
    x = np.arange(len(df))
    w = 0.25
    ax.bar(x - w, df["norm_return"], width=w, label="Return", color="#3498db")
    ax.bar(x, df["norm_area"], width=w, label="Final area", color="#2ecc71")
    ax.bar(x + w, df["norm_biomass"], width=w, label="Biomass", color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Normalized (0 = min, 1 = max)")
    ax.set_title("Zone rankings (normalized)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)

    # 2. Final area vs biomass
    ax = axes[1]
    sns.regplot(
        data=df,
        x="final_area",
        y="biomass_g",
        ax=ax,
        scatter_kws={
            "s": 100,
            "color": "#2ecc71",
            "edgecolors": "k",
            "linewidths": 0.5,
        },
        line_kws={"color": "#27ae60"},
        ci=95,
    )
    for _, row in df.iterrows():
        ax.annotate(
            f"Z{int(row.zone)}",
            (row.final_area, row.biomass_g),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=9,
        )
    ax.set_xlabel("Final clean_area (day 14 01:00)")
    ax.set_ylabel("Biomass (g)")
    ax.set_title("Final area vs biomass", fontweight="bold")
    ax.text(
        0.05,
        0.95,
        corr_label(df["final_area"].values, df["biomass_g"].values),
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # 3. Return vs biomass
    ax = axes[2]
    sns.regplot(
        data=df,
        x="return_mean",
        y="biomass_g",
        ax=ax,
        scatter_kws={
            "s": 100,
            "color": "#3498db",
            "edgecolors": "k",
            "linewidths": 0.5,
        },
        line_kws={"color": "#2980b9"},
        ci=95,
    )
    for _, row in df.iterrows():
        ax.annotate(
            f"Z{int(row.zone)}",
            (row.return_mean, row.biomass_g),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=9,
        )
    ax.set_xlabel(f"Return (reward sum, day ≤ {args.max_day})")
    ax.set_ylabel("Biomass (g)")
    ax.set_title("Return vs biomass", fontweight="bold")
    ax.text(
        0.05,
        0.95,
        corr_label(df["return_mean"].values, df["biomass_g"].values),
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.suptitle(
        "E18: Biomass vs vision final area vs return",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
