#!/usr/bin/env python3
"""
Analyze reward-with-energy schemas for Experiment 18.

Schema A (log-energy penalty):
    return' = return - β * (log E_policy - log E_Constant)

Schema B (beat-Constant gate):
    return' = -E_policy                         if return >= 0.9 * return_Constant
              -fail_mult * max(E)               otherwise
"""

SCHEMA_B_FAIL_ENERGY_MULT = 2.0
SCHEMA_B_GROWTH_FRAC = 0.95
SCHEMA_A_BETAS = [0.0, 0.5, 1.0, 2.0]

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import E18_POLICY_MAP, VERSION


def inner_percentile_mean(x: np.ndarray, pct: float = 95) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return 0.0
    if len(x) == 1:
        return float(x[0])
    lo_pct = (100 - pct) / 2
    hi_pct = 100 - lo_pct
    lo, hi = np.percentile(x, [lo_pct, hi_pct])
    trimmed = x[(x >= lo) & (x <= hi)]
    if len(trimmed) == 0:
        trimmed = x
    return float(trimmed.mean())


def pareto_frontier(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    s = df.sort_values(x)
    rows, max_y = [], -np.inf
    for _, row in s.iterrows():
        if row[y] > max_y:
            rows.append(row)
            max_y = row[y]
    return pd.DataFrame(rows)


def load_e18_summary(parquet: str, max_day: int) -> tuple[pd.DataFrame, float, float]:
    df = pl.read_parquet(parquet)
    df_e18 = df.filter(pl.col("experiment") == 18).with_columns(
        pl.col("zone").replace_strict(E18_POLICY_MAP, default="Unknown").alias("policy")
    )
    df_e18 = df_e18.filter(
        (pl.col("policy") != "Unknown") & (pl.col("day") <= max_day)
    )

    returns = df_e18.group_by(["zone", "policy", "plant_id"]).agg(
        pl.col("reward").sum().alias("raw_return"),
    )
    energy = (
        df_e18.select("zone", "policy", "time", "cumulative_energy")
        .unique(subset=["zone", "time"])
        .group_by("zone", "policy")
        .agg(pl.col("cumulative_energy").max().alias("energy"))
    )
    merged = returns.join(energy, on=["zone", "policy"]).to_pandas()

    e_const = merged.loc[merged["policy"] == "Constant", "energy"].iloc[0]
    r_const = inner_percentile_mean(
        merged.loc[merged["policy"] == "Constant", "raw_return"].to_numpy()
    )

    rows = []
    for policy in merged["policy"].unique():
        sub = merged[merged["policy"] == policy]
        rows.append({
            "policy": policy,
            "energy": sub["energy"].iloc[0],
            "raw_return": inner_percentile_mean(sub["raw_return"].to_numpy()),
            "log_energy_delta": np.log(sub["energy"].iloc[0]) - np.log(e_const),
        })
    return pd.DataFrame(rows), e_const, r_const


def induced_return_schema_a(base: pd.DataFrame, beta: float) -> pd.Series:
    return base["raw_return"] - beta * base["log_energy_delta"]


def schema_b_fail_penalty(max_energy: float) -> float:
    return -SCHEMA_B_FAIL_ENERGY_MULT * max_energy


def schema_b_gate_threshold(r_const: float) -> float:
    return SCHEMA_B_GROWTH_FRAC * r_const


def schema_b_passed(base: pd.DataFrame, r_gate: float) -> pd.Series:
    return base["raw_return"] >= r_gate


def induced_return_schema_b(
    base: pd.DataFrame, r_gate: float, fail_penalty: float | None = None
) -> pd.Series:
    if fail_penalty is None:
        fail_penalty = schema_b_fail_penalty(base["energy"].max())
    passed = schema_b_passed(base, r_gate)
    return np.where(passed, -base["energy"], fail_penalty)


def plot_schema_a_panel(ax, base, beta, palette, energy_label):
    df = base.copy()
    df["induced"] = induced_return_schema_a(df, beta)
    pf = pareto_frontier(df, "energy", "induced")
    for _, row in df.iterrows():
        ax.scatter(
            row["energy"], row["induced"],
            color=palette[row["policy"]], s=80, zorder=3,
        )
    ax.plot(
        pf["energy"], pf["induced"], color="#e74c3c", linestyle="--",
        linewidth=2, alpha=0.85, zorder=1,
    )
    title = "Raw return (β=0)" if beta == 0 else f"Schema A: β={beta:g}"
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(energy_label)
    ax.set_ylabel("Induced return")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze reward-with-energy schemas for E18."
    )
    parser.add_argument(
        "--parquet", "-p",
        default=f"/data/plant-rl/offline/{VERSION}/mixed-{VERSION}.parquet",
    )
    parser.add_argument(
        "--out", "-o", default="results/e18_reward_with_energy.png",
    )
    parser.add_argument("--max-day", type=int, default=13)
    args = parser.parse_args()

    base, e_const, r_const = load_e18_summary(args.parquet, args.max_day)
    policy_order = [
        E18_POLICY_MAP[z]
        for z in [11, 3, 1, 2, 4, 5, 6, 7, 8, 9, 10]
        if E18_POLICY_MAP[z] in set(base["policy"])
    ]
    palette = dict(zip(policy_order, sns.color_palette("muted", len(policy_order))))
    raw_pareto = pareto_frontier(base, "energy", "raw_return")
    energy_label = "Energy (Wh)"
    fail_penalty = schema_b_fail_penalty(base["energy"].max())
    r_gate = schema_b_gate_threshold(r_const)

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1.0], hspace=0.35, wspace=0.3)

    for i, beta in enumerate(SCHEMA_A_BETAS):
        ax = fig.add_subplot(gs[0, i])
        plot_schema_a_panel(ax, base, beta, palette, energy_label)

    # Rank heatmap
    ax_rank = fig.add_subplot(gs[1, 0:2])
    rank_rows = []
    for beta in SCHEMA_A_BETAS:
        ind = induced_return_schema_a(base, beta)
        ranks = pd.Series(ind.values, index=base["policy"]).rank(ascending=False, method="min")
        for policy in policy_order:
            rank_rows.append({"beta": beta, "policy": policy, "rank": ranks[policy]})
    pivot = (
        pd.DataFrame(rank_rows)
        .pivot(index="policy", columns="beta", values="rank")
        .reindex(policy_order)
    )
    sns.heatmap(
        pivot, annot=True, fmt=".0f", cmap="RdYlGn_r", ax=ax_rank,
        cbar_kws={"label": "Rank (1=best)"},
    )
    ax_rank.set_title("Schema A: rank vs β", fontweight="bold")
    ax_rank.set_xlabel("β")
    ax_rank.set_ylabel("Policy")

    # Schema B
    ax_b = fig.add_subplot(gs[1, 2:4])
    df_b = base.copy()
    df_b["passed"] = schema_b_passed(df_b, r_gate)
    df_b["induced"] = induced_return_schema_b(df_b, r_gate, fail_penalty)
    pf_b = pareto_frontier(df_b[df_b["passed"]], "energy", "induced")
    for _, row in df_b.iterrows():
        marker = "o" if row["passed"] else "x"
        ax_b.scatter(
            row["energy"], row["induced"], s=100,
            color=palette[row["policy"]], marker=marker, zorder=3,
        )
        if row["passed"]:
            ax_b.annotate(
                row["policy"].replace("Sequence", "Seq."),
                (row["energy"], row["induced"]),
                fontsize=7, ha="center", va="top",
            )
    ax_b.axhline(
        fail_penalty, color="0.5", linestyle=":", linewidth=1.5,
        label=f"Fail penalty ({SCHEMA_B_FAIL_ENERGY_MULT:.0f}× max E)",
    )
    if not pf_b.empty:
        ax_b.plot(
            pf_b["energy"], pf_b["induced"], color="#e74c3c", linestyle="--",
            linewidth=2, alpha=0.85, label="Pareto (pass only)",
        )
    ax_b.set_title(
        f"Schema B: pass→−E, fail→{fail_penalty:.0f}\n"
        f"(R ≥ {SCHEMA_B_GROWTH_FRAC:.0%} Constant = {r_gate:.3f})",
        fontweight="bold",
    )
    ax_b.set_xlabel(energy_label)
    ax_b.set_ylabel("Induced return")
    ax_b.legend(fontsize=8, loc="lower left")

    fig.suptitle(
        "E18 reward-with-energy schemas",
        fontsize=14, fontweight="bold", y=1.01,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved to {out_path}")

    print(f"\nConstant baseline: E={e_const:.1f} Wh, inner-95% return={r_const:.4f}")
    print(f"Schema B gate: {SCHEMA_B_GROWTH_FRAC:.0%} Constant → R ≥ {r_gate:.4f}")
    print(f"Raw Pareto: {list(raw_pareto['policy'])}")
    print("\nSchema A rankings:")
    for beta in SCHEMA_A_BETAS:
        ind = induced_return_schema_a(base, beta)
        top = pd.Series(ind.values, index=base["policy"]).sort_values(ascending=False).head(3)
        print(f"  β={beta:g}: {', '.join(top.index)}")
    print(f"\nSchema B (pass→−E, fail→{fail_penalty:.0f}):")
    print(df_b.sort_values("induced", ascending=False)[
        ["policy", "raw_return", "energy", "passed", "induced"]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
