#!/usr/bin/env python3
"""
Plot return and energy usage for Experiment 18, grouped by agent/policy.

Uses the pre-computed reward columns from mixed-e18-daily-v27.parquet:
  area_reward               — daily Δ log_clean_area (growth)
  energy_reward_schema_a_β  — β · (log energy_t − log e_const)
  energy_reward_schema_b    — per-step gated energy cost
  reward                    — area_reward − energy_reward_schema_a_1

Return for each schema = per-plant sum of (area_reward − schema_energy_reward).
Energy per zone = median daily energy × number of experiment days.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

from config import E18_POLICY_MAP, VERSION
from visualization.style import PARETO_COLOR, build_palette

SCHEMA_A_BETAS = [0.5, 1.0, 2.0, 4.0, 8.0]


def percentile_bounds(x: np.ndarray, lo_pct: float = 15, hi_pct: float = 100):
    lo, hi = np.percentile(x, [lo_pct, hi_pct])
    return float(lo), float(hi)


def inner_percentile_mean(
    x: np.ndarray, lo_pct: float = 15, hi_pct: float = 100
) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return 0.0
    if len(x) == 1:
        return float(x[0])
    lo, hi = np.percentile(x, [lo_pct, hi_pct])
    trimmed = x[(x >= lo) & (x <= hi)]
    return float((trimmed if len(trimmed) else x).mean())


def inner_percentile_mean_ci(
    x: np.ndarray,
    lo_pct: float = 15,
    hi_pct: float = 100,
    ci: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    p_lo, p_hi = np.percentile(x, [lo_pct, hi_pct])
    cleaned = x[(x >= p_lo) & (x <= p_hi)]
    if len(cleaned) == 0:
        cleaned = x
    mean = float(cleaned.mean())
    if len(cleaned) <= 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    boots = rng.choice(cleaned, size=(n_boot, len(cleaned)), replace=True).mean(axis=1)
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(boots, [100 * alpha, 100 * (1 - alpha)])
    return mean, float(lo), float(hi)


TRIM_LO_PCT = 10
TRIM_HI_PCT = 99


def policy_summary(
    rdf: pd.DataFrame, return_col: str, edf: dict, policy_order: list
) -> pd.DataFrame:
    rows = []
    for policy in policy_order:
        p_df = rdf[rdf["policy"] == policy]
        if p_df.empty:
            continue
        mean_ret, ret_lo, ret_hi = inner_percentile_mean_ci(
            p_df[return_col].to_numpy(),
            TRIM_LO_PCT,
            TRIM_HI_PCT,
        )
        rows.append(
            {
                "policy": policy,
                "return_mean": mean_ret,
                "return_ci_lo": ret_lo,
                "return_ci_hi": ret_hi,
                "energy": edf[policy],
            }
        )
    return pd.DataFrame(rows)


def plot_tradeoff_panel(
    ax,
    s_df: pd.DataFrame,
    color_map: dict,
    *,
    title: str | None = None,
    const_return: float | None = None,
    const_label: str = "Constant",
):
    s_df_sorted = s_df.sort_values("energy")
    pareto, max_return_seen = [], -1e9
    for _, row in s_df_sorted.iterrows():
        if row["return_mean"] > max_return_seen:
            pareto.append(row)
            max_return_seen = row["return_mean"]
    pareto_df = pd.DataFrame(pareto)

    if not pareto_df.empty:
        ax.plot(
            pareto_df["energy"],
            pareto_df["return_mean"],
            color=PARETO_COLOR,
            linestyle="--",
            linewidth=2.0,
            alpha=0.9,
            label="Pareto Frontier",
            zorder=2,
        )
    for _, row in s_df_sorted.iterrows():
        ax.errorbar(
            row["energy"],
            row["return_mean"],
            yerr=[
                [row["return_mean"] - row["return_ci_lo"]],
                [row["return_ci_hi"] - row["return_mean"]],
            ],
            fmt="o",
            color=color_map[row["policy"]],
            ecolor="#888888",
            elinewidth=1.4,
            capsize=3.5,
            markersize=9,
            zorder=3,
        )
    if const_return is not None:
        ax.axhline(
            const_return,
            color="#888888",
            linestyle="--",
            linewidth=1.4,
            zorder=0,
            label=const_label,
        )
    ax.set_xlabel("Energy (Wh)", fontsize=11)
    ax.set_ylabel("Return", fontsize=11, rotation=0, ha="right", va="center")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.0, edgecolor="none")
    sns.despine(ax=ax, top=True, right=True)


def plot_schema_returns_figure(
    rdf: pd.DataFrame,
    edf: dict[str, float],
    policy_order: list[str],
    color_map: dict[str, str],
    const_policy: str,
    out_path: Path,
):
    """Schema A (all betas) + Schema B trade-off panels using pre-computed returns."""
    n_betas = len(SCHEMA_A_BETAS)
    sns.set_theme(style="white")
    fig = plt.figure(figsize=(3.5 * n_betas, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, n_betas, hspace=0.35, wspace=0.3)

    # Top row: Schema A, one panel per beta
    for i, beta in enumerate(SCHEMA_A_BETAS):
        col = f"return_schema_a_{beta:g}"
        s_df = policy_summary(rdf, col, edf, policy_order)
        ax = fig.add_subplot(gs[0, i])
        plot_tradeoff_panel(
            ax,
            s_df,
            color_map,
            title=f"Schema A: β={beta:g}",
            const_return=inner_percentile_mean(
                rdf.loc[rdf["policy"] == const_policy, col].to_numpy(),
                TRIM_LO_PCT,
                TRIM_HI_PCT,
            ),
        )

    # Bottom row: Schema B (centred)
    s_df_b = policy_summary(rdf, "return_schema_b", edf, policy_order)
    const_return_b = inner_percentile_mean(
        rdf.loc[rdf["policy"] == const_policy, "return_schema_b"].to_numpy(),
        TRIM_LO_PCT,
        TRIM_HI_PCT,
    )
    mid = n_betas // 2
    span = 2 if n_betas >= 4 else 1
    ax_b = fig.add_subplot(
        gs[1, mid - span // 2 : mid - span // 2 + span + (1 if n_betas % 2 else 0)]
    )
    plot_tradeoff_panel(
        ax_b,
        s_df_b,
        color_map,
        title="Schema B (step-gated energy reward)",
        const_return=const_return_b,
        const_label="Constant",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved schema return plot to {out_path}")


def policy_label(name: str) -> str:
    return name.removeprefix("Sequence")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Return and Energy Usage for Experiment 18 grouped by agent."
    )
    parser.add_argument(
        "--parquet",
        "-p",
        default=f"/data/plant-rl/offline/{VERSION}/mixed-e18-daily-v27.parquet",
    )
    parser.add_argument("--out", "-o", default="results/e18_return_energy.png")
    parser.add_argument(
        "--out-schemas",
        default="results/e18_return_energy_schemas.png",
        help="Second figure: Schema A (all betas) and Schema B returns.",
    )
    parser.add_argument("--max-day", type=int, default=13)
    args = parser.parse_args()

    print(f"Reading parquet: {args.parquet}")
    df = pl.read_parquet(args.parquet)

    # Check for the pre-computed reward columns
    required = ["area_reward", "reward", "energy_reward_schema_b"] + [
        f"energy_reward_schema_a_{b:g}" for b in SCHEMA_A_BETAS
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Missing columns: {missing}")
        print("Re-generate the parquet with join_zones.py --subsample daily.")
        sys.exit(1)

    df_e18 = (
        df.filter(pl.col("experiment") == 18)
        .with_columns(
            pl.col("zone")
            .replace_strict(E18_POLICY_MAP, default="Unknown")
            .alias("policy")
        )
        .filter((pl.col("policy") != "Unknown") & (pl.col("day") <= args.max_day))
    )

    if df_e18.is_empty():
        print("No E18 data found.")
        sys.exit(1)

    # ── Per-plant returns from pre-computed step rewards ──────────────────────
    agg_exprs = [
        pl.col("area_reward").sum().alias("area_return"),
        pl.col("reward").sum().alias("reward_return"),
    ]
    for beta in SCHEMA_A_BETAS:
        col = f"energy_reward_schema_a_{beta:g}"
        agg_exprs.append(
            (pl.col("area_reward") - pl.col(col))
            .sum()
            .alias(f"return_schema_a_{beta:g}")
        )
    agg_exprs.append(
        (pl.col("area_reward") - pl.col("energy_reward_schema_b"))
        .sum()
        .alias("return_schema_b")
    )

    returns = df_e18.group_by(["zone", "policy", "plant_id"]).agg(agg_exprs)
    rdf = returns.to_pandas()

    if rdf.empty:
        print("No data found for Experiment 18.")
        sys.exit(1)

    # ── Zone energy: median daily energy × days (robust to outlier days) ─────
    zone_energy = (
        df_e18.filter(pl.col("energy").is_not_null())
        .group_by(["zone", "policy", "day"])
        .agg(pl.col("energy").median().alias("daily_energy"))
        .group_by(["zone", "policy"])
        .agg(pl.col("daily_energy").sum().alias("total_energy"))
    )
    edf: dict[str, float] = dict(
        zip(zone_energy["policy"].to_list(), zone_energy["total_energy"].to_list())
    )

    policies = [p for p in rdf["policy"].unique() if p in edf]
    policy_order = sorted(policies, key=lambda p: edf[p])
    policy_labels = [policy_label(p) for p in policy_order]
    policy_to_y = {p: i for i, p in enumerate(policy_order)}

    palette = build_palette(len(policy_order))
    color_map = dict(zip(policy_order, palette))

    # Main return column: combined RL reward (area_reward − energy_reward_a_1)
    return_col = "reward_return"

    per_policy_pct = rdf.groupby("policy", as_index=False)[return_col].agg(
        p_lo=lambda s: percentile_bounds(s.to_numpy(), TRIM_LO_PCT, TRIM_HI_PCT)[0],
        p_hi=lambda s: percentile_bounds(s.to_numpy(), TRIM_LO_PCT, TRIM_HI_PCT)[1],
    )
    rdf = rdf.merge(per_policy_pct, on="policy", how="left")
    rdf["outlier"] = (rdf[return_col] < rdf["p_lo"]) | (rdf[return_col] > rdf["p_hi"])
    rdf["y_idx"] = rdf["policy"].map(policy_to_y)

    const_policy = E18_POLICY_MAP[11]
    const_energy = edf.get(const_policy)
    const_return_df = rdf[rdf["policy"] == const_policy]
    const_return = (
        inner_percentile_mean(
            const_return_df[return_col].to_numpy(), TRIM_LO_PCT, TRIM_HI_PCT
        )
        if not const_return_df.empty
        else None
    )

    summary = []
    for policy in policy_order:
        p_df = rdf[rdf["policy"] == policy]
        if p_df.empty:
            continue
        mean_ret, ret_lo, ret_hi = inner_percentile_mean_ci(
            p_df[return_col].to_numpy(),
            TRIM_LO_PCT,
            TRIM_HI_PCT,
        )
        summary.append(
            {
                "policy": policy,
                "return_mean": mean_ret,
                "return_ci_lo": ret_lo,
                "return_ci_hi": ret_hi,
                "energy": edf[policy],
            }
        )
    s_df = pd.DataFrame(summary)

    # ── Layout ────────────────────────────────────────────────────────────────
    sns.set_theme(style="white")
    fig = plt.figure(figsize=(10, 5.5), constrained_layout=True)
    fig.get_layout_engine().set(h_pad=0.02, w_pad=0.02, hspace=0, wspace=0)
    gs_main = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.35], wspace=0.35)
    gs_a = gs_main[0, 0].subgridspec(1, 2, wspace=0.12, width_ratios=[1.15, 1.0])
    ax_energy = fig.add_subplot(gs_a[0, 0])
    ax_return = fig.add_subplot(gs_a[0, 1])
    ax_tradeoff = fig.add_subplot(gs_main[0, 1])

    y_pos = np.arange(len(policy_order))

    # ── Panel A1: Energy bars ─────────────────────────────────────────────────
    ax_energy.barh(
        y_pos,
        [edf[p] for p in policy_order],
        height=0.72,
        color=[color_map[p] for p in policy_order],
    )
    if const_energy:
        ax_energy.axvline(
            const_energy,
            color="#888888",
            linestyle="--",
            linewidth=1.4,
            zorder=0,
            label="Constant",
        )
    ax_energy.set_yticks(y_pos)
    ax_energy.set_yticklabels(policy_labels, fontsize=10)
    ax_energy.tick_params(axis="y", pad=6)
    ax_energy.invert_yaxis()
    ax_energy.set_ylim(len(policy_order) - 0.5, -0.5)
    ax_energy.set_xlabel("Energy (Wh)", fontsize=11)
    ax_energy.set_ylabel("Agent", fontsize=11, rotation=0, ha="right", va="center")
    sns.despine(ax=ax_energy, right=True, top=True)
    ax_energy.set_yticks(y_pos)
    ax_energy.set_yticklabels(policy_labels, fontsize=10)

    # ── Panel A2: Violin (reward_return distribution + mean ± 95% CI) ─────────
    rdf_clean = rdf[~rdf["outlier"]]
    sns.violinplot(
        data=rdf_clean,
        y="y_idx",
        x=return_col,
        hue="policy",
        hue_order=policy_order,
        palette=color_map,
        dodge=False,
        legend=False,
        ax=ax_return,
        inner=None,
        density_norm="width",
        orient="h",
    )
    sns.stripplot(
        data=rdf_clean,
        y="y_idx",
        x=return_col,
        order=y_pos,
        color="0.3",
        alpha=0.12,
        jitter=0.2,
        size=2.5,
        ax=ax_return,
        zorder=2,
        orient="h",
    )
    half_cap, half_mean, lw = 0.20, 0.14, 2.0
    for policy in policy_order:
        row = s_df[s_df["policy"] == policy]
        if row.empty:
            continue
        row = row.iloc[0]
        i = policy_to_y[policy]
        ax_return.hlines(
            i,
            row["return_ci_lo"],
            row["return_ci_hi"],
            colors="black",
            linewidth=lw * 0.7,
            zorder=7,
        )
        ax_return.vlines(
            row["return_ci_lo"],
            i - half_cap,
            i + half_cap,
            colors="black",
            linewidth=lw,
            zorder=8,
        )
        ax_return.vlines(
            row["return_ci_hi"],
            i - half_cap,
            i + half_cap,
            colors="black",
            linewidth=lw,
            zorder=8,
        )
        ax_return.vlines(
            row["return_mean"],
            i - half_mean,
            i + half_mean,
            colors="black",
            linewidth=lw * 1.2,
            zorder=9,
        )
    if const_return is not None:
        ax_return.axvline(
            const_return, color="#888888", linestyle="--", linewidth=1.4, zorder=0
        )
    if ax_return.get_legend():
        ax_return.get_legend().remove()
    ax_return.set_xlabel("Return (area − β·ΔlogE, β=1)", fontsize=11, rotation=0)
    ax_return.set_ylabel("")
    ax_return.set_yticks(y_pos)
    ax_return.set_yticklabels([])
    ax_return.tick_params(axis="y", left=False, labelleft=False, length=0)
    ax_return.set_ylim(len(policy_order) - 0.5, -0.5)
    sns.despine(ax=ax_return, left=True, right=True, top=True)

    # ── Panel B: Trade-off scatter (energy vs reward_return) ─────────────────
    plot_tradeoff_panel(ax_tradeoff, s_df, color_map, const_return=const_return)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved return and energy plot to {out_path}")

    plot_schema_returns_figure(
        rdf,
        edf,
        policy_order,
        color_map,
        const_policy,
        Path(args.out_schemas),
    )


if __name__ == "__main__":
    main()
