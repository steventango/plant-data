"""
Plot experiment metrics: Return, % Area Change, Final − Initial Area.
X-axis: all zones grouped by agent — lets us disentangle agent vs zone effects.
Each zone: overlapping violin, raw jittered points, mean + 95 % bootstrap CI.
Brackets: within-agent zone pairs (zone effect) and between-agent pairs (agent effect).
Statistical decomposition: nested Welch's ANOVA (agent, zone-within-agent) printed per metric.
"""

import sys
from itertools import combinations
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats as sp_stats

from config import VERSION

# ── Configurable paths ──────────────────────────────────────────────────────
PARQUET_PATH = Path(f"/data/plant-rl/offline/{VERSION}/mixed-{VERSION}.parquet")
OUTPUT_DIR = Path("results/e16_metrics")


# ── Data loading & metric computation ───────────────────────────────────────
def load_episode_metrics(path: Path, exp_id: int, max_steps: int) -> pd.DataFrame:
    df = pl.read_parquet(
        path,
        columns=[
            "experiment",
            "zone",
            "plant_id",
            "wall_time",
            "clean_area",
            "log_clean_area",
            "reward",
            "agent",
        ],
    )
    df = df.filter(pl.col("experiment") == exp_id)
    df = df.sort("zone", "plant_id", "wall_time")

    # Truncate to max_steps timesteps per episode
    df = df.with_columns(
        pl.col("wall_time").rank("ordinal").over("zone", "plant_id").alias("_step"),
    )
    df = df.filter(pl.col("_step") <= max_steps)

    episodes = df.group_by(["zone", "plant_id"]).agg(
        pl.col("reward").sum().alias("return"),
        pl.col("clean_area").first().alias("initial_area"),
        pl.col("clean_area").last().alias("final_area"),
        pl.col("agent").first().alias("agent"),
    )

    episodes = episodes.with_columns(
        (
            (pl.col("final_area") - pl.col("initial_area"))
            / pl.col("initial_area")
            * 100
        ).alias("pct_area_change"),
        ((pl.col("final_area") - pl.col("initial_area")) / 100).alias(
            "abs_area_change"
        ),
    )

    # Drop rows with zero initial area (division issues)
    episodes = episodes.filter(pl.col("initial_area") > 1.0)

    return episodes.to_pandas()


# ── Layout builder (dynamic from data) ──────────────────────────────────────
# Preset palette: base colour + 3 shades (light → dark) per agent slot
_PALETTE_BASE = [
    "#4e9ac7",
    "#e85c4a",
    "#6fbf3e",
    "#a050b0",
    "#e89c2a",
    "#2ab0a0",
    "#c06030",
    "#8080c0",
]
_PALETTE_SHADES = [
    ["#a8d4ef", "#4e9ac7", "#1e6a96"],
    ["#f4a99e", "#e85c4a", "#b52c1e"],
    ["#b5e089", "#6fbf3e", "#3a8a1a"],
    ["#d9a8e8", "#a050b0", "#6a2080"],
    ["#f7d49a", "#e89c2a", "#a06000"],
    ["#9adfd8", "#2ab0a0", "#0a7a70"],
    ["#e8b898", "#c06030", "#803010"],
    ["#c0c0e8", "#8080c0", "#404090"],
]
_PALETTE_BG = [
    "#eaf4fb",
    "#fdecea",
    "#eef9e6",
    "#f5eafb",
    "#fef6e8",
    "#e8f9f7",
    "#faeee8",
    "#eeeef8",
]


def build_layout(pdf: pd.DataFrame):
    """Derive agent_order, zone_map, and colour dicts from data."""
    # Sort agents: put Constant_White first if present, then alphabetical
    agents_in_data = sorted(pdf["agent"].unique())
    if "Constant_White" in agents_in_data:
        agents_in_data = ["Constant_White"] + [
            a for a in agents_in_data if a != "Constant_White"
        ]
    agent_order = agents_in_data

    zone_map: dict[str, list[int]] = {}
    for ag in agent_order:
        zones = sorted(pdf.loc[pdf["agent"] == ag, "zone"].unique().tolist())
        zone_map[ag] = zones

    # Assign colours
    agent_colors = {}
    zone_colors: dict[int, str] = {}
    agent_bg = {}
    for ai, ag in enumerate(agent_order):
        idx = ai % len(_PALETTE_BASE)
        agent_colors[ag] = _PALETTE_BASE[idx]
        agent_bg[ag] = _PALETTE_BG[idx]
        shades = _PALETTE_SHADES[idx]
        for zi, z in enumerate(zone_map[ag]):
            zone_colors[z] = shades[min(zi, len(shades) - 1)]

    return agent_order, zone_map, agent_colors, zone_colors, agent_bg


# ── Bootstrap 95 % CI for the mean ─────────────────────────────────────────
def bootstrap_ci(x, n_boot: int = 10_000, ci: float = 0.95, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return x.mean(), lo, hi


def iqr_bounds(values: pd.Series, multiplier: float = 1.5) -> tuple[float, float]:
    """Return lower/upper Tukey IQR bounds for a numeric series."""
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return q1, q3
    return q1 - multiplier * iqr, q3 + multiplier * iqr


def drop_iqr_outliers(
    pdf: pd.DataFrame, metric: str, multiplier: float = 1.5
) -> pd.DataFrame:
    """Drop rows whose metric value lies outside Tukey IQR bounds."""
    vals = pdf[metric].dropna()
    if len(vals) < 4:
        return pdf
    lo, hi = iqr_bounds(vals, multiplier=multiplier)
    return pdf.loc[pdf[metric].between(lo, hi)].reset_index(drop=True)


# ── Holm–Bonferroni correction on a list of p-values ───────────────────────
def holm_bonferroni(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    for rank, idx in enumerate(order):
        adjusted[idx] = min(pvals[idx] * (m - rank), 1.0)
    for i in range(1, m):
        adjusted[order[i]] = max(adjusted[order[i]], adjusted[order[i - 1]])
    return adjusted


# ── Pairwise Welch's t-tests with Holm–Bonferroni correction ───────────────
def pairwise_tests(pdf: pd.DataFrame, metric: str, group_col: str):
    groups = sorted(pdf[group_col].unique())
    pairs, pvals = [], []
    for a, b in combinations(groups, 2):
        va = pdf.loc[pdf[group_col] == a, metric].dropna().values
        vb = pdf.loc[pdf[group_col] == b, metric].dropna().values
        _, p = sp_stats.ttest_ind(va, vb, equal_var=False)
        pairs.append((a, b))
        pvals.append(p)
    adjusted = holm_bonferroni(pvals)
    return list(zip(pairs, adjusted))


# ── Nested ANOVA decomposition ────────────────────────────────────────────
def nested_anova(pdf: pd.DataFrame, metric: str, agent_order, zone_map):
    """Partition SS_total into SS_agent, SS_zone(agent), SS_residual.

    Returns (ss_agent, ss_zone_within, ss_resid, ss_total,
             eta2_agent, eta2_zone_within).
    """
    vals = pdf[metric].dropna()
    grand_mean = vals.mean()
    ss_total = ((vals - grand_mean) ** 2).sum()

    # SS_agent: between-agent
    ss_agent = 0.0
    for ag in agent_order:
        g = pdf.loc[pdf["agent"] == ag, metric].dropna()
        ss_agent += len(g) * (g.mean() - grand_mean) ** 2

    # SS_zone(agent): between-zone-within-agent
    ss_zone_within = 0.0
    for ag in agent_order:
        agent_mean = pdf.loc[pdf["agent"] == ag, metric].dropna().mean()
        for z in zone_map[ag]:
            g = pdf.loc[pdf["zone"] == z, metric].dropna()
            if len(g) == 0:
                continue
            ss_zone_within += len(g) * (g.mean() - agent_mean) ** 2

    ss_resid = ss_total - ss_agent - ss_zone_within

    eta2_agent = ss_agent / ss_total if ss_total > 0 else 0.0
    eta2_zone_within = ss_zone_within / ss_total if ss_total > 0 else 0.0

    return ss_agent, ss_zone_within, ss_resid, ss_total, eta2_agent, eta2_zone_within


# ── Welch's ANOVA (does not assume equal variances) ──────────────────────
def welch_anova(groups_data: list[np.ndarray]):
    """Welch's ANOVA via scipy.stats.alexandergovern. Returns (statistic, p)."""
    groups_data = [g for g in groups_data if len(g) >= 2]
    if len(groups_data) < 2:
        return float("nan"), float("nan")
    result = sp_stats.alexandergovern(*groups_data)
    return result.statistic, result.pvalue


def sig_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ── Layout constants ─────────────────────────────────────────────────────────
# Inter-zone spacing within an agent group and gap between agent groups
_ZONE_STEP = 0.32
_GROUP_GAP = 0.55


def _zone_positions(agent_order, zone_map) -> dict[int, float]:
    """Return x-position for each zone."""
    pos: dict[int, float] = {}
    x = 0.0
    for ag in agent_order:
        for zi, z in enumerate(zone_map[ag]):
            pos[z] = x + zi * _ZONE_STEP
        x += len(zone_map[ag]) * _ZONE_STEP + _GROUP_GAP
    return pos


def _agent_center(zone_pos: dict[int, float], agent: str, zone_map) -> float:
    return np.mean([zone_pos[z] for z in zone_map[agent]])


METRICS = [
    ("return", "Return (Σ reward)"),
    ("pct_area_change", "Area Change (%)"),
    ("abs_area_change", "Final − Initial Area (cm²)"),
]


# ── Main draw ───────────────────────────────────────────────────────────────
def draw(
    pdf: pd.DataFrame,
    output_dir: Path,
    exp_id: int,
    agent_order,
    zone_map,
    agent_colors,
    zone_colors,
    agent_bg,
    drop_outliers: bool = False,
    iqr_multiplier: float = 1.5,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    zone_pos = _zone_positions(agent_order, zone_map)
    all_zones_ordered = [z for ag in agent_order for z in zone_map[ag]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.rcParams.update({"font.family": "sans-serif"})

    for ax, (metric, label) in zip(axes, METRICS):
        metric_pdf = (
            drop_iqr_outliers(pdf, metric, multiplier=iqr_multiplier)
            if drop_outliers
            else pdf
        )
        vals_all = metric_pdf[metric].dropna()
        if len(vals_all) == 0:
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                f"No data after IQR filter\n{label}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="0.4",
            )
            continue
        ymin_data, ymax_data = vals_all.min(), vals_all.max()
        yrange = ymax_data - ymin_data

        # ── Agent background shading ────────────────────────────────────
        for ag in agent_order:
            zs = zone_map[ag]
            x0 = zone_pos[zs[0]] - _ZONE_STEP * 0.6
            x1 = zone_pos[zs[-1]] + _ZONE_STEP * 0.6
            ax.axvspan(x0, x1, color=agent_bg[ag], alpha=0.55, zorder=0, linewidth=0)

        # ── Violin per zone ─────────────────────────────────────────────
        for z in all_zones_ordered:
            vals = metric_pdf.loc[metric_pdf["zone"] == z, metric].dropna().values
            if len(vals) < 3:
                continue
            vp = ax.violinplot(
                vals,
                positions=[zone_pos[z]],
                widths=_ZONE_STEP * 0.88,
                showextrema=False,
            )
            for body in vp["bodies"]:
                body.set_facecolor(zone_colors[z])
                body.set_edgecolor("white")
                body.set_alpha(0.4)
                body.set_linewidth(0.6)

        # ── Raw data jittered ───────────────────────────────────────────
        for z in all_zones_ordered:
            vals = metric_pdf.loc[metric_pdf["zone"] == z, metric].dropna().values
            rng = np.random.default_rng(z + 42)
            jitter = rng.uniform(-_ZONE_STEP * 0.32, _ZONE_STEP * 0.32, size=len(vals))
            ax.scatter(
                zone_pos[z] + jitter,
                vals,
                s=9,
                alpha=0.5,
                color=zone_colors[z],
                edgecolors="white",
                linewidths=0.2,
                zorder=3,
            )

        # ── Mean + 95 % bootstrap CI per zone ──────────────────────────
        for z in all_zones_ordered:
            vals = metric_pdf.loc[metric_pdf["zone"] == z, metric].dropna().values
            if len(vals) == 0:
                continue
            mean, lo, hi = bootstrap_ci(vals, seed=z)
            ax.errorbar(
                zone_pos[z],
                mean,
                yerr=[[mean - lo], [hi - mean]],
                fmt="_",
                color="black",
                markersize=8,
                markeredgewidth=1.8,
                elinewidth=1.5,
                capsize=4,
                capthick=1.5,
                zorder=6,
            )

        # ── Within-agent brackets (zone effect) ────────────────────────
        bracket_ceil = ymax_data + 0.04 * yrange
        bracket_step = 0.055 * yrange
        all_within_pairs = []
        all_within_pvals = []
        for ag in agent_order:
            for za, zb in combinations(zone_map[ag], 2):
                va = metric_pdf.loc[metric_pdf["zone"] == za, metric].dropna().values
                vb = metric_pdf.loc[metric_pdf["zone"] == zb, metric].dropna().values
                _, p = sp_stats.ttest_ind(va, vb, equal_var=False)
                all_within_pairs.append((za, zb))
                all_within_pvals.append(p)
        k_within = 0
        if all_within_pvals:
            adj_within = holm_bonferroni(all_within_pvals)
            for (za, zb), p in zip(all_within_pairs, adj_within):
                if p >= 0.05:
                    continue
                xi, xj = zone_pos[za], zone_pos[zb]
                y = bracket_ceil + k_within * bracket_step
                tip = bracket_step * 0.18
                ax.plot(
                    [xi, xi, xj, xj],
                    [y, y + tip, y + tip, y],
                    lw=0.8,
                    color="0.35",
                    zorder=7,
                )
                ax.text(
                    (xi + xj) / 2,
                    y + tip * 1.1,
                    sig_label(p),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="0.35",
                    zorder=7,
                )
                k_within += 1

        # ── Between-agent brackets (agent effect) ──────────────────────
        agent_pairs = list(combinations(agent_order, 2))
        if agent_pairs:
            agent_pairs_pvals = []
            for aa, ab in agent_pairs:
                va = metric_pdf.loc[metric_pdf["agent"] == aa, metric].dropna().values
                vb = metric_pdf.loc[metric_pdf["agent"] == ab, metric].dropna().values
                _, p = sp_stats.ttest_ind(va, vb, equal_var=False)
                agent_pairs_pvals.append(p)
            adj_a = holm_bonferroni(agent_pairs_pvals)
            sig_agent = [(pair, p) for pair, p in zip(agent_pairs, adj_a) if p < 0.05]
            between_y0 = bracket_ceil + (k_within + 0.8) * bracket_step
            for k_a, ((aa, ab), p) in enumerate(sig_agent):
                xi = _agent_center(zone_pos, aa, zone_map)
                xj = _agent_center(zone_pos, ab, zone_map)
                y = between_y0 + k_a * bracket_step * 1.1
                tip = bracket_step * 0.22
                ax.plot(
                    [xi, xi, xj, xj],
                    [y, y + tip, y + tip, y],
                    lw=1.1,
                    color="0.15",
                    zorder=7,
                )
                ax.text(
                    (xi + xj) / 2,
                    y + tip * 1.1,
                    sig_label(p),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    color="0.15",
                    zorder=7,
                )

        # ── Agent-level mean + CI ────────────────────────────────────
        for ag in agent_order:
            vals = metric_pdf.loc[metric_pdf["agent"] == ag, metric].dropna().values
            if len(vals) == 0:
                continue
            mean, lo, hi = bootstrap_ci(vals, seed=hash(ag) % 1000)
            cx = _agent_center(zone_pos, ag, zone_map)
            ax.errorbar(
                cx,
                mean,
                yerr=[[mean - lo], [hi - mean]],
                fmt="_",
                color=agent_colors[ag],
                markersize=14,
                markeredgewidth=2.5,
                elinewidth=2.5,
                capsize=6,
                capthick=2.5,
                alpha=0.85,
                zorder=5,
            )

        # ── Nested ANOVA decomposition annotation ──────────────────────
        agent_groups = [
            metric_pdf.loc[metric_pdf["agent"] == ag, metric].dropna().values
            for ag in agent_order
        ]
        W_ag, p_ag = welch_anova(agent_groups)
        _, _, _, _, eta2_ag, eta2_zw = nested_anova(
            metric_pdf,
            metric,
            agent_order,
            zone_map,
        )
        ax.text(
            0.98,
            0.02,
            f"Welch Agent: W={W_ag:.2f}, p={p_ag:.3f}, η²={eta2_ag:.3f}\n"
            f"Zone|Agent: η²={eta2_zw:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            color="0.4",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.85", lw=0.6),
        )

        # ── Zone x-tick labels ──────────────────────────────────────────
        ax.set_xticks([zone_pos[z] for z in all_zones_ordered])
        ax.set_xticklabels([str(z) for z in all_zones_ordered], fontsize=8)
        ax.set_xlabel("Zone", fontsize=9)
        ax.set_ylabel(label, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linewidth=0.3, alpha=0.45, zorder=0)
        ax.tick_params(axis="both", which="both", length=2)
        ax.set_xlim(
            zone_pos[all_zones_ordered[0]] - _ZONE_STEP,
            zone_pos[all_zones_ordered[-1]] + _ZONE_STEP,
        )

        # ── Agent labels below x-axis ───────────────────────────────────
        for ag in agent_order:
            cx = _agent_center(zone_pos, ag, zone_map)
            ax.text(
                cx,
                ax.get_ylim()[0] - 0.30 * yrange,
                ag.replace("_", "\n"),
                ha="center",
                va="top",
                fontsize=7.5,
                color=agent_colors[ag],
                fontweight="bold",
                clip_on=False,
            )

    fig.suptitle(
        f"Experiment {exp_id} — Agent vs Zone Performance",
        fontsize=13,
        fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out = output_dir / f"e{exp_id}_agent_zone_metrics.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)

    # ── Console summary ─────────────────────────────────────────────────
    print("\n══ Levene's test for homogeneity of variances ══")
    for metric, label in METRICS:
        metric_pdf = (
            drop_iqr_outliers(pdf, metric, multiplier=iqr_multiplier)
            if drop_outliers
            else pdf
        )
        agent_groups = [
            metric_pdf.loc[metric_pdf["agent"] == ag, metric].dropna().values
            for ag in agent_order
        ]
        all_zones_sorted = sorted(metric_pdf["zone"].unique())
        zone_groups = [
            metric_pdf.loc[metric_pdf["zone"] == z, metric].dropna().values
            for z in all_zones_sorted
        ]
        valid_ag = [g for g in agent_groups if len(g) >= 2]
        valid_zn = [g for g in zone_groups if len(g) >= 2]
        if len(valid_ag) >= 2:
            L_ag, p_lev_ag = sp_stats.levene(*valid_ag)
            verdict_ag = "VIOLATED" if p_lev_ag < 0.05 else "ok"
            print(f"\n{label}")
            print(
                f"  Agent groups:  Levene W={L_ag:.3f}, p={p_lev_ag:.4f}  [{verdict_ag}]"
            )
        if len(valid_zn) >= 2:
            L_zn, p_lev_zn = sp_stats.levene(*valid_zn)
            verdict_zn = "VIOLATED" if p_lev_zn < 0.05 else "ok"
            print(
                f"  Zone groups:   Levene W={L_zn:.3f}, p={p_lev_zn:.4f}  [{verdict_zn}]"
            )
        for ag in agent_order:
            zs = zone_map[ag]
            if len(zs) < 2:
                continue
            within_groups = [
                metric_pdf.loc[metric_pdf["zone"] == z, metric].dropna().values
                for z in zs
            ]
            variances = [g.var(ddof=1) for g in within_groups if len(g) >= 2]
            valid_w = [g for g in within_groups if len(g) >= 2]
            if len(valid_w) >= 2:
                L_w, p_w = sp_stats.levene(*valid_w)
                v_ag = "VIOLATED" if p_w < 0.05 else "ok"
                print(
                    f"    {ag:20s} zone vars={[f'{v:.2f}' for v in variances]}  "
                    f"Levene W={L_w:.3f}, p={p_w:.4f}  [{v_ag}]"
                )
    print("  (Welch's tests used throughout — robust to unequal variances)")

    print("\n══ Nested ANOVA decomposition (Welch) ══")
    for metric, label in METRICS:
        metric_pdf = (
            drop_iqr_outliers(pdf, metric, multiplier=iqr_multiplier)
            if drop_outliers
            else pdf
        )
        agent_groups = [
            metric_pdf.loc[metric_pdf["agent"] == ag, metric].dropna().values
            for ag in agent_order
        ]
        W_ag, p_ag = welch_anova(agent_groups)
        ss_ag, ss_zw, ss_res, ss_tot, eta2_ag, eta2_zw = nested_anova(
            metric_pdf,
            metric,
            agent_order,
            zone_map,
        )
        eta2_res = ss_res / ss_tot if ss_tot > 0 else 0.0
        print(f"\n{label}")
        print(
            f"  Welch Agent ({len(agent_order)} groups): W={W_ag:.3f}, p={p_ag:.4f}, η²={eta2_ag:.4f}"
        )
        print(f"  Zone|Agent (nested):    η²={eta2_zw:.4f}")
        print(f"  Residual:               η²={eta2_res:.4f}")
        print(f"  → Agent η²={eta2_ag:.4f}, Zone|Agent η²={eta2_zw:.4f}")

    if len(agent_order) > 1:
        print("\n══ Between-agent pairwise (Holm–Bonferroni) ══")
        for metric, label in METRICS:
            metric_pdf = (
                drop_iqr_outliers(pdf, metric, multiplier=iqr_multiplier)
                if drop_outliers
                else pdf
            )
            print(f"\n{label}:")
            for (a, b), p in pairwise_tests(metric_pdf, metric, "agent"):
                print(f"  {a:20s} vs {b:20s}  p={p:.4f}  {sig_label(p)}")

    print("\n══ Within-agent pairwise by zone (global Holm–Bonferroni) ══")
    for metric, label in METRICS:
        metric_pdf = (
            drop_iqr_outliers(pdf, metric, multiplier=iqr_multiplier)
            if drop_outliers
            else pdf
        )
        print(f"\n{label}:")
        all_pairs_info = []
        for ag in agent_order:
            for za, zb in combinations(zone_map[ag], 2):
                va = metric_pdf.loc[metric_pdf["zone"] == za, metric].dropna().values
                vb = metric_pdf.loc[metric_pdf["zone"] == zb, metric].dropna().values
                _, p = sp_stats.ttest_ind(va, vb, equal_var=False)
                all_pairs_info.append((ag, za, zb, p))
        if not all_pairs_info:
            print("  (no within-agent zone pairs)")
            continue
        adj = holm_bonferroni([p for _, _, _, p in all_pairs_info])
        for (ag, za, zb, _), p_adj in zip(all_pairs_info, adj):
            print(f"  [{ag}] zone {za} vs zone {zb}  p={p_adj:.4f}  {sig_label(p_adj)}")


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        type=str,
        default=str(PARQUET_PATH),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR),
    )
    parser.add_argument(
        "--min-return",
        type=float,
        default=None,
        metavar="R",
        help="Optional: exclude plants whose total return is less than R (e.g. 2 excludes return <2).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=14,
        metavar="S",
        help="Maximum number of steps to include in the analysis.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="16",
        help="Experiment ID to plot, or 'all' to run all experiments in the dataset.",
    )
    parser.add_argument(
        "--drop-iqr-outliers",
        action="store_true",
        help="Drop Tukey IQR outliers before metric-level plotting and stats.",
    )
    parser.add_argument(
        "--iqr-multiplier",
        type=float,
        default=1.5,
        help="Tukey IQR multiplier used when --drop-iqr-outliers is set.",
    )
    args = parser.parse_args()

    parquet_path = Path(args.parquet)

    if args.experiment == "all":
        df_exps = pl.read_parquet(parquet_path, columns=["experiment"])
        exp_ids = sorted(df_exps["experiment"].unique().to_list())
        print(f"Found experiments: {exp_ids}")
    else:
        exp_ids = [int(args.experiment)]

    for exp_id in exp_ids:
        print(f"\n{'═' * 60}")
        print(f"  Experiment {exp_id}")
        print(f"{'═' * 60}")

        pdf = load_episode_metrics(parquet_path, exp_id, args.max_steps)
        print(f"Loaded {len(pdf)} plant episodes from E{exp_id}")

        if len(pdf) == 0:
            print(f"  No data for E{exp_id}, skipping.")
            continue

        if args.min_return is not None:
            before = len(pdf)
            pdf = pdf.loc[pdf["return"] >= args.min_return].reset_index(drop=True)
            print(
                f"Excluded {before - len(pdf)} plants with return <{args.min_return}"
                f" ({len(pdf)} remain)"
            )

        if len(pdf) == 0:
            print(f"  No data remaining after filter for E{exp_id}, skipping.")
            continue

        agent_order, zone_map, agent_colors, zone_colors, agent_bg = build_layout(pdf)
        print(f"Agents: {agent_order}")
        print(f"Zones:  {sorted(pdf['zone'].unique())}")

        draw(
            pdf,
            Path(args.output),
            exp_id,
            agent_order,
            zone_map,
            agent_colors,
            zone_colors,
            agent_bg,
            drop_outliers=args.drop_iqr_outliers,
            iqr_multiplier=args.iqr_multiplier,
        )
