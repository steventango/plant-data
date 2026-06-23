"""
Plot reward-over-time line charts with 95% bootstrap confidence intervals.

Outputs:
1) Grouped by zone and agent
2) Grouped by red/blue zone sets

Also produces the same two plot types for exp(reward).
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from config import VERSION

PARQUET_PATH = Path(f"/data/plant-rl/offline/{VERSION}/mixed-{VERSION}.parquet")
OUTPUT_DIR = Path("results/reward_over_time")
RED_ZONES = {1, 2, 5, 6, 9, 10}
BLUE_ZONES = {3, 4, 7, 8, 11, 12}


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 3000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        v = float(values[0])
        return v, v, v

    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return float(values.mean()), lo, hi


def load_reward_rows(path: Path, exp_id: int, max_steps: int) -> pd.DataFrame:
    df = pl.read_parquet(
        path,
        columns=["experiment", "zone", "plant_id", "wall_time", "reward", "agent"],
    )
    df = df.filter(pl.col("experiment") == exp_id)
    df = df.sort("zone", "plant_id", "wall_time")

    # Keep first max_steps timesteps per trajectory.
    df = df.with_columns(
        pl.col("wall_time").rank("ordinal").over("zone", "plant_id").alias("_step")
    )
    df = df.filter(pl.col("wall_time") <= max_steps)
    df = df.filter(pl.col("reward").is_not_null())
    df = df.filter(pl.col("wall_time") > 0)

    return df.select(["zone", "agent", "plant_id", "wall_time", "reward"]).to_pandas()


def summarize_bootstrap(
    pdf: pd.DataFrame,
    group_cols: list[str],
    metric_col: str,
    n_boot: int,
    ci: float,
) -> pd.DataFrame:
    rows: list[dict] = []

    grouped = pdf.groupby(group_cols + ["wall_time"], dropna=False)
    for keys, g in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        metric_values = g[metric_col].to_numpy(dtype=float)

        # Stable seed from keys and timestamp for deterministic bootstrap.
        key_seed = hash((tuple(keys), metric_col)) % (2**32)
        mean, lo, hi = bootstrap_mean_ci(
            metric_values, n_boot=n_boot, ci=ci, seed=key_seed
        )

        row = {col: val for col, val in zip(group_cols + ["wall_time"], keys)}
        row.update({"mean": mean, "ci_low": lo, "ci_high": hi, "n": len(metric_values)})
        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=group_cols + ["wall_time", "mean", "ci_low", "ci_high", "n"]
        )

    out = pd.DataFrame(rows)
    return out.sort_values(group_cols + ["wall_time"]).reset_index(drop=True)


def draw_zone_agent(
    summary: pd.DataFrame,
    output_dir: Path,
    exp_id: int,
    metric_label: str,
    file_name: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    plt.rcParams.update({"font.family": "sans-serif"})

    combos = (
        summary[["agent", "zone"]]
        .drop_duplicates()
        .sort_values(["agent", "zone"])
        .itertuples(index=False)
    )
    combos = list(combos)

    agents = sorted(summary["agent"].dropna().unique().tolist())
    cmap = plt.get_cmap("tab10")
    agent_color = {agent: cmap(i % 10) for i, agent in enumerate(agents)}

    for agent, zone in combos:
        g = summary[
            (summary["agent"] == agent) & (summary["zone"] == zone)
        ].sort_values("wall_time")
        x = g["wall_time"].to_numpy(dtype=float)
        y = g["mean"].to_numpy(dtype=float)
        lo = g["ci_low"].to_numpy(dtype=float)
        hi = g["ci_high"].to_numpy(dtype=float)

        color = agent_color[agent]
        label = f"{agent} | z{zone}"

        ax.plot(x, y, color=color, alpha=0.85, linewidth=1.8, label=label)
        ax.fill_between(x, lo, hi, color=color, alpha=0.16)

    ax.set_title(
        f"Experiment {exp_id}: {metric_label} over Time (Zone + Agent)", fontsize=13
    )
    ax.set_xlabel("Wall Time (days)")
    ax.set_ylabel(metric_label)
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=3, fontsize=8, framealpha=0.8)

    fig.tight_layout()
    out_path = output_dir / file_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def draw_red_blue(
    summary: pd.DataFrame,
    output_dir: Path,
    exp_id: int,
    metric_label: str,
    file_name: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    plt.rcParams.update({"font.family": "sans-serif"})

    colors = {"red": "#d62728", "blue": "#1f77b4"}
    labels = {"red": "Red zones (1,2,5,6,9,10)", "blue": "Blue zones (3,4,7,8,11,12)"}

    for group in ["red", "blue"]:
        g = summary[summary["zone_group"] == group].sort_values("wall_time")
        if g.empty:
            continue
        x = g["wall_time"].to_numpy(dtype=float)
        y = g["mean"].to_numpy(dtype=float)
        lo = g["ci_low"].to_numpy(dtype=float)
        hi = g["ci_high"].to_numpy(dtype=float)

        ax.plot(x, y, color=colors[group], linewidth=2.3, label=labels[group])
        ax.fill_between(x, lo, hi, color=colors[group], alpha=0.22)

    ax.set_title(
        f"Experiment {exp_id}: {metric_label} over Time (Red vs Blue)", fontsize=13
    )
    ax.set_xlabel("Wall Time (days)")
    ax.set_ylabel(metric_label)
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, framealpha=0.85)

    fig.tight_layout()
    out_path = output_dir / file_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def draw_agent_only(
    summary: pd.DataFrame,
    output_dir: Path,
    exp_id: int,
    metric_label: str,
    file_name: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rcParams.update({"font.family": "sans-serif"})

    agents = sorted(summary["agent"].dropna().unique().tolist())
    cmap = plt.get_cmap("tab10")

    for i, agent in enumerate(agents):
        g = summary[summary["agent"] == agent].sort_values("wall_time")
        if g.empty:
            continue
        x = g["wall_time"].to_numpy(dtype=float)
        y = g["mean"].to_numpy(dtype=float)
        lo = g["ci_low"].to_numpy(dtype=float)
        hi = g["ci_high"].to_numpy(dtype=float)

        color = cmap(i % 10)
        ax.plot(x, y, color=color, linewidth=2.2, alpha=0.95, label=agent)
        ax.fill_between(x, lo, hi, color=color, alpha=0.2)

    ax.set_title(
        f"Experiment {exp_id}: {metric_label} over Time (Agent Only)", fontsize=13
    )
    ax.set_xlabel("Wall Time (days)")
    ax.set_ylabel(metric_label)
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.85)

    fig.tight_layout()
    out_path = output_dir / file_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def add_zone_group(pdf: pd.DataFrame) -> pd.DataFrame:
    out = pdf.copy()

    def map_group(z: int) -> str:
        if int(z) in RED_ZONES:
            return "red"
        if int(z) in BLUE_ZONES:
            return "blue"
        return "other"

    out["zone_group"] = out["zone"].map(map_group)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Plot reward-over-time with 95% bootstrap CI (zone+agent and red/blue)."
    )
    parser.add_argument("--parquet", type=str, default=str(PARQUET_PATH))
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--experiment", type=int, default=17)
    parser.add_argument("--max-steps", type=int, default=14)
    parser.add_argument("--bootstrap-samples", type=int, default=3000)
    parser.add_argument("--ci", type=float, default=0.95)
    args = parser.parse_args()

    output_dir = Path(args.output)
    pdf = load_reward_rows(
        Path(args.parquet), exp_id=args.experiment, max_steps=args.max_steps
    )
    if pdf.empty:
        print(f"No rows found for experiment {args.experiment}")
        return

    # Keep only red/blue zone sets for grouped red-vs-blue chart.
    pdf = add_zone_group(pdf)
    rb_pdf = pdf[pdf["zone_group"].isin(["red", "blue"])].copy()

    # Metric 1: reward
    reward_zone_agent = summarize_bootstrap(
        pdf,
        group_cols=["agent", "zone"],
        metric_col="reward",
        n_boot=args.bootstrap_samples,
        ci=args.ci,
    )
    reward_red_blue = summarize_bootstrap(
        rb_pdf,
        group_cols=["zone_group"],
        metric_col="reward",
        n_boot=args.bootstrap_samples,
        ci=args.ci,
    )
    reward_agent_only = summarize_bootstrap(
        pdf,
        group_cols=["agent"],
        metric_col="reward",
        n_boot=args.bootstrap_samples,
        ci=args.ci,
    )

    draw_zone_agent(
        reward_zone_agent,
        output_dir,
        exp_id=args.experiment,
        metric_label="Reward",
        file_name=f"e{args.experiment}_reward_over_time_zone_agent.png",
    )
    draw_red_blue(
        reward_red_blue,
        output_dir,
        exp_id=args.experiment,
        metric_label="Reward",
        file_name=f"e{args.experiment}_reward_over_time_red_blue.png",
    )
    draw_agent_only(
        reward_agent_only,
        output_dir,
        exp_id=args.experiment,
        metric_label="Reward",
        file_name=f"e{args.experiment}_reward_over_time_agent_only.png",
    )

    # Metric 2: cumulative reward over time (per-trajectory cumulative sum)
    pdf_cum = pdf.copy()
    rb_pdf_cum = rb_pdf.copy()

    # Ensure proper ordering then compute cumulative reward per trajectory
    pdf_cum = pdf_cum.sort_values(["zone", "agent", "plant_id", "wall_time"])
    pdf_cum["cum_reward"] = pdf_cum.groupby(["zone", "agent", "plant_id"])[
        "reward"
    ].cumsum()

    rb_pdf_cum = rb_pdf_cum.sort_values(["zone", "agent", "plant_id", "wall_time"])
    rb_pdf_cum["cum_reward"] = rb_pdf_cum.groupby(["zone", "agent", "plant_id"])[
        "reward"
    ].cumsum()

    cum_zone_agent = summarize_bootstrap(
        pdf_cum,
        group_cols=["agent", "zone"],
        metric_col="cum_reward",
        n_boot=args.bootstrap_samples,
        ci=args.ci,
    )
    cum_red_blue = summarize_bootstrap(
        rb_pdf_cum,
        group_cols=["zone_group"],
        metric_col="cum_reward",
        n_boot=args.bootstrap_samples,
        ci=args.ci,
    )
    cum_agent_only = summarize_bootstrap(
        pdf_cum,
        group_cols=["agent"],
        metric_col="cum_reward",
        n_boot=args.bootstrap_samples,
        ci=args.ci,
    )

    draw_zone_agent(
        cum_zone_agent,
        output_dir,
        exp_id=args.experiment,
        metric_label="Cumulative Reward",
        file_name=f"e{args.experiment}_cumulative_reward_over_time_zone_agent.png",
    )
    draw_red_blue(
        cum_red_blue,
        output_dir,
        exp_id=args.experiment,
        metric_label="Cumulative Reward",
        file_name=f"e{args.experiment}_cumulative_reward_over_time_red_blue.png",
    )
    draw_agent_only(
        cum_agent_only,
        output_dir,
        exp_id=args.experiment,
        metric_label="Cumulative Reward",
        file_name=f"e{args.experiment}_cumulative_reward_over_time_agent_only.png",
    )

    # Metric 2: exp(reward)
    pdf_exp = pdf.copy()
    rb_pdf_exp = rb_pdf.copy()
    pdf_exp["exp_reward"] = np.exp(pdf_exp["reward"].to_numpy(dtype=float))
    rb_pdf_exp["exp_reward"] = np.exp(rb_pdf_exp["reward"].to_numpy(dtype=float))

    exp_zone_agent = summarize_bootstrap(
        pdf_exp,
        group_cols=["agent", "zone"],
        metric_col="exp_reward",
        n_boot=args.bootstrap_samples,
        ci=args.ci,
    )
    exp_red_blue = summarize_bootstrap(
        rb_pdf_exp,
        group_cols=["zone_group"],
        metric_col="exp_reward",
        n_boot=args.bootstrap_samples,
        ci=args.ci,
    )
    exp_agent_only = summarize_bootstrap(
        pdf_exp,
        group_cols=["agent"],
        metric_col="exp_reward",
        n_boot=args.bootstrap_samples,
        ci=args.ci,
    )

    draw_zone_agent(
        exp_zone_agent,
        output_dir,
        exp_id=args.experiment,
        metric_label="exp(reward)",
        file_name=f"e{args.experiment}_exp_reward_over_time_zone_agent.png",
    )
    draw_red_blue(
        exp_red_blue,
        output_dir,
        exp_id=args.experiment,
        metric_label="exp(reward)",
        file_name=f"e{args.experiment}_exp_reward_over_time_red_blue.png",
    )
    draw_agent_only(
        exp_agent_only,
        output_dir,
        exp_id=args.experiment,
        metric_label="exp(reward)",
        file_name=f"e{args.experiment}_exp_reward_over_time_agent_only.png",
    )


if __name__ == "__main__":
    main()
