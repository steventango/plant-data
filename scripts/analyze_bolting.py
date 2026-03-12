import argparse
import sys
from pathlib import Path

# Add project root to sys.path to import config.py
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import scipy.stats as stats
import pingouin as pg
from statsmodels.stats.oneway import anova_oneway
from statsmodels.formula.api import ols
import statsmodels.api as sm

from config import VERSION
from transforms.attributes import get_agent_name


def compute_days_until_bolting(df: pl.DataFrame) -> pl.DataFrame:
    """
    For each plant, find the first wall_time where bolted_pred is True.
    Returns a DataFrame with one row per plant: experiment, zone, plant_id, days_until_bolting.
    Plants that never bolt have days_until_bolting = None.
    """
    # Get first bolting day per plant
    bolted_rows = (
        df.filter(pl.col("bolted_pred"))
        .sort("wall_time")
        .group_by(["experiment", "zone", "plant_id"])
        .agg(pl.col("wall_time").first().alias("days_until_bolting"))
    )

    # Get all unique plants
    all_plants = df.select(["experiment", "zone", "plant_id"]).unique()

    # Left join to include plants that never bolted (days_until_bolting = None)
    result = all_plants.join(
        bolted_rows, on=["experiment", "zone", "plant_id"], how="left"
    )

    return result


def analyze_bolting(parquet_path: Path):
    print(f"Loading parquet: {parquet_path}")
    df = pl.read_parquet(parquet_path)

    # Filter for experiment 16
    print("Filtering for Experiment 16...")
    df = df.filter(pl.col("experiment") == 16)

    if df.is_empty():
        print("Error: No data for Experiment 16.")
        return

    # Compute days_until_bolting per plant
    print("Computing days_until_bolting per plant...")
    df_bolting = compute_days_until_bolting(df)

    # Assign agent names
    df_bolting = get_agent_name(df_bolting)
    df_bolting = df_bolting.filter(pl.col("agent") != "Other")

    # Report censoring (never-bolted plants)
    total_plants = df_bolting.height
    never_bolted = df_bolting.filter(pl.col("days_until_bolting").is_null()).height
    bolted_plants = total_plants - never_bolted
    print(f"\nTotal plants: {total_plants}")
    print(f"  Bolted: {bolted_plants}")
    print(f"  Never bolted (censored): {never_bolted}")

    if never_bolted > 0:
        print("\nNever-bolted plants by agent:")
        never_bolted_by_agent = (
            df_bolting.filter(pl.col("days_until_bolting").is_null())
            .group_by("agent")
            .agg(pl.col("plant_id").count().alias("count"))
            .sort("agent")
        )
        print(never_bolted_by_agent)

    # Drop never-bolted plants for parametric/non-parametric analysis
    df_analysis = df_bolting.filter(pl.col("days_until_bolting").is_not_null())

    if df_analysis.is_empty():
        print("Error: No plants bolted. Cannot perform analysis.")
        return

    # --- Descriptive Statistics ---
    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)

    print("\n--- Stats per Agent (Policy) ---")
    stats_by_agent = (
        df_analysis.group_by("agent")
        .agg(
            [
                pl.col("days_until_bolting").mean().alias("mean"),
                pl.col("days_until_bolting").median().alias("median"),
                pl.col("days_until_bolting").count().alias("count"),
                pl.col("days_until_bolting").std().alias("std"),
                pl.col("days_until_bolting").min().alias("min"),
                pl.col("days_until_bolting").max().alias("max"),
            ]
        )
        .sort("mean")
    )
    print(stats_by_agent)

    print("\n--- Stats per Zone ---")
    stats_by_zone = (
        df_analysis.group_by("zone")
        .agg(
            [
                pl.col("days_until_bolting").mean().alias("mean"),
                pl.col("days_until_bolting").median().alias("median"),
                pl.col("days_until_bolting").count().alias("count"),
                pl.col("days_until_bolting").std().alias("std"),
            ]
        )
        .sort("zone")
    )
    print(stats_by_zone)

    print("\n--- Stats per Agent × Zone ---")
    stats_by_agent_zone = (
        df_analysis.group_by(["agent", "zone"])
        .agg(
            [
                pl.col("days_until_bolting").mean().alias("mean"),
                pl.col("days_until_bolting").median().alias("median"),
                pl.col("days_until_bolting").count().alias("count"),
                pl.col("days_until_bolting").std().alias("std"),
            ]
        )
        .sort(["agent", "zone"])
    )
    print(stats_by_agent_zone)

    # Convert to pandas for statistical tests and plotting
    pdf = df_analysis.to_pandas()

    # --- Assumption Testing ---
    print("\n" + "=" * 60)
    print("ASSUMPTION TESTING")
    print("=" * 60)

    # Shapiro-Wilk test for normality per group
    print("\n--- Shapiro-Wilk Normality Test (per agent) ---")
    for agent in sorted(pdf["agent"].unique()):
        data = pdf[pdf["agent"] == agent]["days_until_bolting"]
        if len(data) >= 3:
            w, p = stats.shapiro(data)
            sig = " *" if p < 0.05 else ""
            print(f"  {agent}: W={w:.4f}, p={p:.4f}{sig}")
        else:
            print(f"  {agent}: insufficient data (n={len(data)})")

    # Levene's test for homogeneity of variance across policies
    groups_policy = [
        pdf[pdf["agent"] == a]["days_until_bolting"] for a in pdf["agent"].unique()
    ]
    stat_levene, p_levene = stats.levene(*groups_policy)
    print(f"\n--- Levene's Test (across policies) ---")
    print(f"  Statistic={stat_levene:.4f}, p={p_levene:.4f}")
    if p_levene < 0.05:
        print("  → Unequal variances detected. Welch's ANOVA is appropriate.")
    else:
        print("  → Equal variances assumption holds.")

    # Levene's test across zones
    groups_zone = [
        pdf[pdf["zone"] == z]["days_until_bolting"] for z in pdf["zone"].unique()
    ]
    stat_levene_z, p_levene_z = stats.levene(*groups_zone)
    print(f"\n--- Levene's Test (across zones) ---")
    print(f"  Statistic={stat_levene_z:.4f}, p={p_levene_z:.4f}")

    # --- One-Way Analysis: Policy Effect ---
    print("\n" + "=" * 60)
    print("ONE-WAY ANALYSIS: POLICY EFFECT")
    print("=" * 60)

    # Standard ANOVA
    f_stat, p_anova = stats.f_oneway(*groups_policy)
    print(f"\nStandard One-Way ANOVA: F={f_stat:.4f}, p={p_anova:.4f}")

    # Welch's ANOVA (robust to unequal variances)
    welch_result = anova_oneway(
        pdf["days_until_bolting"], pdf["agent"], use_var="unequal"
    )
    print(f"Welch's ANOVA: F={welch_result.statistic:.4f}, p={welch_result.pvalue:.4f}")

    # Kruskal-Wallis (non-parametric)
    kw_stat, kw_p = stats.kruskal(*groups_policy)
    print(f"Kruskal-Wallis: H={kw_stat:.4f}, p={kw_p:.4f}")

    # --- Post-hoc: Policy ---
    print("\n--- Post-hoc: Games-Howell (Policy) ---")
    try:
        gh_policy = pg.pairwise_gameshowell(
            data=pdf, dv="days_until_bolting", between="agent"
        )
        p_col = "pval" if "pval" in gh_policy.columns else "p-unc"
        print(gh_policy[["A", "B", "diff", "se", "T", "df", p_col]].to_string())
    except Exception as e:
        print(f"Error running Games-Howell: {e}")

    # --- One-Way Analysis: Zone Effect (Constant White only) ---
    print("\n" + "=" * 60)
    print("ONE-WAY ANALYSIS: ZONE EFFECT (Constant_White only)")
    print("=" * 60)

    cw_data = pdf[pdf["agent"] == "Constant_White"]
    if len(cw_data) > 0:
        cw_zones = cw_data["zone"].unique()
        cw_groups = [
            cw_data[cw_data["zone"] == z]["days_until_bolting"] for z in cw_zones
        ]
        if len(cw_groups) >= 2 and all(len(g) >= 2 for g in cw_groups):
            welch_cw = anova_oneway(
                cw_data["days_until_bolting"], cw_data["zone"], use_var="unequal"
            )
            print(
                f"Welch's ANOVA (CW across zones): F={welch_cw.statistic:.4f}, p={welch_cw.pvalue:.4f}"
            )

            kw_cw_stat, kw_cw_p = stats.kruskal(*cw_groups)
            print(
                f"Kruskal-Wallis (CW across zones): H={kw_cw_stat:.4f}, p={kw_cw_p:.4f}"
            )
        else:
            print("Insufficient data for zone comparison within Constant_White.")
    else:
        print("No Constant_White data found.")

    # --- Two-Way Analysis: Policy × Zone ---
    print("\n" + "=" * 60)
    print("TWO-WAY ANALYSIS: POLICY × ZONE")
    print("=" * 60)

    try:
        model = ols("days_until_bolting ~ C(agent) * C(zone)", data=pdf).fit()

        print("\nStandard 2-Way ANOVA:")
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(anova_table)

        print("\nRobust 2-Way ANOVA (HC3):")
        anova_table_robust = sm.stats.anova_lm(model, typ=2, robust="hc3")
        print(anova_table_robust)

        # Effect sizes (Eta Squared)
        ss_total = anova_table["sum_sq"].sum()
        anova_table["eta_sq"] = anova_table["sum_sq"] / ss_total
        print("\n--- Eta Squared (Proportion of Variance Explained) ---")
        print(anova_table[["sum_sq", "eta_sq"]])

    except Exception as e:
        print(f"Error in two-way ANOVA: {e}")

    # --- Post-hoc: Policy × Zone combinations ---
    print("\n--- Post-hoc: Games-Howell (Policy × Zone) ---")
    try:
        pdf["group_zone"] = pdf["agent"] + " (Z" + pdf["zone"].astype(str) + ")"
        gh_results = pg.pairwise_gameshowell(
            data=pdf, dv="days_until_bolting", between="group_zone"
        )
        p_col = "pval" if "pval" in gh_results.columns else "p-unc"
        sig_gh = gh_results[gh_results[p_col] < 0.05]
        if not sig_gh.empty:
            print(f"Showing {len(sig_gh)} significant pairwise differences (p < 0.05):")
            cols = ["A", "B", "diff", "se", "T", "df", p_col]
            print(sig_gh[cols].to_string())
        else:
            print("No significant pairwise differences found at alpha=0.05")
    except Exception as e:
        print(f"Error running Games-Howell: {e}")

    # --- Visualization ---
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    agent_order = stats_by_agent["agent"].to_list()

    # Plot 1: Violin plot by Agent (Policy)
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(16, 8))

    sns.violinplot(
        data=pdf,
        x="agent",
        y="days_until_bolting",
        hue="agent",
        legend=False,
        palette="pastel",
        inner="box",
        order=agent_order,
    )

    sns.pointplot(
        data=pdf,
        x="agent",
        y="days_until_bolting",
        hue="agent",
        palette="dark",
        legend=False,
        order=agent_order,
        markers="_",
        capsize=0.2,
        alpha=0.5,
    )

    plt.xticks(rotation=45, ha="right")
    plt.title("Days Until Bolting by Policy (E16)")
    plt.xlabel("Policy")
    plt.ylabel("Days Until Bolting")
    plt.tight_layout()

    Path("results").mkdir(exist_ok=True)
    output_path_1 = "results/bolting_violinplot_by_policy.png"
    plt.savefig(output_path_1)
    print(f"Plot saved to '{output_path_1}'")

    # Plot 2: Violin plot by Zone, colored by Policy
    plt.figure(figsize=(16, 8))

    sns.violinplot(
        data=pdf,
        x="zone",
        y="days_until_bolting",
        hue="agent",
        palette="pastel",
        inner="box",
    )

    plt.title("Days Until Bolting by Zone and Policy (E16)")
    plt.xlabel("Zone")
    plt.ylabel("Days Until Bolting")
    plt.legend(title="Policy", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    output_path_2 = "results/bolting_violinplot_by_zone_policy.png"
    plt.savefig(output_path_2)
    print(f"Plot saved to '{output_path_2}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze days until bolting by policy and zone."
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default=f"/data/plant-rl/offline/{VERSION}/mixed-{VERSION}.parquet",
        help="Path to the parquet file",
    )
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        print(f"Error: Parquet file not found at {parquet_path}")
    else:
        analyze_bolting(parquet_path)
