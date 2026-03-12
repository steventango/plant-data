import polars as pl
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statsmodels.stats.oneway import anova_oneway
import pingouin as pg


def main():
    # Use the latest version if available, else v23 for testing script structure
    v24_path = Path("/data/plant-rl/offline/v24/mixed-v24.parquet")
    if v24_path.exists():
        data_path = v24_path
    else:
        raise FileNotFoundError("v24 data not found")

    print(f"Loading data from {data_path}")
    df = pl.read_parquet(data_path)

    # Filter for valid data (truncated to wall_time <= 13) and calculate plant returns
    df_filtered = df.filter((~pl.col("bolted_pred")) & (pl.col("wall_time") <= 20))
    plant_returns = (
        df_filtered.group_by(["experiment", "zone", "plant_id", "agent"])
        .agg(pl.col("reward").sum().alias("total_return"))
        .to_pandas()
    )

    if 16 in plant_returns["experiment"].values:
        print("Analyzing Experiment 16")
        e16_data = plant_returns[plant_returns["experiment"] == 16].copy()

        # Define factors
        # Policy categories: Constant_White, InAC_Seed6, InAC_Seed7, InAC_Seed21
        e16_data["policy_group"] = e16_data["agent"]

        # Filter out outlier plants that didn't grow (e.g., total_return < 0.2)
        initial_count = len(e16_data)
        e16_data = e16_data[e16_data["total_return"] > 0.2]
        removed_count = initial_count - len(e16_data)
        if removed_count > 0:
            print(
                f"Filtered out {removed_count} outlier plants with low return (< 0.2)"
            )

        # A. Constant White Analysis across zones
        cw_data = e16_data[e16_data["agent"] == "Constant_White"]
        if not cw_data.empty:
            print("\n--- A. Constant White Mean across Zones ---")
            zones = cw_data["zone"].unique()
            groups = [cw_data[cw_data["zone"] == z]["total_return"] for z in zones]
            f_stat, p_val = stats.f_oneway(*groups)
            print(
                f"Standard ANOVA for Constant White across zones {zones}: F={f_stat:.4f}, p={p_val:.4f}"
            )

            # Welch's ANOVA
            welch_a = anova_oneway(
                cw_data["total_return"], cw_data["zone"], use_var="unequal"
            )
            print(
                f"Welch's ANOVA for Constant White across zones: F={welch_a.statistic:.4f}, p={welch_a.pvalue:.4f}"
            )

            # Post-hoc if significant (using standard p_val for now or welch p_val)
            if welch_a.pvalue < 0.05:
                tukey = pairwise_tukeyhsd(
                    cw_data["total_return"], cw_data["zone"], alpha=0.05
                )
                print(tukey)

        # B. Two-Factor Analysis (Policy and Zone)
        print("\n--- B. Two-Factor Analysis (Policy and Zone) ---")
        # Factorial model (with interaction)
        model = ols("total_return ~ C(policy_group) * C(zone)", data=e16_data).fit()

        print("Standard 2-Way ANOVA:")
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(anova_table)

        print("\nRobust 2-Way ANOVA (HC3 - Robust to unequal variances):")
        # This is the "2-way Welch" equivalent using heteroscedasticity-robust standard errors
        anova_table_robust = sm.stats.anova_lm(model, typ=2, robust="hc3")
        print(anova_table_robust)

        # Calculate Effect Size (Eta Squared) on robust table if needed, or standard
        ss_total = anova_table["sum_sq"].sum()
        anova_table["eta_sq"] = anova_table["sum_sq"] / ss_total
        print("\n--- Proportion of Variance Explained (Eta Squared) ---")
        print(anova_table[["sum_sq", "eta_sq"]])

        # C. Assumption Testing
        print("\n--- C. Assumption Testing ---")
        # 1. Normality (Shapiro-Wilk)
        w, p = stats.shapiro(model.resid)
        print(f"Shapiro-Wilk test for normality: W={w:.4f}, p={p:.4f}")

        # 2. Homogeneity of Variance (Levene's)
        # Testing across policy groups
        groups_policy = [
            e16_data[e16_data["policy_group"] == p]["total_return"]
            for p in e16_data["policy_group"].unique()
        ]
        stat, p_levene_policy = stats.levene(*groups_policy)
        print(f"Levene's test (Policy): stat={stat:.4f}, p={p_levene_policy:.4f}")

        # D. Welch's ANOVA (Robust to unequal variances)
        print("\n--- D. Welch's ANOVA ---")
        welch_policy = anova_oneway(
            e16_data["total_return"], e16_data["policy_group"], use_var="unequal"
        )
        print(
            f"Welch's ANOVA across policies: F={welch_policy.statistic:.4f}, p={welch_policy.pvalue:.4f}"
        )

        # E. Non-parametric Analysis (Robust to assumption violations)
        print("\n--- E. Robust Analysis (Kruskal-Wallis) ---")
        kw_stat, kw_p = stats.kruskal(*groups_policy)
        print(f"Kruskal-Wallis test across policies: H={kw_stat:.4f}, p={kw_p:.4f}")

        # F. Robust Post-hoc (Games-Howell)
        print(
            "\n--- F. Robust Post-hoc (Games-Howell) for Policy x Zone Combinations ---"
        )
        e16_data["group_zone"] = (
            e16_data["policy_group"] + " (Z" + e16_data["zone"].astype(str) + ")"
        )

        try:
            gh_results = pg.pairwise_gameshowell(
                data=e16_data, dv="total_return", between="group_zone"
            )
            # Filter for significant results to keep output clean
            # Note: Pingouin uses 'pval' in recent versions
            p_col = "pval" if "pval" in gh_results.columns else "p-unc"
            m_a = "mean_A" if "mean_A" in gh_results.columns else "mean(A)"
            m_b = "mean_B" if "mean_B" in gh_results.columns else "mean(B)"

            sig_gh = gh_results[gh_results[p_col] < 0.05]
            if not sig_gh.empty:
                print(
                    f"Showing {len(sig_gh)} significant pairwise differences (p < 0.05):"
                )
                cols = ["A", "B", m_a, m_b, "diff", "se", "T", "df", p_col]
                print(sig_gh[cols])
            else:
                print("No significant pairwise differences found at alpha=0.05")

            # Also provide the policy-only GH for comparison to previous section
            print("\n--- Games-Howell Post-hoc (Policy main effect) ---")
            gh_policy = pg.pairwise_gameshowell(
                data=e16_data, dv="total_return", between="policy_group"
            )
            p_col_pol = "pval" if "pval" in gh_policy.columns else "p-unc"
            print(gh_policy[["A", "B", "diff", p_col_pol]])

        except Exception as e:
            print(f"\nError running Games-Howell post-hoc: {e}")

        # Visualization
        plt.figure(figsize=(12, 6))
        sns.boxplot(x="zone", y="total_return", hue="policy_group", data=e16_data)
        plt.title("Total Return by Zone and Policy (E16)")
        plt.savefig("results/e16_policy_zone_analysis.png")
        print("\nBoxplot saved to results/e16_policy_zone_analysis.png")

    else:
        print("Experiment 16 data not found in the dataset.")


if __name__ == "__main__":
    main()
