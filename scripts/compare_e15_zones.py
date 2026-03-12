import polars as pl
import scipy.stats as stats
import numpy as np

# Load data
df = pl.read_parquet("/data/plant-rl/offline/v23/mixed-v23.parquet")

# Filter out rows after terminal state
# shift reward forwards by 1 over each plant trajectory
df_filtered = df.with_columns(
    pl.col("reward").shift(1).over(["experiment", "zone", "plant_id"])
)
df_filtered = df_filtered.filter(~pl.col("terminal"))

# Calculate total return for each plant trajectory
plant_returns = df_filtered.group_by(["experiment", "zone", "plant_id"]).agg(
    pl.col("reward").sum().alias("total_return")
)

# Filter for Experiment 15 and Zones 2, 3, 4
e15_data = plant_returns.filter(
    (pl.col("experiment") == 15) & (pl.col("zone").is_in([2, 3, 4]))
)

# Check which zones are present
present_zones = e15_data.select("zone").unique().sort("zone")["zone"].to_list()
print(f"Zones present for Experiment 15: {present_zones}")

if not present_zones:
    print("No data found for Experiment 15, Zones 2, 3, 4.")
else:
    # Prepare data for statistical test
    groups = []
    group_names = []
    for zone in present_zones:
        group_data = e15_data.filter(pl.col("zone") == zone)["total_return"].to_numpy()
        groups.append(group_data)
        group_names.append(f"Z{zone}")
        print(
            f"Zone {zone}: mean={np.mean(group_data):.4f}, std={np.std(group_data):.4f}, n={len(group_data)}"
        )

    if len(groups) < 2:
        print("Not enough groups for comparison.")
    else:
        # 1. Test for Normality (Shapiro-Wilk)
        print("\n--- Normality Test (Shapiro-Wilk) ---")
        for name, data in zip(group_names, groups):
            stat, p = stats.shapiro(data)
            print(
                f"{name}: stat={stat:.4f}, p-value={p:.4f} ({'Normal' if p > 0.05 else 'Not Normal'})"
            )

        # 2. Test for Homogeneity of Variance (Levene's test)
        print("\n--- Homogeneity of Variance (Levene) ---")
        stat, p = stats.levene(*groups)
        print(
            f"Levene's test: stat={stat:.4f}, p-value={p:.4f} ({'Homogeneous' if p > 0.05 else 'Not Homogeneous'})"
        )

        # 3. Perform Statistical Test
        # If all are normal and variances are equal, use ANOVA.
        # Otherwise, use Kruskal-Wallis (non-parametric).

        all_normal = all(stats.shapiro(g)[1] > 0.05 for g in groups)
        variances_equal = stats.levene(*groups)[1] > 0.05

        if all_normal and variances_equal:
            print("\n--- ANOVA ---")
            f_stat, p_val = stats.f_oneway(*groups)
            print(f"F-statistic: {f_stat:.4f}, p-value: {p_val:.4f}")
            if p_val < 0.05:
                print("Result: Significant difference between means (p < 0.05)")
                # Post-hoc test (Tukey's HSD)
                from statsmodels.stats.multicomp import pairwise_tukeyhsd

                # Flatten data for Tukey
                flat_data = np.concatenate(groups)
                flat_labels = np.concatenate(
                    [[name] * len(g) for name, g in zip(group_names, groups)]
                )
                tukey = pairwise_tukeyhsd(
                    endog=flat_data, groups=flat_labels, alpha=0.05
                )
                print("\nTukey HSD Post-hoc Test:")
                print(tukey)
            else:
                print("Result: No significant difference (p >= 0.05)")
        else:
            print("\n--- Kruskal-Wallis Test (Non-parametric) ---")
            h_stat, p_val = stats.kruskal(*groups)
            print(f"H-statistic: {h_stat:.4f}, p-value: {p_val:.4f}")
            if p_val < 0.05:
                print("Result: Significant difference between distributions (p < 0.05)")
                # Post-hoc test (Dunn's test)
                try:
                    import scikit_posthocs as sp

                    print("\nDunn's Post-hoc Test:")
                    posthoc = sp.posthoc_dunn(groups, p_adjust="holm")
                    posthoc.columns = group_names
                    posthoc.index = group_names
                    print(posthoc)
                except ImportError:
                    print(
                        "\nPost-hoc test (Dunn's) skipped because scikit-posthocs is not installed."
                    )
                    print(
                        "Performing pairwise Mann-Whitney U tests with Bonferroni correction instead:"
                    )
                    import itertools

                    pairs = list(itertools.combinations(range(len(groups)), 2))
                    num_comparisons = len(pairs)
                    for i, j in pairs:
                        u_stat, p_mw = stats.mannwhitneyu(groups[i], groups[j])
                        p_adj = min(1.0, p_mw * num_comparisons)
                        print(
                            f"{group_names[i]} vs {group_names[j]}: p-adj={p_adj:.4f}"
                        )
            else:
                print("Result: No significant difference (p >= 0.05)")
