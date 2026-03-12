import polars as pl
import scipy.stats as stats
import numpy as np
import scikit_posthocs as sp
from statsmodels.stats.oneway import anova_oneway

# Load data
df = pl.read_parquet("/data/plant-rl/offline/v23/mixed-v23.parquet")
df_filtered = df.with_columns(
    pl.col("reward").shift(1).over(["experiment", "zone", "plant_id"])
)
df_filtered = df_filtered.filter(~pl.col("terminal"))
plant_returns = df_filtered.group_by(["experiment", "zone", "plant_id"]).agg(
    pl.col("reward").sum().alias("total_return")
)

# Extract Groups
z2 = plant_returns.filter((pl.col("experiment") == 15) & (pl.col("zone") == 2))[
    "total_return"
].to_numpy()
z3 = plant_returns.filter((pl.col("experiment") == 15) & (pl.col("zone") == 3))[
    "total_return"
].to_numpy()
z4 = plant_returns.filter((pl.col("experiment") == 15) & (pl.col("zone") == 4))[
    "total_return"
].to_numpy()

print("--- Testing the Arithmetic Mean ---")

# 1. Alexander-Govern Test
ag_test = stats.alexandergovern(z2, z3, z4)
print(f"Alexander-Govern test (p-value): {ag_test.pvalue:.8f}")

# 2. Welch's ANOVA
# anova_oneway handles the unequal variance (Welch's correction)
welch_test = anova_oneway([z2, z3, z4], use_var="unequal")
print(f"Welch's ANOVA (p-value):         {welch_test.pvalue:.8f}")

# 3. Post-hoc for Means (Tamhane's T2)
print("\n--- Tamhane's T2 Post-hoc (Means, Unequal Variance) ---")

combined_data = [z2, z3, z4]
# Tamhane's T2 is robust to unequal variances
gh_test = sp.posthoc_tamhane(combined_data)
gh_test.index = ["Z2", "Z3", "Z4"]
gh_test.columns = ["Z2", "Z3", "Z4"]
print(gh_test)


# 4. Permutation Test (Non-parametric mean comparison)
# Useful for verifying the above parametric results
def get_p_perm(a, b):
    res = stats.permutation_test(
        (a, b),
        lambda x, y: np.mean(x) - np.mean(y),
        permutation_type="independent",
        vectorized=False,
        n_resamples=10000,
    )
    return res.pvalue


print("\n--- Permutation Test (Direct Mean Comparison) ---")
print(f"Z2 vs Z3: p = {get_p_perm(z2, z3):.4f}")
print(f"Z2 vs Z4: p = {get_p_perm(z2, z4):.4f}")
print(f"Z3 vs Z4: p = {get_p_perm(z3, z4):.4f}")
