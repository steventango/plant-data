import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os


def main():
    path = "/data/plant-rl/offline/v22/mixed-v22.parquet"
    df = pl.read_parquet(path)

    # Filter to 14 days and valid data
    df = df.with_columns(
        (
            (
                pl.col("time")
                - pl.col("time").min().over("experiment", "zone", "plant_id")
            ).dt.total_seconds()
            / (24 * 3600)
        ).alias("days")
    ).filter((pl.col("days") <= 14) & pl.col("valid") & (pl.col("clean_area") > 0))

    # Sort and shift to get next area
    df = df.sort("experiment", "zone", "plant_id", "time")
    df = df.with_columns(
        pl.col("clean_area")
        .shift(-1)
        .over("experiment", "zone", "plant_id")
        .alias("next_area")
    ).filter(pl.col("next_area").is_not_null())

    area = df["clean_area"].to_numpy()
    next_area = df["next_area"].to_numpy()

    log_area = np.log(area)
    log_next_area = np.log(next_area)

    os.makedirs("results/gp_analysis", exist_ok=True)

    # 1. Raw Space Analysis
    model_raw = LinearRegression().fit(area.reshape(-1, 1), next_area)
    preds_raw = model_raw.predict(area.reshape(-1, 1))
    resid_raw = next_area - preds_raw
    r2_raw = r2_score(next_area, preds_raw)

    # 2. Log Space Analysis
    model_log = LinearRegression().fit(log_area.reshape(-1, 1), log_next_area)
    preds_log = model_log.predict(log_area.reshape(-1, 1))
    resid_log = log_next_area - preds_log
    r2_log = r2_score(log_next_area, preds_log)

    print(f"R-squared (Raw Space): {r2_raw:.6f}")
    print(f"R-squared (Log Space): {r2_log:.6f}")

    # Check for heteroscedasticity (residuals vs input)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Raw Scatter
    axes[0, 0].scatter(area, next_area, alpha=0.1, s=1)
    axes[0, 0].set_title(f"Next Area vs Current Area (Raw)\nR2: {r2_raw:.4f}")
    axes[0, 0].set_xlabel("Area_t")
    axes[0, 0].set_ylabel("Area_{t+1}")

    # Log Scatter
    axes[0, 1].scatter(log_area, log_next_area, alpha=0.1, s=1)
    axes[0, 1].set_title(f"Next Log Area vs Current Log Area\nR2: {r2_log:.4f}")
    axes[0, 1].set_xlabel("Log Area_t")
    axes[0, 1].set_ylabel("Log Area_{t+1}")

    # Raw Residuals
    axes[1, 0].scatter(area, resid_raw, alpha=0.1, s=1)
    axes[1, 0].axhline(0, color="r", linestyle="--")
    axes[1, 0].set_title("Raw Residuals (Showing Heteroscedasticity)")
    axes[1, 0].set_xlabel("Area_t")
    axes[1, 0].set_ylabel("Residual")

    # Log Residuals
    axes[1, 1].scatter(log_area, resid_log, alpha=0.1, s=1)
    axes[1, 1].axhline(0, color="r", linestyle="--")
    axes[1, 1].set_title("Log Residuals (More Homoscedastic)")
    axes[1, 1].set_xlabel("Log Area_t")
    axes[1, 1].set_ylabel("Residual")

    plt.tight_layout()
    plt.savefig("results/gp_analysis/residual_comparison.png", dpi=150)
    plt.close()

    # Variance check
    print(f"\nStandard Deviation of Residuals:")
    print(f"Raw: {np.std(resid_raw):.4f}")
    print(f"Log: {np.std(resid_log):.4f}")

    # Coefficient of variation check (residual std / scale)
    print(f"\nNormalized Residual Spread (std/mean):")
    print(f"Raw: {np.std(resid_raw) / np.mean(next_area):.4f}")
    print(f"Log: {np.std(resid_log) / np.mean(log_next_area):.4f}")


if __name__ == "__main__":
    main()
