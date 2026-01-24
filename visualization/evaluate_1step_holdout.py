import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import os


def main():
    path = "/data/plant-rl/offline/v22/mixed-v22.parquet"
    print(f"Loading data from {path}...")
    df_raw = pl.read_parquet(path)

    pca_dim = 5
    df = df_raw.with_columns(
        (
            (
                pl.col("time")
                - pl.col("time").min().over("experiment", "zone", "plant_id")
            ).dt.total_seconds()
            / (24 * 3600)
        ).alias("days")
    ).filter((pl.col("days") <= 14) & pl.col("valid") & (pl.col("clean_area") > 0))

    df = df.with_columns(
        [pl.col("cls_token_pca").list.get(i).alias(f"pca_{i}") for i in range(pca_dim)]
    )

    action_cols = ["red_coef", "white_coef", "blue_coef"]
    pca_cols = [f"pca_{i}" for i in range(pca_dim)]
    df = df.sort("experiment", "zone", "plant_id", "time")

    # 1. Split Data: Hold-out E14 Z1
    train_mask = ~((pl.col("experiment") == 14) & (pl.col("zone") == 1))
    test_mask = (pl.col("experiment") == 14) & (pl.col("zone") == 1)

    # Prepare dynamics targets (deltas)
    df_dyn = df.with_columns(
        [
            (
                pl.col("clean_area")
                .log()
                .shift(-1)
                .over("experiment", "zone", "plant_id")
                - pl.col("clean_area").log()
            ).alias("delta_ln_area"),
        ]
        + [
            (
                pl.col(f"pca_{i}").shift(-1).over("experiment", "zone", "plant_id")
                - pl.col(f"pca_{i}")
            ).alias(f"delta_pca_{i}")
            for i in range(pca_dim)
        ]
    ).filter(pl.col("delta_ln_area").is_not_null())

    df_train = df_dyn.filter(train_mask)
    df_test = df_dyn.filter(test_mask)

    print(f"Train samples: {len(df_train)}, Test samples (E14/Z1): {len(df_test)}")

    feature_cols = ["clean_area"] + action_cols + pca_cols
    target_cols = ["delta_ln_area"] + [f"delta_pca_{i}" for i in range(pca_dim)]

    # Subsample training for speed
    df_train_sample = df_train.sample(n=min(1500, len(df_train)), seed=42)

    X_raw_train = df_train_sample[feature_cols].to_numpy()
    X_raw_train[:, 0] = np.log(X_raw_train[:, 0])  # ln(Area)
    Y_train = df_train_sample[target_cols].to_numpy()

    scaler_x = StandardScaler()
    X_scaled_train = scaler_x.fit_transform(X_raw_train)

    # 2. Train GP
    print("Training GP on all data EXCEPT E14/Z1 (Constrained Noise: 0.01)...")
    kernel = C(1.0) * Matern(
        length_scale=np.ones(X_scaled_train.shape[1]), nu=1.5
    ) + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 0.01))
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=5, normalize_y=True
    )
    gp.fit(X_scaled_train, Y_train)

    # --- NEW: Inspect Kernel Parameters ---
    print("\nKernel Param Inspection:")
    kernel_params = gp.kernel_.get_params()
    # Matern is typically nested under ConstantKernel
    m_kernel = kernel_params["k1__k2"]
    w_kernel = kernel_params["k2"]

    print(f"Learned Length Scales: {m_kernel.length_scale}")
    print(f"  (Order: Area, R, W, B, PCA_0, PCA_1, PCA_2, PCA_3, PCA_4)")
    print(f"Learned Noise Level: {w_kernel.noise_level:.6f}")

    # --- NEW: Data Density Analysis (k-NN) ---
    print("\nAnalyzing Uncertainty vs Data Density...")
    nn = NearestNeighbors(n_neighbors=5).fit(X_scaled_train)

    # Calculate density for a subset of training data to save time
    density_sample_idx = np.random.choice(
        len(X_scaled_train), min(500, len(X_scaled_train)), replace=False
    )
    X_density_sample = X_scaled_train[density_sample_idx]

    distances, _ = nn.kneighbors(X_density_sample)
    avg_dist = np.mean(distances, axis=1)  # Proxy for "OOD-ness"

    _, sigma_sample = gp.predict(X_density_sample, return_std=True)
    sigma_log_area = sigma_sample[:, 0] if sigma_sample.ndim > 1 else sigma_sample

    corr = np.corrcoef(avg_dist, sigma_log_area)[0, 1]
    print(f"Correlation between k-NN Distance and Sigma: {corr:.4f}")

    # --- NEW: Out-of-Distribution (OOD) Stress Test ---
    print("\nOOD Stress Test (Extreme Area):")
    # Area = 10k (way above training range)
    ood_raw = np.zeros((1, len(feature_cols)))
    ood_raw[0, 0] = 50000.0  # Extreme Area
    # Standard PCA/Actions
    ood_raw[0, 1:4] = [0.0, 1.0, 0.0]

    # Scale it
    ood_raw_log = ood_raw.copy()
    ood_raw_log[0, 0] = np.log(ood_raw[0, 0])
    ood_scaled = scaler_x.transform(ood_raw_log)

    _, ood_sigma = gp.predict(ood_scaled, return_std=True)
    ood_s = ood_sigma[0, 0] if ood_sigma.ndim > 1 else ood_sigma[0]
    print(f"Sigma at Area=98 (In-dist): ~{np.mean(sigma_log_area):.4f}")
    print(f"Sigma at Area=50,000 (OOD): {ood_s:.4f}")

    # 3. Select 10 Random Plants from Test Set
    unique_test_plants = df_test.select(["experiment", "zone", "plant_id"]).unique()
    sampled_plants = unique_test_plants.sample(
        n=min(10, len(unique_test_plants)), seed=42
    ).to_dicts()

    os.makedirs("results/gp_holdout", exist_ok=True)

    # We will create two multi-page or multi-grid plots: Raw and Log
    fig_raw, axes_raw = plt.subplots(2, 5, figsize=(25, 12))
    fig_log, axes_log = plt.subplots(2, 5, figsize=(25, 12))
    axes_raw = axes_raw.flatten()
    axes_log = axes_log.flatten()

    total_points = 0
    points_in_ci = 0

    for i, plant_row in enumerate(sampled_plants):
        plant_df = df_test.filter(
            (pl.col("experiment") == plant_row["experiment"])
            & (pl.col("zone") == plant_row["zone"])
            & (pl.col("plant_id") == plant_row["plant_id"])
        ).sort("days")

        real_days = plant_df["days"].to_numpy()
        real_area = plant_df["clean_area"].to_numpy()
        real_ln_area = np.log(real_area)

        # 1-Step Predictions
        X_test_plant = plant_df[feature_cols].to_numpy()
        X_test_plant[:, 0] = np.log(X_test_plant[:, 0])  # ln(Area_t)
        X_scaled_test = scaler_x.transform(X_test_plant)

        mu_delta, sigma_delta = gp.predict(X_scaled_test, return_std=True)

        # Predicted ln(Area_t+1) = ln(Area_t) + mu_delta
        mu_ln_next = X_test_plant[:, 0] + mu_delta[:, 0]
        s_ln = sigma_delta[:, 0] if sigma_delta.ndim > 1 else sigma_delta

        # Raw space stats (Log-Normal)
        pred_mean_raw = np.exp(mu_ln_next + (s_ln**2) / 2)
        pred_lower_raw = np.exp(mu_ln_next - 2 * s_ln)
        pred_upper_raw = np.exp(mu_ln_next + 2 * s_ln)

        # Log space stats (Normal)
        pred_mean_log = mu_ln_next
        pred_lower_log = mu_ln_next - 2 * s_ln
        pred_upper_log = mu_ln_next + 2 * s_ln

        # Ground truth next area
        real_next_area = np.exp(
            np.log(real_area) + plant_df["delta_ln_area"].to_numpy()
        )
        real_next_ln_area = np.log(real_next_area)

        # Coverage calculation
        total_points += len(real_next_area)
        points_in_ci += np.sum(
            (real_next_area >= pred_lower_raw) & (real_next_area <= pred_upper_raw)
        )

        t_plot = real_days + 1

        # RAW Plot
        axes_raw[i].plot(
            real_days, real_area, "b-o", markersize=3, label="Actual Area_t", alpha=0.5
        )
        axes_raw[i].errorbar(
            t_plot,
            pred_mean_raw,
            yerr=[pred_mean_raw - pred_lower_raw, pred_upper_raw - pred_mean_raw],
            fmt="r^",
            label="1-Step Pred",
            capsize=3,
            markersize=4,
            alpha=0.7,
        )
        axes_raw[i].scatter(
            t_plot,
            real_next_area,
            color="black",
            marker="x",
            label="Actual Area_t+1",
            s=20,
            zorder=5,
        )
        axes_raw[i].set_title(f"Plant {plant_row['plant_id']}")
        axes_raw[i].grid(True, alpha=0.2)

        # LOG Plot
        axes_log[i].plot(
            real_days,
            real_ln_area,
            "b-o",
            markersize=3,
            label="Actual ln(Area)_t",
            alpha=0.5,
        )
        axes_log[i].errorbar(
            t_plot,
            pred_mean_log,
            yerr=2 * s_ln,
            fmt="r^",
            label="1-Step Pred",
            capsize=3,
            markersize=4,
            alpha=0.7,
        )
        axes_log[i].scatter(
            t_plot,
            real_next_ln_area,
            color="black",
            marker="x",
            label="Actual ln(Area)_t+1",
            s=20,
            zorder=5,
        )
        axes_log[i].set_title(f"Plant {plant_row['plant_id']}")
        axes_log[i].grid(True, alpha=0.2)

    coverage = (points_in_ci / total_points) * 100

    fig_raw.suptitle(
        f"GP 1-Step Prediction Coverage (E14/Z1 Hold-out): {coverage:.1f}% points in 95% CI",
        fontsize=16,
    )
    fig_raw.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_raw.savefig("results/gp_holdout_1step_coverage.png", dpi=150)

    fig_log.suptitle(
        f"GP 1-Step Prediction Coverage (LOG SPACE): Relative uncertainty visualization",
        fontsize=16,
    )
    fig_log.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_log.savefig("results/gp_holdout_1step_coverage_log.png", dpi=150)

    plt.close("all")

    print(f"\nEvaluation Complete. Coverage: {coverage:.1f}%")


if __name__ == "__main__":
    main()
