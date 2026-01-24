import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
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

    feature_cols = ["clean_area"] + action_cols + pca_cols
    target_cols = ["delta_ln_area"] + [f"delta_pca_{i}" for i in range(pca_dim)]

    # --- Setup Iteration Parameters ---
    noise_bounds_list = [
        (1e-5, 1.0),  # Baseline (unconstrained noise)
        (1e-5, 0.1),  # Constrained
        (1e-5, 0.01),  # Highly Constrained (force epistemic signal)
    ]
    train_sizes = [1500, 500, 100]  # Fewer points = bigger knowledge gaps

    results_dir = "results/gp_epistemic_search"
    os.makedirs(results_dir, exist_ok=True)

    # We will plot Sigma vs Area for each combination
    fig, axes = plt.subplots(
        len(noise_bounds_list),
        len(train_sizes),
        figsize=(20, 15),
        sharex=True,
        sharey=False,
    )

    # Area sweep range for visualization (Test OOD behavior)
    area_sweep_raw = np.linspace(10, 10000, 200)  # Training max is around 3k
    ln_area_sweep = np.log(area_sweep_raw)

    for i, noise_bounds in enumerate(noise_bounds_list):
        for j, n_train in enumerate(train_sizes):
            print(f"Testing Noise: {noise_bounds}, Size: {n_train}...")

            df_sample = df_dyn.sample(n=min(n_train, len(df_dyn)), seed=42)
            X_raw = df_sample[feature_cols].to_numpy()
            X_raw[:, 0] = np.log(X_raw[:, 0])  # ln(Area)
            Y_delta = df_sample[target_cols].to_numpy()

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_raw)

            kernel = C(1.0) * Matern(
                length_scale=np.ones(X_scaled.shape[1]), nu=1.5
            ) + WhiteKernel(noise_level=0.01, noise_level_bounds=noise_bounds)
            gp = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=3, normalize_y=True
            )
            gp.fit(X_scaled, Y_delta)

            # Learned Noise Level
            learned_noise = gp.kernel_.get_params()["k2"].noise_level

            # Predict sweep
            # Features: [ln_area, 0, 1, 0, 0, 0, 0, 0, 0] (Mean white agent)
            X_sweep_raw = np.zeros((len(area_sweep_raw), len(feature_cols)))
            X_sweep_raw[:, 0] = ln_area_sweep
            X_sweep_raw[:, 2] = 1.0  # White light constant
            # Mean PCA values from sample
            for k in range(pca_dim):
                X_sweep_raw[:, 4 + k] = np.mean(X_raw[:, 4 + k])

            X_sweep_scaled = scaler.transform(X_sweep_raw)
            _, sigma_sweep = gp.predict(X_sweep_scaled, return_std=True)

            s = sigma_sweep[:, 0] if sigma_sweep.ndim > 1 else sigma_sweep

            ax = axes[i, j]
            ax.plot(area_sweep_raw, s, color="purple", lw=2)
            ax.axvline(
                X_raw[:, 0].max(),
                color="red",
                ls="--",
                alpha=0.3,
                label="Max Train ln(A)",
            )  # Wait, this is ln space
            ax.axvline(np.exp(X_raw[:, 0].max()), color="red", ls="--", alpha=0.5)

            ax.set_title(
                f"Size={n_train}, NoiseLimit={noise_bounds[1]}\nLearned Noise={learned_noise:.3f}"
            )
            if i == len(noise_bounds_list) - 1:
                ax.set_xlabel("Plant Area (mm^2)")
            if j == 0:
                ax.set_ylabel("Predictive Sigma")
            ax.grid(True, alpha=0.2)

    plt.suptitle(
        "GP Uncertainty Sensitivity: Searching for visible Epistemic Signals\nHorizontal Spike indicates transition to Out-of-Distribution",
        fontsize=18,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{results_dir}/epistemic_search_grid.png", dpi=150)
    plt.close()

    print(
        "\nIteration complete. View results in results/gp_epistemic_search/epistemic_search_grid.png"
    )


if __name__ == "__main__":
    main()
