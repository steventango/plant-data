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
    df_all = pl.read_parquet(path)

    pca_dim = 5
    df = df_all.with_columns(
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

    # --- 1. Train State-Reactive GP (Constrained Noise) ---
    print("Training State-Reactive GP (Noise floor: 0.01)...")
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

    df_sample = df_dyn.sample(n=min(1500, len(df_dyn)), seed=42)
    X_raw = df_sample[feature_cols].to_numpy()
    X_raw[:, 0] = np.log(X_raw[:, 0])
    Y_delta = df_sample[target_cols].to_numpy()

    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X_raw)

    # CONSTRAINED NOISE: Forces kernel to explain variance
    kernel = C(1.0) * Matern(
        length_scale=np.ones(X_scaled.shape[1]), nu=1.5
    ) + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 0.01))
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=5, normalize_y=True
    )
    gp.fit(X_scaled, Y_delta)

    learned_noise = gp.kernel_.get_params()["k2"].noise_level
    print(f"Learned Noise Level: {learned_noise:.4f}")

    # --- 2. Monte Carlo Rollouts ---
    # Setup initial state from overall mean
    initial_area = df.filter(pl.col("days") < 1)["clean_area"].mean()
    initial_pca = [
        df.filter(pl.col("days") < 1)[f"pca_{i}"].mean() for i in range(pca_dim)
    ]

    n_steps = 14
    n_paths = 100
    white_policy = [0.0, 1.0, 0.0]
    red_policy = [1.0, 0.0, 0.0]

    def mc_rollout_delta(policy):
        paths_area = np.zeros((n_paths, n_steps + 1))
        paths_area[:, 0] = initial_area
        current_states = np.zeros((n_paths, 1 + pca_dim))
        current_states[:, 0] = np.log(initial_area)
        current_states[:, 1:] = initial_pca

        sigmas_over_time = []  # To track average predictive sigma per step

        for t in range(n_steps):
            act = np.tile(policy, (n_paths, 1))
            inp_raw = np.hstack([current_states[:, :1], act, current_states[:, 1:]])
            inp_scaled = scaler_x.transform(inp_raw)
            mu_delta, sigma_delta = gp.predict(inp_scaled, return_std=True)

            s = sigma_delta[:, 0] if sigma_delta.ndim > 1 else sigma_delta
            sigmas_over_time.append(np.mean(s))

            noise = np.random.normal(0, 1, size=(n_paths, 1 + pca_dim)) * (
                sigma_delta if sigma_delta.ndim > 1 else sigma_delta[:, np.newaxis]
            )
            current_states += mu_delta + noise
            paths_area[:, t + 1] = np.exp(current_states[:, 0])

        return paths_area, np.array(sigmas_over_time)

    print("Simulating rollouts...")
    w_paths, w_sigmas = mc_rollout_delta(white_policy)
    r_paths, r_sigmas = mc_rollout_delta(red_policy)

    # --- 3. Visualization ---
    os.makedirs("results/gp_rollouts_reactive", exist_ok=True)
    steps = np.arange(n_steps + 1)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Subplot 1: Area Rollout
    w_m, w_l, w_u = (
        np.mean(w_paths, axis=0),
        np.percentile(w_paths, 2.5, axis=0),
        np.percentile(w_paths, 97.5, axis=0),
    )
    r_m, r_l, r_u = (
        np.mean(r_paths, axis=0),
        np.percentile(r_paths, 2.5, axis=0),
        np.percentile(r_paths, 97.5, axis=0),
    )

    axes[0].plot(steps, w_m, "g-", lw=3, label="White Policy Mean")
    axes[0].fill_between(
        steps, w_l, w_u, color="green", alpha=0.15, label="White 95% CI"
    )
    axes[0].plot(steps, r_m, "r-", lw=3, label="Red Policy Mean")
    axes[0].fill_between(steps, r_l, r_u, color="red", alpha=0.15, label="Red 95% CI")
    axes[0].set_title(
        "MC Rollouts: Constrained Noise (0.01)\nComparing Growth Trajectories"
    )
    axes[0].set_xlabel("Days")
    axes[0].set_ylabel("Area (mm^2)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    # Subplot 2: Predictive Sigma (State-Dependent Sensitivity)
    axes[1].plot(
        steps[1:], w_sigmas, "g-o", lw=2, markersize=4, label="White Policy Sigma"
    )
    axes[1].plot(
        steps[1:], r_sigmas, "r-o", lw=2, markersize=4, label="Red Policy Sigma"
    )
    axes[1].set_title(
        "Predictive Uncertainty ($\sigma$) over Time\nForced state-reactivity highlights model knowledge gaps"
    )
    axes[1].set_xlabel("Days")
    axes[1].set_ylabel("Mean Predictive Sigma (Log-Space Delta)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(
        "results/gp_rollouts_reactive/reactive_uncertainty_comparison.png", dpi=150
    )
    plt.close()

    print(
        "\nState-Reactive analysis complete. View results in results/gp_rollouts_reactive/"
    )


if __name__ == "__main__":
    main()
