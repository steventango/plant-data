import polars as pl
import numpy as np
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
    feature_cols = ["clean_area"] + action_cols + pca_cols

    # 1. Prepare Dynamics Data
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

    # 2. Train GP on N=3000 subset
    print("Training GP on N=3000 sample...")
    df_sample = df_dyn.sample(n=min(3000, len(df_dyn)), seed=42)
    X_train_raw = df_sample[feature_cols].to_numpy()
    X_train_raw[:, 0] = np.log(X_train_raw[:, 0])
    Y_train = df_sample[
        ["delta_ln_area"] + [f"delta_pca_{i}" for i in range(pca_dim)]
    ].to_numpy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)

    kernel = C(1.0) * Matern(
        length_scale=np.ones(X_train_scaled.shape[1]), nu=1.5
    ) + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 0.01))
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=3, normalize_y=True
    )
    gp.fit(X_train_scaled, Y_train)

    learned_scales = gp.kernel_.get_params()["k1__k2"].length_scale
    print(f"Learned Length Scales: {learned_scales}")

    # 3. Setup k-NN Proximity Diagnostic
    print("\nInitializing k-NN search on Scaled Training Space...")
    knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
    knn.fit(X_train_scaled)

    # 4. Analyze Trajectory Points (Plant 0)
    print("\nAnalyzing Trajectory Proximity for Plant 0:")
    traj_df = df_dyn.filter(pl.col("plant_id") == 0).sort("days")
    days_to_check = [0, 4, 8, 12]

    print(f"{'Day':<6} | {'Area':<8} | {'k-NN Dist':<10} | {'Sigma':<8}")
    print("-" * 45)

    for day in days_to_check:
        idx = (traj_df["days"] - day).abs().arg_min()
        row = traj_df[idx].to_dicts()[0]

        test_state_raw = np.hstack(
            [
                np.log(row["clean_area"]),
                row["red_coef"],
                row["white_coef"],
                row["blue_coef"],
                [row[f"pca_{k}"] for k in range(pca_dim)],
            ]
        ).reshape(1, -1)

        test_state_scaled = scaler.transform(test_state_raw)

        # Distance to neighbors
        dists, indices = knn.kneighbors(test_state_scaled)
        min_dist = dists[0, 0]

        # GP Prediction
        _, sigma = gp.predict(test_state_scaled, return_std=True)
        s = sigma[0, 0] if sigma.ndim > 1 else sigma[0]

        print(
            f"{row['days']:<6.1f} | {row['clean_area']:<8.1f} | {min_dist:<10.4f} | {s:<8.4f}"
        )

    # 5. Global Distance/Sigma Correlation
    print("\nMeasuring Global Sensitivity (N=500 random points)...")
    test_sample = df_dyn.sample(n=500, seed=123)
    X_test_raw = test_sample[feature_cols].to_numpy()
    X_test_raw[:, 0] = np.log(X_test_raw[:, 0])
    X_test_scaled = scaler.transform(X_test_raw)

    dists_all, _ = knn.kneighbors(X_test_scaled)
    min_dists_all = dists_all[:, 0]
    _, sigmas_all = gp.predict(X_test_scaled, return_std=True)
    s_all = sigmas_all[:, 0] if sigmas_all.ndim > 1 else sigmas_all

    corr = np.corrcoef(min_dists_all, s_all)[0, 1]
    print(f"Correlation (Proximity vs Uncertainty): {corr:.4f}")


if __name__ == "__main__":
    main()
