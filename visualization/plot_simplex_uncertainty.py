import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import mpltern
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans


def main():
    # 1. Load Full Data
    path = "/data/plant-rl/offline/v22/mixed-v22.parquet"
    print(f"Loading full dataset from {path}...")
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
    df = df.sort("experiment", "zone", "plant_id", "time")

    action_cols = ["red_coef", "white_coef", "blue_coef"]
    pca_cols = [f"pca_{i}" for i in range(pca_dim)]
    feature_cols = ["clean_area"] + action_cols + pca_cols

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
    ).filter(
        pl.col("delta_ln_area").is_not_null() & pl.col("delta_ln_area").is_finite()
    )

    X_raw = df_dyn[feature_cols].to_numpy()
    X_raw[:, 0] = np.log(X_raw[:, 0])  # ln(Area)
    Y_raw = df_dyn["delta_ln_area"].to_numpy().reshape(-1, 1)

    print(f"Original dataset size: {len(X_raw)}")

    # 2. Centroid-Summarized GP for Robust Stability
    print("Summarizing 25k dataset into 3,000 representative centroids using KMeans...")
    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_raw)

    kmeans = MiniBatchKMeans(n_clusters=3000, random_state=42, n_init=3)
    kmeans.fit(X_full_scaled)
    X_centroid = kmeans.cluster_centers_

    # Use closest real samples for stable labels
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=1).fit(X_full_scaled)
    _, idx = nn.kneighbors(X_centroid)
    X_train = X_full_scaled[idx.flatten()]
    Y_train = Y_raw[idx.flatten()]

    print(f"Training Standard GP on {len(X_train)} summarized samples...")
    kernel = C(1.0) * Matern(
        length_scale=np.ones(X_train.shape[1]), nu=1.5
    ) + WhiteKernel(noise_level=0.01)
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=2, normalize_y=True
    )
    gp.fit(X_train, Y_train)

    # Diagnostic: Length Scales
    ls = gp.kernel_.get_params()["k1__k2"].length_scale
    print("\nLearned Length Scales (Scaled Space):")
    print(f"  ln(Area): {ls[0]:.4f}")
    print(f"  RED action: {ls[1]:.4f}")
    print(f"  WHITE action: {ls[2]:.4f}")
    print(f"  BLUE action: {ls[3]:.4f}")
    print(f"  PCA Dims: {ls[4:]}")

    # 3. Smooth Evolutionary Simplex Analysis
    traj_df = df_dyn.filter(pl.col("plant_id") == 0).sort("days")
    days_to_plot = [0, 4, 8, 12]

    # Higher resolution for continuous shading
    n_grid = 100
    t_v = np.linspace(0, 1, n_grid)
    l_v = np.linspace(0, 1, n_grid)
    T, L = np.meshgrid(t_v, l_v)
    R_g = 1.0 - T - L
    mask = R_g >= -1e-10
    tm, lm, rm = T[mask], L[mask], R_g[mask]

    fig = plt.figure(figsize=(24, 8))
    for i, day in enumerate(days_to_plot):
        idx_t = (traj_df["days"] - day).abs().arg_min()
        row = traj_df[idx_t].to_dicts()[0]
        ln_area = np.log(row["clean_area"])
        pca_v = [row[f"pca_{k}"] for k in range(pca_dim)]
        real_act = [row["red_coef"], row["white_coef"], row["blue_coef"]]
        s_act = sum(real_act)
        real_act = [a / s_act for a in real_act]

        X_grid = np.zeros((len(tm), len(feature_cols)))
        X_grid[:, 0] = ln_area
        X_grid[:, 1], X_grid[:, 2], X_grid[:, 3] = lm, tm, rm
        for k in range(pca_dim):
            X_grid[:, 4 + k] = pca_v[k]
        X_grid_scaled = scaler.transform(X_grid)

        _, sigmas = gp.predict(X_grid_scaled, return_std=True)
        s_raw = sigmas.reshape(-1)

        ax = fig.add_subplot(1, 4, i + 1, projection="ternary")

        # Smooth Contouring instead of Hexbin
        levels = np.linspace(s_raw.min(), s_raw.max(), 50)
        cnt = ax.tricontourf(tm, lm, rm, s_raw, levels=levels, cmap="magma")

        # Overlay Real Action
        ax.scatter(
            real_act[1],
            real_act[0],
            real_act[2],
            color="white",
            marker="*",
            s=300,
            edgecolors="black",
            label="Observed Action",
            zorder=10,
        )

        ax.set_tlabel("WHITE", fontsize=11, fontweight="bold")
        ax.set_llabel("RED", fontsize=11, fontweight="bold")
        ax.set_rlabel("BLUE", fontsize=11, fontweight="bold")
        ax.set_title(
            f"Day {row['days']:.1f}\nArea {row['clean_area']:.0f} mm^2",
            fontsize=15,
            pad=25,
        )

        if i == 0:
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2))

        # Add colorbar for absolute reference
        cb = plt.colorbar(cnt, ax=ax, shrink=0.6, pad=0.1)
        cb.set_label("Predictive Sigma", fontsize=10)

    plt.suptitle(
        "Smoothed Action Simplex Uncertainty Evolution\nCapturing High-Fidelity Confidence Valleys via Centroid-Summarized GPs",
        fontsize=22,
        y=1.08,
    )
    plt.tight_layout()
    os.makedirs("results/gp_analysis", exist_ok=True)
    plt.savefig(
        "results/gp_analysis/sparse_simplex_evolution.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(
        "\nSmooth evolution plots generated in results/gp_analysis/sparse_simplex_evolution.png"
    )


if __name__ == "__main__":
    main()
