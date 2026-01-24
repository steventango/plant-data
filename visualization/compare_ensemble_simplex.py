import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import mpltern
import os
import pickle
import jax
import jax.numpy as jnp
import equinox as eqx
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C


# Model definition for deserialization
class HeteroscedasticMLP(eqx.Module):
    layers: list

    def __init__(self, in_size, out_size, key):
        keys = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(in_size, 128, key=keys[0]),
            jax.nn.relu,
            eqx.nn.Linear(128, 128, key=keys[1]),
            jax.nn.relu,
            eqx.nn.Linear(128, out_size * 2, key=keys[2]),
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        mean, log_var = jnp.split(x, 2)
        log_var = jnp.clip(log_var, -8, 2)
        return mean, log_var


def main():
    # 1. Load Data & Metadata
    meta_path = "results/ensemble_models/metadata.pkl"
    if not os.path.exists(meta_path):
        print("Ensemble metadata not found. Please run train_deep_ensemble.py first.")
        return

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    scaler = meta["scaler"]
    feature_cols = meta["feature_cols"]

    path = "/data/plant-rl/offline/v22/mixed-v22.parquet"
    df_raw = pl.read_parquet(path)
    df = df_raw.with_columns(
        days=(
            pl.col("time") - pl.col("time").min().over("experiment", "zone", "plant_id")
        ).dt.total_seconds()
        / (24 * 3600)
    ).with_columns(
        [pl.col("cls_token_pca").list.get(i).alias(f"pca_{i}") for i in range(5)]
    )

    # Dynamics for "Reference Plant"
    traj_df = df.filter(pl.col("plant_id") == 0).sort("days")
    days_to_plot = [0, 4, 8, 12]

    # 2. Load Models
    print("Loading Ensemble and GP Models...")
    # Loading Ensemble
    ensemble_size = 5
    ensemble = []
    # template for loading
    template = HeteroscedasticMLP(len(feature_cols), 1, jax.random.PRNGKey(0))
    for i in range(ensemble_size):
        m = eqx.tree_deserialise_leaves(
            f"results/ensemble_models/member_{i}.eqx", template
        )
        ensemble.append(m)

    # Load GP (We'll retrain a Centroid-GP quickly for this script to ensure absolute sync)
    # Re-using the logic from plot_simplex_uncertainty.py
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
    X_raw[:, 0] = np.log(X_raw[:, 0])
    Y_raw = df_dyn["delta_ln_area"].to_numpy().reshape(-1, 1)

    from sklearn.cluster import MiniBatchKMeans

    X_f_s = scaler.transform(X_raw)
    kmeans = MiniBatchKMeans(n_clusters=2000, random_state=42).fit(X_f_s)
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=1).fit(X_f_s)
    _, idx = nn.kneighbors(kmeans.cluster_centers_)
    gp = GaussianProcessRegressor(
        kernel=C(1.0) * Matern(nu=1.5) + WhiteKernel(noise_level=0.01), normalize_y=True
    )
    gp.fit(X_f_s[idx.flatten()], Y_raw[idx.flatten()])
    print("Models loaded and synced.")

    # 3. Evolutionary Comparison
    n_grid = 70  # Balanced res for dual plot
    t_v = np.linspace(0, 1, n_grid)
    l_v = np.linspace(0, 1, n_grid)
    T, L = np.meshgrid(t_v, l_v)
    R_g = 1.0 - T - L
    mask = R_g >= -1e-10
    tm, lm, rm = T[mask], L[mask], R_g[mask]

    fig = plt.figure(figsize=(24, 12))

    for i, day in enumerate(days_to_plot):
        idx_t = (traj_df["days"] - day).abs().arg_min()
        row = traj_df[idx_t].to_dicts()[0]
        ln_area = np.log(row["clean_area"])
        pca_v = [row[f"pca_{k}"] for k in range(5)]
        real_act = [row["red_coef"], row["white_coef"], row["blue_coef"]]
        s_act = sum(real_act)
        real_act = [a / s_act for a in real_act]

        X_grid = np.zeros((len(tm), len(feature_cols)))
        X_grid[:, 0] = ln_area
        X_grid[:, 1], X_grid[:, 2], X_grid[:, 3] = lm, tm, rm  # R, W, B
        for k in range(5):
            X_grid[:, 4 + k] = pca_v[k]
        X_grid_scaled = scaler.transform(X_grid)

        # GP Prediction (Absolute reference)
        _, gp_sigma = gp.predict(X_grid_scaled, return_std=True)
        gp_sigma = gp_sigma.reshape(-1)

        # Ensemble Prediction
        ens_preds = []
        for m in ensemble:
            mu, _ = jax.vmap(m)(jnp.array(X_grid_scaled))
            ens_preds.append(mu)
        ens_preds = jnp.stack(ens_preds)
        # Disagreement = Standard Deviation of ensemble means
        disagreement = jnp.std(ens_preds, axis=0).reshape(-1)

        # Plot GP (Top Row)
        ax_gp = fig.add_subplot(2, 4, i + 1, projection="ternary")
        cnt_gp = ax_gp.tricontourf(tm, lm, rm, gp_sigma, levels=30, cmap="viridis")
        ax_gp.scatter(
            real_act[1],
            real_act[0],
            real_act[2],
            color="red",
            marker="*",
            s=200,
            edgecolors="white",
            zorder=10,
        )
        ax_gp.set_title(
            f"Day {row['days']:.0f} | GP Uncertainty\nArea {row['clean_area']:.0f}mm^2",
            fontsize=14,
        )
        if i == 0:
            ax_gp.set_ylabel(
                "GAUSSIAN PROCESS\n(Kernel Epistemic)",
                fontsize=16,
                labelpad=40,
                fontweight="bold",
            )

        # Plot Ensemble Disagreement (Bottom Row)
        ax_en = fig.add_subplot(2, 4, i + 5, projection="ternary")
        cnt_en = ax_en.tricontourf(tm, lm, rm, disagreement, levels=30, cmap="magma")
        ax_en.scatter(
            real_act[1],
            real_act[0],
            real_act[2],
            color="cyan",
            marker="*",
            s=200,
            edgecolors="black",
            zorder=10,
        )
        ax_en.set_title(f"Day {row['days']:.0f} | Ensemble Disagreement", fontsize=14)
        if i == 0:
            ax_en.set_ylabel(
                "DEEP ENSEMBLE\n(Model Disagreement)",
                fontsize=16,
                labelpad=40,
                fontweight="bold",
            )

        # Shared ternary labels
        for ax in [ax_gp, ax_en]:
            ax.set_tlabel("WHITE", fontsize=9)
            ax.set_llabel("RED", fontsize=9)
            ax.set_rlabel("BLUE", fontsize=9)

    plt.suptitle(
        "Epistemic Uncertainty Scaling: Gaussian Process vs Deep Ensembles\nComparing kernel-based knowledge gaps to neural network ensemble disagreement",
        fontsize=24,
        y=1.02,
    )
    plt.tight_layout()
    os.makedirs("results/ensemble_analysis", exist_ok=True)
    plt.savefig(
        "results/ensemble_analysis/ensemble_vs_gp_simplex.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(
        "\nComparison complete. Results in results/ensemble_analysis/ensemble_vs_gp_simplex.png"
    )


if __name__ == "__main__":
    main()
