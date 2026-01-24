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

    # 1. Action Frequency Analysis
    print("\nAnalyzing Action Frequencies in Dataset...")
    # Bin continuous actions for frequency count
    # Pure White: W > 0.9, R < 0.1, B < 0.1
    # Pure Red:   R > 0.9, W < 0.1, B < 0.1
    # Pure Blue:  B > 0.9, W < 0.1, R < 0.1

    w_count = df.filter(
        (pl.col("white_coef") > 0.9)
        & (pl.col("red_coef") < 0.1)
        & (pl.col("blue_coef") < 0.1)
    ).height
    r_count = df.filter(
        (pl.col("red_coef") > 0.9)
        & (pl.col("white_coef") < 0.1)
        & (pl.col("blue_coef") < 0.1)
    ).height
    b_count = df.filter(
        (pl.col("blue_coef") > 0.9)
        & (pl.col("white_coef") < 0.1)
        & (pl.col("red_coef") < 0.1)
    ).height

    print(f"Pure White count: {w_count}")
    print(f"Pure Red   count: {r_count}")
    print(f"Pure Blue  count: {b_count}")

    # 2. Train GP Model (Constrained Noise: 0.01)
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
    X_raw_train = df_sample[feature_cols].to_numpy()
    X_raw_train[:, 0] = np.log(X_raw_train[:, 0])  # ln(Area)
    Y_train = df_sample[target_cols].to_numpy()

    scaler_x = StandardScaler()
    X_scaled_train = scaler_x.fit_transform(X_raw_train)

    print("Training GP...")
    kernel = C(1.0) * Matern(
        length_scale=np.ones(X_scaled_train.shape[1]), nu=1.5
    ) + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 0.01))
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=5, normalize_y=True
    )
    gp.fit(X_scaled_train, Y_train)

    # 3. Action Sensitivity Test
    # Fixed representative state (Mean initial area and morphology)
    ref_ln_area = np.log(df["clean_area"].mean())
    ref_pca = [df[f"pca_{i}"].mean() for i in range(pca_dim)]

    test_actions = [
        [1.0, 0.0, 0.0],  # Pure Red
        [0.0, 1.0, 0.0],  # Pure White
        [0.0, 0.0, 1.0],  # Pure Blue
    ]
    labels = ["Pure Red", "Pure White", "Pure Blue"]
    freqs = [r_count, w_count, b_count]
    sigmas = []

    for act in test_actions:
        # Features: [ln_area, r, w, b, pca0..4]
        inp_raw = np.array([[ref_ln_area] + act + ref_pca])
        inp_scaled = scaler_x.transform(inp_raw)
        _, sigma = gp.predict(inp_scaled, return_std=True)
        s = sigma[0, 0] if sigma.ndim > 1 else sigma[0]
        sigmas.append(s)

    # 4. Visualization
    os.makedirs("results/gp_analysis", exist_ok=True)
    plt.figure(figsize=(10, 7))

    bars = plt.bar(labels, sigmas, color=["red", "green", "blue"], alpha=0.7)
    plt.ylabel("Predictive Sigma (1-step Delta)")
    plt.title(
        "Action-Dependent Uncertainty Hypothesis Verification\nDoes higher data frequency lead to lower sigma?"
    )

    for i, bar in enumerate(bars):
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.001,
            f"Freq: {freqs[i]}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    plt.grid(True, axis="y", alpha=0.3)
    plt.ylim(0, max(sigmas) * 1.3)
    plt.tight_layout()
    plt.savefig("results/gp_analysis/action_uncertainty_comparison.png", dpi=150)
    plt.close()

    print("\nAction Sensitivity Test Complete.")
    for i, label in enumerate(labels):
        print(f"{label}: Sigma = {sigmas[i]:.4f} (Training Count: {freqs[i]})")


if __name__ == "__main__":
    main()
