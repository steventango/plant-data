import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import os
import joblib


def main():
    path = "/data/plant-rl/offline/v22/mixed-v22.parquet"
    print(f"Loading data from {path}...")
    df = pl.read_parquet(path)

    # 1. Prepare data (Days 0-14)
    pca_dim = 5
    df = df.with_columns(
        (
            (
                pl.col("time")
                - pl.col("time").min().over("experiment", "zone", "plant_id")
            ).dt.total_seconds()
            / (24 * 3600)
        ).alias("days")
    ).filter((pl.col("days") <= 14) & pl.col("valid") & (pl.col("clean_area") > 0))

    # Expand PCA columns
    df = df.with_columns(
        [pl.col("cls_token_pca").list.get(i).alias(f"pca_{i}") for i in range(pca_dim)]
    )

    action_cols = ["red_coef", "white_coef", "blue_coef"]
    pca_cols = [f"pca_{i}" for i in range(pca_dim)]

    # Sort for shift
    df = df.sort("experiment", "zone", "plant_id", "time")

    # Create Targets: Delta States
    # y = ln(Area_t+1) - ln(Area_t)
    # y_pca = PCA_i_t+1 - PCA_i_t
    df = df.with_columns(
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

    # Subsample for training efficiency
    df_sample = df.sample(n=min(1500, len(df)), seed=42)

    X_raw = df_sample[feature_cols].to_numpy()
    Y_raw = df_sample[target_cols].to_numpy()

    # Log transform input area
    X_train_raw = X_raw.copy()
    X_train_raw[:, 0] = np.log(X_raw[:, 0])

    # 2. Input Standardization
    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train_raw)

    # Store next area for RMSE calculation in raw space
    # (A_t+1 = exp(ln(A_t) + delta_ln_area))
    ln_area_t = X_train_raw[:, 0]
    next_area_actual = np.exp(ln_area_t + Y_raw[:, 0])

    # 3. Train/Test Split
    (
        X_train,
        X_test,
        Y_train,
        Y_test,
        area_t_train,
        area_t_test,
        area_next_train,
        area_next_test,
    ) = train_test_split(
        X_train_scaled,
        Y_raw,
        ln_area_t,
        next_area_actual,
        test_size=0.2,
        random_state=42,
    )

    os.makedirs("results/gp_comparison_best_practices", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 4. Train GP (Matern + ARD)
    print("Training GP on Delta-States with Standardization...")
    kernel = C(1.0) * Matern(
        length_scale=np.ones(X_train.shape[1]), nu=1.5
    ) + WhiteKernel(noise_level=1e-2)
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=5, normalize_y=True
    )
    gp.fit(X_train, Y_train)

    # Save model and scaler for rollout
    joblib.dump(gp, "models/gp_dynamics.joblib")
    joblib.dump(scaler_x, "models/scaler_x.joblib")

    # --- Evaluation ---
    print("\nEvaluating Model...")

    # Predict Deltas
    pred_delta, sigma_delta = gp.predict(X_test, return_std=True)

    # Transform back to Raw Area Space
    # ln(Area_t+1) = ln_area_t + delta_ln_area
    pred_ln_area_next = area_t_test + pred_delta[:, 0]

    # For Log-Normal mean if we have variance: exp(mu + sigma^2 / 2)
    # sigma_delta[0] is std of delta_ln_area
    if sigma_delta.ndim == 1:
        s_area = sigma_delta
    else:
        s_area = sigma_delta[:, 0]

    y_pred_back = np.exp(pred_ln_area_next + (s_area**2) / 2)

    rmse = np.sqrt(mean_squared_error(area_next_test, y_pred_back))
    mae = mean_absolute_error(area_next_test, y_pred_back)

    print(f"Best Practices GP - RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # --- Visualization ---
    plt.figure(figsize=(10, 8))
    plt.scatter(area_next_test, y_pred_back, alpha=0.5, color="blue", s=15)
    plt.plot(
        [area_next_test.min(), area_next_test.max()],
        [area_next_test.min(), area_next_test.max()],
        "k--",
        lw=2,
    )
    plt.title(
        f"GP Best Practices: Predicted vs Actual\nRMSE: {rmse:.2f} (Predicting Deltas + Normalization)"
    )
    plt.xlabel("Actual Area_{t+1}")
    plt.ylabel("Predicted Area_{t+1}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/gp_comparison_best_practices/predicted_vs_actual.png", dpi=150)
    plt.close()

    # Log summary
    with open("results/gp_comparison_best_practices/summary.txt", "w") as f:
        f.write(f"GP Best Practices RMSE: {rmse:.4f}\n")
        f.write(f"GP Best Practices MAE: {mae:.4f}\n")


if __name__ == "__main__":
    main()
