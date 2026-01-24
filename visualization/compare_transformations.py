import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


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

    groups = df.group_by(["experiment", "zone", "plant_id"])

    log_r2 = []
    sqrt_r2 = []

    for (exp, zone, pid), group in groups:
        if len(group) < 5:
            continue

        x = group["days"].to_numpy().reshape(-1, 1)
        area = group["clean_area"].to_numpy()

        y_log = np.log(area)
        y_sqrt = np.sqrt(area)

        # Fit linear model to log(area)
        model_log = LinearRegression().fit(x, y_log)
        r2_l = r2_score(y_log, model_log.predict(x))
        log_r2.append(r2_l)

        # Fit linear model to sqrt(area)
        model_sqrt = LinearRegression().fit(x, y_sqrt)
        r2_s = r2_score(y_sqrt, model_sqrt.predict(x))
        sqrt_r2.append(r2_s)

    print(f"Mean R-squared for linear fit to log(area): {np.mean(log_r2):.4f}")
    print(f"Mean R-squared for linear fit to sqrt(area): {np.mean(sqrt_r2):.4f}")

    better_sqrt = (np.array(sqrt_r2) > np.array(log_r2)).sum()
    print(
        f"Sqrt(area) is more linear for {better_sqrt / len(log_r2) * 100:.1f}% of plants."
    )


if __name__ == "__main__":
    main()
