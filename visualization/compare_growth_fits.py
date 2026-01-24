import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import os
import argparse


def quadratic(x, a, b, c):
    return a * x**2 + b * x + c


def exponential(x, a, b):
    # Using a * exp(b * x)
    return a * np.exp(b * x)


def fit_curves(group):
    # Ensure time is relative to start of episode for better fitting
    x = (
        (group["time"] - group["time"].min()).dt.total_seconds() / (24 * 3600)
    ).to_numpy()  # days
    y = group["clean_area"].to_numpy()

    # Filter out zeros or negative values for area if any (though unlikely for clean_area)
    valid_mask = y > 0
    if (
        valid_mask.sum() < 4
    ):  # Need at least 4 points for quadratic (3 params) and stability
        return None

    x = x[valid_mask]
    y = y[valid_mask]

    # Fit Quadratic
    try:
        popt_quad, _ = curve_fit(quadratic, x, y)
        y_quad = quadratic(x, *popt_quad)
        r2_quad = r2_score(y, y_quad)
    except Exception:
        r2_quad = -np.inf

    # Fit Exponential
    try:
        # Initial guess for exp: y[0] for 'a', then estimate 'b' from end points
        # b = ln(y_end/y_start) / delta_x
        b_guess = np.log(y[-1] / y[0]) / (x[-1] - x[0]) if x[-1] > x[0] else 0.1
        popt_exp, _ = curve_fit(exponential, x, y, p0=[y[0], b_guess], maxfev=2000)
        y_exp = exponential(x, *popt_exp)
        r2_exp = r2_score(y, y_exp)
    except Exception:
        r2_exp = -np.inf

    return {"r2_quad": r2_quad, "r2_exp": r2_exp, "n_points": len(x)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, default="/data/plant-rl/offline/v22/mixed-v22.parquet"
    )
    parser.add_argument("--output-dir", type=str, default="results/growth_fits")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading data from {args.input}...")
    df = pl.read_parquet(args.input)

    # Remove outliers or invalid data if necessary
    if "valid" in df.columns:
        df = df.filter(pl.col("valid"))

    # Filter to 14 days as requested
    df = df.with_columns(
        (
            (
                pl.col("time")
                - pl.col("time").min().over("experiment", "zone", "plant_id")
            ).dt.total_seconds()
            / (24 * 3600)
        ).alias("days")
    )
    df = df.filter(pl.col("days") <= 14)

    groups = df.group_by(["experiment", "zone", "plant_id"])

    results = []
    n_plants = df.select(["experiment", "zone", "plant_id"]).unique().height
    print(f"Fitting curves for {n_plants} plants...")

    # Sample some plants for plotting later
    plot_samples = []

    for (exp, zone, pid), group in groups:
        if len(group) < 5:
            continue

        fit_res = fit_curves(group)
        if fit_res:
            fit_res.update({"experiment": exp, "zone": zone, "plant_id": pid})
            results.append(fit_res)

            # Save some groups for plotting (limit to avoid memory bloat)
            if len(plot_samples) < 9:
                plot_samples.append(((exp, zone, pid), group))

    res_df = pl.DataFrame(results)

    # Filter out failed fits
    res_df = res_df.filter((pl.col("r2_quad") > -1) & (pl.col("r2_exp") > -1))

    avg_quad = res_df["r2_quad"].mean()
    avg_exp = res_df["r2_exp"].mean()

    better_exp = (res_df["r2_exp"] > res_df["r2_quad"]).sum()
    total = len(res_df)

    print("\n--- Summary Results ---")
    print(f"Total plants analyzed: {total}")
    print(f"Average R-squared (Quadratic): {avg_quad:.4f}")
    print(f"Average R-squared (Exponential): {avg_exp:.4f}")
    print(
        f"Plants where Exponential fits better: {better_exp} ({100 * better_exp / total:.1f}%)"
    )
    print(
        f"Plants where Quadratic fits better: {total - better_exp} ({100 * (total - better_exp) / total:.1f}%)"
    )

    # 1. Distribution Plot
    plt.figure(figsize=(10, 6))
    sns.histplot(
        res_df["r2_quad"], label="Quadratic", color="blue", kde=True, alpha=0.5
    )
    sns.histplot(
        res_df["r2_exp"], label="Exponential", color="orange", kde=True, alpha=0.5
    )
    plt.xlabel("R-squared")
    plt.title("Distribution of R-squared: Quadratic vs Exponential")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/r2_distribution.png", dpi=150)
    plt.close()

    # 2. Scatter comparison
    plt.figure(figsize=(8, 8))
    plt.scatter(res_df["r2_quad"], res_df["r2_exp"], alpha=0.3, s=10)
    plt.plot([0, 1], [0, 1], "r--", label="y=x")
    plt.xlabel("Quadratic R2")
    plt.ylabel("Exponential R2")
    plt.title("Quadratic vs Exponential R-squared Comparison")
    plt.xlim(0.5, 1.05)
    plt.ylim(0.5, 1.05)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/r2_scatter.png", dpi=150)
    plt.close()

    # 3. Sample Plots
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()

    for i, ((exp, zone, pid), group) in enumerate(plot_samples):
        ax = axes[i]
        x_raw = group["days"].to_numpy()
        y = group["clean_area"].to_numpy()

        ax.scatter(x_raw, y, label="Data", color="black", s=15)

        # Re-fit for plotting smooth curves
        x_smooth = np.linspace(0, 14, 100)

        try:
            popt_quad, _ = curve_fit(quadratic, x_raw, y)
            ax.plot(
                x_smooth,
                quadratic(x_smooth, *popt_quad),
                label=f"Quad (R2: {res_df.filter((pl.col('experiment') == exp) & (pl.col('zone') == zone) & (pl.col('plant_id') == pid))['r2_quad'][0]:.3f})",
                color="blue",
            )
        except Exception:
            pass

        try:
            b_guess = (
                np.log(y[-1] / y[0]) / (x_raw[-1] - x_raw[0])
                if x_raw[-1] > x_raw[0]
                else 0.1
            )
            popt_exp, _ = curve_fit(exponential, x_raw, y, p0=[y[0], b_guess])
            ax.plot(
                x_smooth,
                exponential(x_smooth, *popt_exp),
                label=f"Exp (R2: {res_df.filter((pl.col('experiment') == exp) & (pl.col('zone') == zone) & (pl.col('plant_id') == pid))['r2_exp'][0]:.3f})",
                color="orange",
            )
        except Exception:
            pass

        ax.set_xlim(0, 14)

        ax.set_title(f"Plant E{exp}/Z{zone}/P{pid}")
        ax.set_xlabel("Days")
        ax.set_ylabel("Area")
        ax.legend(fontsize="small")

    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/sample_fits.png", dpi=150)
    plt.close()

    print(f"\nPlots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
