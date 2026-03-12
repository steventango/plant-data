import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    pca_model_path = "/data/plant-rl/offline/v22/pca_model.joblib"
    output_path = "results/pca_variance.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info(f"Loading PCA model from {pca_model_path}...")
    try:
        pca = joblib.load(pca_model_path)
    except Exception as e:
        logger.error(f"Failed to load PCA model: {e}")
        return

    variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_ratio)
    num_components = len(variance_ratio)

    logger.info(f"Plotting variance for {num_components} components...")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar plot for individual variance
    color = "tab:blue"
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio", color=color)
    bars = ax1.bar(
        range(1, num_components + 1),
        variance_ratio,
        color=color,
        alpha=0.7,
        label="Individual Variance",
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_xticks(range(1, num_components + 1))

    # Line plot for cumulative variance
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Cumulative Explained Variance", color=color)
    line = ax2.plot(
        range(1, num_components + 1),
        cumulative_variance,
        color=color,
        marker="o",
        label="Cumulative Variance",
    )
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0, 1.05)

    # Title and grid
    plt.title("PCA Explained Variance by Component")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Legends
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # Add text for total variance explained if relevant
    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
