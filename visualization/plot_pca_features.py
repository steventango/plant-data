import polars as pl
import numpy as np
import joblib
import matplotlib.pyplot as plt
from PIL import Image
import os
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_image(path, target_size=(128, 128)):
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail(target_size, Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        logger.warning(f"Failed to load image at {path}: {e}")
        return Image.new("RGB", target_size, color="gray")


def main():
    # Paths
    input_path = "/data/plant-rl/offline/v22/mixed-v22.parquet"
    pca_model_path = "/data/plant-rl/offline/v22/pca_model.joblib"
    output_dir = "results/pca_features"
    os.makedirs(output_dir, exist_ok=True)

    K = 10
    num_samples = 12

    logger.info(f"Loading data from {input_path}...")
    df = pl.read_parquet(input_path, columns=["cls_token_pca", "image_path"])

    logger.info(f"Loading PCA model from {pca_model_path}...")
    pca = joblib.load(pca_model_path)

    logger.info("Extracting projections...")
    projections = np.array(df["cls_token_pca"].to_list())

    for k in range(K):
        logger.info(f"Visualizing PC {k + 1}...")
        pc_values = projections[:, k]

        # Linspace sampling across the range
        min_val = pc_values.min()
        max_val = pc_values.max()
        target_values = np.linspace(min_val, max_val, num_samples)

        indices_to_plot = []
        for target in target_values:
            idx = np.argmin(np.abs(pc_values - target))
            indices_to_plot.append(idx)

        sample_values = pc_values[indices_to_plot]
        sample_paths = df["image_path"][indices_to_plot].to_list()

        # Plot layout: Histogram top, Images bottom in two rows of 6
        fig = plt.figure(figsize=(15, 8))
        gs = fig.add_gridspec(3, 6, height_ratios=[1.5, 1, 1])

        # Histogram on the top
        ax_hist = fig.add_subplot(gs[0, :])
        ax_hist.hist(pc_values, bins=100, color="skyblue", edgecolor="black", alpha=0.7)
        ax_hist.set_title(
            f"PC {k + 1} Distribution (Var: {pca.explained_variance_ratio_[k]:.2%})"
        )
        ax_hist.set_xlabel("Value")
        ax_hist.set_ylabel("Count")

        # Add red markers for samples on the histogram
        for val in sample_values:
            ax_hist.axvline(x=val, color="red", linestyle="--", alpha=0.3)
            ax_hist.scatter(val, 0, color="red", zorder=5, s=20)

        # Plot samples
        for i in range(num_samples):
            row = 1 + (i // 6)
            col = i % 6
            ax = fig.add_subplot(gs[row, col])
            img = load_image(sample_paths[i])
            ax.imshow(img)
            ax.set_title(f"Val: {sample_values[i]:.2f}", fontsize=9)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/pc_{k + 1:02d}.png")
        plt.close()

    logger.info(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
