import polars as pl
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
import os
import logging
from umap import UMAP

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_image(path, target_size=(64, 64)):
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail(target_size, Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        logger.warning(f"Failed to load image at {path}: {e}")
        return Image.new("RGB", target_size, color="gray")


def main():
    # Paths
    input_path = "/data/plant-rl/offline/v23/mixed-v23.parquet"
    pca_model_path = "/data/plant-rl/offline/v23/pca_model.joblib"
    output_dir = "results/pca_scatter"
    os.makedirs(output_dir, exist_ok=True)

    K = 10
    num_samples = 1000

    logger.info(f"Loading data from {input_path}...")
    df = pl.read_parquet(
        input_path, columns=["cls_token", "cls_token_pca", "image_path"]
    )

    logger.info(f"Loading PCA model from {pca_model_path}...")
    pca = joblib.load(pca_model_path)

    logger.info("Extracting projections...")
    all_embeddings = np.array(df["cls_token"].to_list())
    all_projections = np.array(df["cls_token_pca"].to_list())

    logger.info(f"Sampling {num_samples} random images...")
    indices = np.random.choice(len(df), num_samples, replace=False)

    sample_embeddings = all_embeddings[indices]
    sample_projections = all_projections[indices]
    sample_paths = df["image_path"][indices].to_list()

    logger.info("Computing 1D UMAP for Y-axis (context)...")
    reducer = UMAP(n_components=1, random_state=42)
    y_coords = reducer.fit_transform(sample_embeddings).flatten()

    # Scale Y coords for better plotting
    y_coords = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min())

    logger.info("Loading sampled images...")
    images = [load_image(path) for path in sample_paths]

    for k in range(K):
        logger.info(f"Visualizing PC {k + 1} scatter...")
        x_coords = sample_projections[:, k]

        fig, ax = plt.subplots(figsize=(20, 12))

        # Plot background scatter for distribution context
        pc_values_all = all_projections[:, k]
        ax.hist(
            pc_values_all, bins=100, color="gray", alpha=0.1, density=True, zorder=0
        )

        # Plot images
        for i in range(num_samples):
            img = images[i]
            x, y = x_coords[i], y_coords[i]

            im = OffsetImage(img, zoom=0.6)
            ab = AnnotationBbox(im, (x, y), frameon=False)
            ax.add_artist(ab)

            # Small dot to show center
            ax.scatter(x, y, color="red", s=5, alpha=0.5)

        ax.set_title(
            f"PC {k + 1} Interpretation (X=Value, Y=UMAP Similarity)\nVar: {pca.explained_variance_ratio_[k]:.2%}",
            fontsize=20,
        )
        ax.set_xlabel(f"PC {k + 1} Value", fontsize=15)
        ax.set_ylabel("UMAP 1 (Rest of Dimensions)", fontsize=15)

        # Adjust limits
        ax.set_xlim(x_coords.min() * 1.1, x_coords.max() * 1.1)
        ax.set_ylim(-0.1, 1.1)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/pc_{k + 1:02d}_scatter.png", dpi=150)
        plt.close()

    logger.info(f"Scatter visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
