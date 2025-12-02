"""Visualization utilities for UMAP embeddings."""

import logging
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

logger = logging.getLogger(__name__)


def plot_umap_embeddings(
    umap_embeddings: np.ndarray,
    images: List[Image.Image],
    colors: np.ndarray,
    output_path: str = "umap_visualization.png",
    figsize: Tuple[int, int] = (20, 20),
    image_zoom: float = 0.1,
    scatter_alpha: float = 0.5,
    scatter_size: int = 50,
    cmap: str = "viridis",
) -> None:
    """
    Plot UMAP embeddings with images overlaid.

    Args:
        umap_embeddings: 2D array of UMAP coordinates (n_samples, 2)
        images: List of PIL Images corresponding to each sample
        colors: Array of values for coloring points (n_samples,)
        output_path: Path to save the plot
        figsize: Figure size tuple
        image_zoom: Zoom level for overlaid images
        scatter_alpha: Alpha transparency for scatter points
        scatter_size: Size of scatter points
        cmap: Colormap name
    """
    logger.info(f"Creating plot with {len(images)} samples...")

    fig, ax = plt.subplots(figsize=figsize)

    # Create scatter plot
    scatter = ax.scatter(
        umap_embeddings[:, 0],
        umap_embeddings[:, 1],
        c=colors,
        cmap=cmap,
        alpha=scatter_alpha,
        s=scatter_size,
    )

    # Add colorbar
    plt.colorbar(scatter, ax=ax, label="Normalized Value")

    # Add images
    logger.info("Adding images to plot...")
    for x, y, img in zip(umap_embeddings[:, 0], umap_embeddings[:, 1], images):
        im = OffsetImage(img, zoom=image_zoom)
        ab = AnnotationBbox(im, (x, y), frameon=False, pad=0)
        ax.add_artist(ab)

    ax.set_title("UMAP Projection of Plant Embeddings")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    # Remove ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

    logger.info(f"Saving plot to {output_path}...")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Plot saved successfully.")
