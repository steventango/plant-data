"""UMAP dimensionality reduction and visualization for plant embeddings (New Implementation)."""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from PIL import Image
from umap import UMAP

from visualization.umap_visualization import visualize

logger = logging.getLogger(__name__)


def load_parquet_data(
    parquet_path: Path,
    filter_plant_ids: Optional[List[int]] = None,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, List[Path], pl.DataFrame]:
    """
    Load embeddings and image paths from parquet file.

    Args:
        parquet_path: Path to parquet file
        filter_plant_ids: Optional list of plant IDs to include
        max_samples: Optional maximum number of samples to load

    Returns:
        Tuple of (embeddings array, image paths list, dataframe)
    """
    logger.info(f"Loading data from {parquet_path}")
    df = pl.read_parquet(parquet_path)

    # Filter by plant IDs if specified
    if filter_plant_ids is not None:
        df = df.filter(pl.col("plant_id").is_in(filter_plant_ids))
        logger.info(f"Filtered to {len(df)} samples from plant IDs: {filter_plant_ids}")

    # Limit samples if specified
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, seed=42)
        logger.info(f"Sampled {max_samples} random samples")

    # Extract embeddings
    # Assuming embedding column is List[Float32]
    embeddings = np.stack(df["embedding"].to_numpy())

    # Build full image paths
    # Assuming image_path is relative to /data/offline
    # We need to construct the absolute path based on where the script is running or where data is mounted
    # For now, let's assume /data/offline is the base
    offline_dir = Path("/data/offline")
    image_paths = [offline_dir / path for path in df["image_path"]]

    logger.info(
        f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}"
    )

    return embeddings, image_paths, df


def load_images(
    image_paths: List[Path],
    max_size: Tuple[int, int] = (224, 224),
) -> List[Image.Image]:
    """
    Load images from paths.

    Args:
        image_paths: List of paths to images
        max_size: Maximum size for loaded images (for memory efficiency)

    Returns:
        List of PIL Images
    """
    images = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            images.append(img.convert("RGB"))
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}")
            # Create a blank placeholder image
            images.append(Image.new("RGB", max_size, color="gray"))

    return images


def compute_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """
    Compute UMAP dimensionality reduction.

    Args:
        embeddings: Input embeddings array (n_samples, n_features)
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance for UMAP
        metric: Distance metric
        random_state: Random seed

    Returns:
        2D UMAP coordinates (n_samples, 2)
    """
    logger.info(
        f"Computing UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}"
    )
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=True,
    )
    umap_embeddings = reducer.fit_transform(embeddings)
    logger.info(f"UMAP complete: {umap_embeddings.shape}")
    return umap_embeddings


def create_plant_id_colors(df: pl.DataFrame) -> np.ndarray:
    """
    Create normalized colors based on plant_id for visualization.

    Args:
        df: DataFrame with 'plant_id' column

    Returns:
        Normalized color values (0-1) for each sample
    """
    plant_ids = df["plant_id"].to_numpy()
    unique_ids = sorted(np.unique(plant_ids))
    id_to_color = {
        pid: i / (len(unique_ids) - 1) if len(unique_ids) > 1 else 0.5
        for i, pid in enumerate(unique_ids)
    }
    colors = np.array([id_to_color[pid] for pid in plant_ids])
    return colors


def create_date_colors(df: pl.DataFrame) -> np.ndarray:
    """
    Create normalized colors based on days since the start of each experiment.

    Args:
        df: DataFrame with 'time' column

    Returns:
        Normalized color values (0-1) for each sample, representing time progression
    """
    if "time" not in df.columns:
        logger.warning("No time column found, falling back to plant_id colors")
        return create_plant_id_colors(df)

    timestamps = df["time"].to_numpy()
    min_time = timestamps.min()
    max_time = timestamps.max()

    # Calculate total seconds range
    total_seconds = (max_time - min_time).astype("timedelta64[s]").astype(float)

    if total_seconds == 0:
        return np.full(len(timestamps), 0.5)

    # Normalize
    colors = (timestamps - min_time).astype("timedelta64[s]").astype(
        float
    ) / total_seconds
    return colors


def process_parquet(
    parquet_path: Path,
    output_path: Optional[Path] = None,
    filter_plant_ids: Optional[List[int]] = None,
    max_samples: Optional[int] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    visualize_results: bool = True,
    color_by: str = "date",
) -> Dict:
    """
    Process a Parquet file: load embeddings, compute UMAP, and visualize.

    Args:
        parquet_path: Path to parquet file
        output_path: Optional output path for visualization
        filter_plant_ids: Optional list of plant IDs to include
        max_samples: Optional maximum number of samples
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        visualize_results: Whether to generate visualization
        color_by: Color by 'plant_id' or 'date' (default: 'date')

    Returns:
        Dictionary with processing results
    """
    # Load data
    embeddings, image_paths, df = load_parquet_data(
        parquet_path,
        filter_plant_ids=filter_plant_ids,
        max_samples=max_samples,
    )

    # Compute UMAP
    umap_embeddings = compute_umap(
        embeddings,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )

    # Create visualization
    if visualize_results:
        if output_path is None:
            output_path = parquet_path.parent / "umap_visualization_new.png"

        logger.info("Loading images for visualization...")
        images = load_images(image_paths)

        logger.info(f"Creating color mapping by {color_by}...")
        if color_by == "date":
            colors = create_date_colors(df)
        else:
            colors = create_plant_id_colors(df)

        logger.info(f"Generating visualization at {output_path}...")
        visualize.plot_umap_embeddings(
            umap_embeddings,
            images,
            colors,
            output_path=str(output_path),
        )

    return {
        "umap_embeddings": umap_embeddings,
        "dataframe": df,
        "output_path": output_path if visualize_results else None,
    }


def main():
    """Command-line interface for UMAP visualization."""
    parser = argparse.ArgumentParser(
        description="Generate UMAP visualization from parquet embeddings"
    )
    parser.add_argument(
        "parquet_path",
        type=Path,
        help="Path to parquet file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for visualization (default: umap_visualization_new.png in same directory)",
    )
    parser.add_argument(
        "--plant-ids",
        type=int,
        nargs="+",
        help="Filter to specific plant IDs",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to visualize (random sample)",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter (default: 15)",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter (default: 0.1)",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip visualization (only compute UMAP)",
    )
    parser.add_argument(
        "--color-by",
        choices=["plant_id", "date"],
        default="date",
        help="Color images by plant_id or date (default: date)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Validate input
    if not args.parquet_path.exists():
        logger.error(f"Parquet file not found: {args.parquet_path}")
        return

    # Process
    result = process_parquet(
        args.parquet_path,
        output_path=args.output,
        filter_plant_ids=args.plant_ids,
        max_samples=args.max_samples,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        visualize_results=not args.no_visualize,
        color_by=args.color_by,
    )

    if result["output_path"]:
        print(f"\nVisualization saved to: {result['output_path']}")
    print(f"Processed {len(result['dataframe'])} samples")


if __name__ == "__main__":
    main()
