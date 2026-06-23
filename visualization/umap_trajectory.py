import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from umap import UMAP

from transforms.normalization import load_normalization_stats
from visualization.common import RESULTS_DIR

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

STATE_COLS = [
    "wall_time",
    "clean_area",
    "clean_convex_hull_area",
    "clean_solidity",
    "clean_perimeter",
    "clean_width",
    "clean_height",
    "clean_longest_path",
    "clean_center_of_mass_x",
    "clean_center_of_mass_y",
    "clean_convex_hull_vertices",
    "clean_ellipse_center_x",
    "clean_ellipse_center_y",
    "clean_ellipse_major_axis",
    "clean_ellipse_minor_axis",
    "clean_ellipse_angle",
    "clean_ellipse_eccentricity",
    "red_coef_trace_0.9",
    "white_coef_trace_0.9",
    "blue_coef_trace_0.9",
]


def load_data(parquet_path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} rows")
    return df


def process_cls_tokens(df: pd.DataFrame) -> np.ndarray:
    # cls_token is a numpy array in the parquet file according to my research
    tokens = np.stack(df["cls_token"].values)
    return tokens


def process_state_embeddings(df: pd.DataFrame) -> np.ndarray:
    stats = df[STATE_COLS].to_numpy().astype(np.float32)
    cls_tokens = np.stack(df["cls_token"].values).astype(np.float32)
    state = np.concatenate([stats, cls_tokens], axis=1)
    return state


def load_images(
    image_paths: List[str], target_size: Tuple[int, int] = (64, 64)
) -> List[Image.Image]:
    logger.info(f"Loading {len(image_paths)} images")
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail(target_size, Image.Resampling.LANCZOS)
            images.append(img)
        except Exception as e:
            logger.warning(f"Failed to load image at {path}: {e}")
            images.append(Image.new("RGB", target_size, color="gray"))
    return images


def plot_umap_trajectory(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    targets: List[Tuple[int, int]],
    output_path: str,
    stats: Optional[dict] = None,
    use_state: bool = False,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
):
    logger.info(f"Fitting UMAP on all embeddings (use_state={use_state})")

    # Standardize data
    if stats:
        logger.info("Using pre-calculated normalization stats")
        if use_state:
            # Construct mean and std vectors
            means = [stats[col]["mean"] for col in STATE_COLS]
            stds = [stats[col]["std"] for col in STATE_COLS]
            means.extend(stats["cls_token"]["mean"])
            stds.extend(stats["cls_token"]["std"])
        else:
            means = stats["cls_token"]["mean"]
            stds = stats["cls_token"]["std"]

        means = np.array(means, dtype=np.float32)
        stds = np.array(stds, dtype=np.float32)
        stds[stds == 0] = 1.0

        embeddings_scaled = (embeddings - means) / stds
    else:
        # logger.info(
        #     "Standardizing embeddings using StandardScaler (fitting on current data)"
        # )
        # scaler = StandardScaler()
        # embeddings_scaled = scaler.fit_transform(embeddings)
        embeddings_scaled = embeddings

    reducer = UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42
    )
    umap_2d = reducer.fit_transform(embeddings_scaled)

    # Use a generic normalization function for individual points/trajectories
    def normalize_data(data: np.ndarray) -> np.ndarray:
        if stats:
            return (data - means) / stds
        else:
            return data

    # Use the appropriate processing function
    process_fn = process_state_embeddings if use_state else process_cls_tokens

    # Add UMAP coordinates to dataframe
    df["umap_x"] = umap_2d[:, 0]
    df["umap_y"] = umap_2d[:, 1]

    fig, ax = plt.subplots(figsize=(20, 20))

    # 1. Plot all points as images (very light alpha for background)
    logger.info("Adding background images to plot")
    image_paths = df["image_path"].tolist()
    images = load_images(image_paths, target_size=(32, 32))

    df["log_clean_area"] = df["clean_area"].apply(lambda x: np.log(x + 1))
    # Set up colormap for area
    area_min = df["log_clean_area"].min()
    area_max = df["log_clean_area"].max()
    sm = plt.cm.ScalarMappable(
        cmap="viridis", norm=plt.Normalize(vmin=area_min, vmax=area_max)
    )

    # Helper to check if a row is in any target
    def is_in_targets(row):
        for exp, zone in targets:
            if row["experiment"] == exp and row["zone"] == zone:
                return True
        return False

    for i, (x, y, img) in enumerate(zip(df["umap_x"], df["umap_y"], images)):
        row = df.iloc[i]
        if is_in_targets(row):
            # Target images will be drawn later or differently?
            # Let's just draw everything with low alpha first, then highlight targets.
            alpha = 0.05
        else:
            alpha = 0.1

        color = sm.to_rgba(row["log_clean_area"])
        im = OffsetImage(img, zoom=0.5)
        ab = AnnotationBbox(
            im,
            (x, y),
            frameon=True,
            alpha=alpha,
            pad=0,
            bboxprops=dict(edgecolor=color, linewidth=2, alpha=1.0),
        )
        ax.add_artist(ab)

    # Add colorbar for area
    plt.colorbar(sm, ax=ax, label="Clean Area", pad=0.02, fraction=0.046)

    # 2. Plot trajectories for each target
    colors = plt.cm.get_cmap("tab10", len(targets))

    for target_idx, (experiment, zone) in enumerate(targets):
        color = colors(target_idx)
        logger.info(
            f"Processing trajectory for E{experiment} Z{zone} with color {color}"
        )

        target_df = df[(df["experiment"] == experiment) & (df["zone"] == zone)].copy()
        if target_df.empty:
            logger.warning(f"No data for E{experiment} Z{zone}")
            continue

        # Plot individual plant trajectories
        plant_ids = target_df["plant_id"].unique()
        for pid in plant_ids:
            plant_df = target_df[target_df["plant_id"] == pid].sort_values("wall_time")
            if len(plant_df) < 2:
                continue

            # Need to project individual points
            plant_embeddings = process_fn(plant_df)
            plant_embeddings_scaled = normalize_data(plant_embeddings)
            plant_2d = reducer.transform(plant_embeddings_scaled)

            ax.plot(
                plant_2d[:, 0],
                plant_2d[:, 1],
                color=color,
                alpha=0.2,
                linewidth=1,
                zorder=5,
            )

        # Calculate and plot average trajectory
        wall_times = sorted(target_df["wall_time"].unique())
        daily_avg_embeddings = []
        for wt in wall_times:
            day_df = target_df[target_df["wall_time"] == wt]
            day_embeddings = process_fn(day_df)
            avg_embedding = np.mean(day_embeddings, axis=0)
            daily_avg_embeddings.append(avg_embedding)

        daily_avg_embeddings = np.array(daily_avg_embeddings)
        daily_avg_embeddings_scaled = normalize_data(daily_avg_embeddings)
        daily_avg_2d = reducer.transform(daily_avg_embeddings_scaled)

        label = f"E{experiment} Z{zone}"
        # Plot average line
        ax.plot(
            daily_avg_2d[:, 0],
            daily_avg_2d[:, 1],
            color=color,
            alpha=1.0,
            linestyle="-",
            linewidth=3,
            zorder=10,
            label=label,
        )

        # Plot day circles and annotations for the average
        for i in range(len(daily_avg_2d)):
            x, y = daily_avg_2d[i]
            ax.scatter(x, y, color=color, s=150, edgecolors="white", zorder=11)

            # Only label a few days or just first/last to avoid clutter?
            # Let's label all but smaller text
            ax.text(
                x,
                y,
                f"{int(wall_times[i])}",
                color="white",
                fontsize=7,
                ha="center",
                va="center",
                fontweight="bold",
                zorder=12,
            )

            if i < len(daily_avg_2d) - 1:
                next_x, next_y = daily_avg_2d[i + 1]
                ax.annotate(
                    "",
                    xy=(next_x, next_y),
                    xytext=(x, y),
                    arrowprops=dict(
                        arrowstyle="->", color=color, lw=2, alpha=0.8, mutation_scale=15
                    ),
                    zorder=10,
                )

    ax.set_title("UMAP Projection with Trajectories Comparison")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend()

    # Set limits based on all data
    ax.set_xlim(df["umap_x"].min() - 0.5, df["umap_x"].max() + 0.5)
    ax.set_ylim(df["umap_y"].min() - 0.5, df["umap_y"].max() + 0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    logger.info(f"Saved visualization to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet", type=str, default="/data/plant-rl/offline/v21/mixed-v21.parquet"
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="14:1,14:2,14:3,14:4,14:5,14:6,14:8,14:9,14:10,14:11,14:12,15:2,15:3,15:4",
        help="Comma-separated experiment:zone pairs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR / "umap_trajectory_E14_all_zones.png"),
    )
    parser.add_argument(
        "--max_samples", type=int, default=30000, help="Limit total samples for speed"
    )
    parser.add_argument(
        "--use_state",
        action="store_true",
        help="Use full state vector (concat wall_time, clean_stats, action traces, cls_token)",
    )
    parser.add_argument(
        "--norm_stats",
        type=str,
        default="/data/plant-rl/offline/v21/normalization-stats-v21.json",
        help="Path to pre-calculated normalization stats JSON",
    )

    args = parser.parse_args()

    # Parse targets
    targets = []
    for t in args.targets.split(","):
        exp, zone = map(int, t.split(":"))
        targets.append((exp, zone))

    df = load_data(args.parquet)

    # Sample data if it's too large for quick visualization
    if args.max_samples and len(df) > args.max_samples:
        logger.info(f"Sampling {args.max_samples} points for background")
        # Ensure we keep all target experiment/zone data
        target_indices = []
        for exp, zone in targets:
            target_indices.extend(
                df[(df["experiment"] == exp) & (df["zone"] == zone)].index.tolist()
            )

        target_indices = list(set(target_indices))
        other_indices = df.index.difference(target_indices)

        # Sample from others
        num_others = max(0, args.max_samples - len(target_indices))
        if num_others < len(other_indices):
            sampled_other_indices = np.random.choice(
                other_indices, num_others, replace=False
            )
            keep_indices = np.concatenate([target_indices, sampled_other_indices])
            df = df.loc[keep_indices].reset_index(drop=True)
            logger.info(f"Reduced to {len(df)} points")

    if args.use_state:
        embeddings = process_state_embeddings(df)
    else:
        embeddings = process_cls_tokens(df)

    stats = None
    if args.norm_stats and Path(args.norm_stats).exists() and False:
        stats = load_normalization_stats(Path(args.norm_stats))

    plot_umap_trajectory(
        df, embeddings, targets, args.output, stats=stats, use_state=args.use_state
    )


if __name__ == "__main__":
    main()
