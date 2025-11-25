from itertools import product
import base64
import glob
import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import requests
from PIL import Image
from tqdm import tqdm

from config import BLUE, RED, WHITE


def compute_action_coefficients(action: np.ndarray) -> np.ndarray:
    """
    Derive action coefficients by projecting action onto the basis spanned by RED, WHITE, BLUE.

    Solves: action ≈ coef[0] * RED + coef[1] * WHITE + coef[2] * BLUE

    Args:
        action: Array of shape (6,) representing the action vector

    Returns:
        coefficients: Array of shape (3,) with [red_coef, white_coef, blue_coef]
    """
    # Create basis matrix where each column is a basis vector
    basis = np.column_stack([RED, WHITE, BLUE])  # Shape: (6, 3)

    # Solve least squares: basis @ coefficients = action
    # This finds coefficients that minimize ||action - basis @ coefficients||^2
    coefficients, residuals, rank, s = np.linalg.lstsq(basis, action, rcond=None)

    # Clip coefficients to [0, 1] range
    coefficients = np.clip(coefficients, 0.0, 1.0)

    # Normalize coefficients to sum to 1
    coefficients_sum = np.sum(coefficients)
    if coefficients_sum > 0:
        coefficients /= coefficients_sum

    return coefficients


def transform_action(df: pl.DataFrame) -> pl.DataFrame:
    # shift action backwards
    df = df.with_columns(pl.col("action.0").shift(-1).over("plant_id"))
    df = df.with_columns(pl.col("action.1").shift(-1).over("plant_id"))
    df = df.with_columns(pl.col("action.2").shift(-1).over("plant_id"))
    df = df.with_columns(pl.col("action.3").shift(-1).over("plant_id"))
    df = df.with_columns(pl.col("action.4").shift(-1).over("plant_id"))
    df = df.with_columns(pl.col("action.5").shift(-1).over("plant_id"))

    groups = df.group_by("time", "plant_id")
    groups_with_counts = groups.agg(pl.len())
    # Assert that the count for all groups is 1
    assert (groups_with_counts["len"] == 1).all()

    df2 = groups.agg(
        pl.col("action.0").mean(),
        pl.col("action.1").mean(),
        pl.col("action.2").mean(),
        pl.col("action.3").mean(),
        pl.col("action.4").mean(),
        pl.col("action.5").mean(),
        pl.col("clean_area").mean(),
    ).sort("time", "plant_id")
    df2 = df2.with_columns(
        pl.concat_arr(
            [
                pl.col("action.0"),
                pl.col("action.1"),
                pl.col("action.2"),
                pl.col("action.3"),
                pl.col("action.4"),
                pl.col("action.5"),
            ]
        ).alias("action")
    )
    df2 = df2.with_columns(
        red_diff=(pl.col("action") - RED[None])
        .arr.to_list()
        .list.eval(pl.element().abs())
        .list.sum(),
        white_diff=(pl.col("action") - WHITE[None])
        .arr.to_list()
        .list.eval(pl.element().abs())
        .list.sum(),
        blue_diff=(pl.col("action") - BLUE[None])
        .arr.to_list()
        .list.eval(pl.element().abs())
        .list.sum(),
    )

    # Compute continuous action coefficients using least squares projection
    def compute_coefficients_for_row(action_list):
        if action_list is None or len(action_list) != 6:
            return None
        action = np.array(action_list, dtype=np.float64)
        coeffs = compute_action_coefficients(action)
        return coeffs.tolist()

    df2 = df2.with_columns(
        pl.col("action")
        .map_elements(compute_coefficients_for_row, return_dtype=pl.List(pl.Float64))
        .alias("action_coefficients")
    )

    # Extract individual coefficients
    df2 = df2.with_columns(
        pl.col("action_coefficients").list.get(0).alias("red_coef"),
        pl.col("action_coefficients").list.get(1).alias("white_coef"),
        pl.col("action_coefficients").list.get(2).alias("blue_coef"),
    )

    eps = 0.1
    df2 = df2.with_columns(
        pl.when(pl.col("red_diff") < eps)
        .then(0)
        .when(pl.col("white_diff") < eps)
        .then(1)
        .when(pl.col("blue_diff") < eps)
        .then(2)
        .otherwise(None)
        .alias("discrete_action")
    )
    df = df.join(
        df2.select(
            [
                "time",
                "plant_id",
                "discrete_action",
                "action_coefficients",
                "red_coef",
                "white_coef",
                "blue_coef",
            ]
        ),
        on=["time", "plant_id"],
        how="left",
    )
    return df


def transform_reward(df):
    df = df.with_columns(
        pl.col("clean_area").shift(1).over("plant_id").alias("prev_clean_area"),
    )
    df = df.with_columns(
        (
            (pl.col("clean_area") - pl.col("prev_clean_area"))
            / pl.col("prev_clean_area")
        ).alias("reward"),
    )
    df = df.drop("prev_clean_area")
    return df


def transform_action_traces(df):
    df = df.sort("plant_id", "time")
    action_cols = [
        "action.0",
        "action.1",
        "action.2",
        "action.3",
        "action.4",
        "action.5",
        "red_coef",
        "white_coef",
        "blue_coef",
    ]
    betas = [0.5, 0.7, 0.9]
    for col, beta in product(action_cols, betas):
        alpha = 1 - beta
        df = df.with_columns(
            pl.col(col)
            .ewm_mean(alpha=alpha, adjust=True)
            .over("experiment", "zone", "plant_id")
            .alias(f"{col}_trace_{beta}"),
        )
    # Create one-hot for discrete_action
    df = df.with_columns(
        pl.when(pl.col("discrete_action") == 0)
        .then(1.0)
        .otherwise(0.0)
        .alias("discrete_action_0"),
        pl.when(pl.col("discrete_action") == 1)
        .then(1.0)
        .otherwise(0.0)
        .alias("discrete_action_1"),
        pl.when(pl.col("discrete_action") == 2)
        .then(1.0)
        .otherwise(0.0)
        .alias("discrete_action_2"),
    )
    for beta in betas:
        alpha = 1 - beta
        df = df.with_columns(
            pl.col("discrete_action_0")
            .ewm_mean(alpha=alpha, adjust=True)
            .over("experiment", "zone", "plant_id")
            .alias(f"discrete_action_trace_0_{beta}"),
            pl.col("discrete_action_1")
            .ewm_mean(alpha=alpha, adjust=True)
            .over("experiment", "zone", "plant_id")
            .alias(f"discrete_action_trace_1_{beta}"),
            pl.col("discrete_action_2")
            .ewm_mean(alpha=alpha, adjust=True)
            .over("experiment", "zone", "plant_id")
            .alias(f"discrete_action_trace_2_{beta}"),
        )
    return df


def transform_state(df):
    df = df.with_columns(
        pl.col("clean_area")
        .mean()
        .over("experiment", "zone", "time")
        .alias("mean_clean_area"),
    )
    return df


def transform_outlier_detection(
    df: pl.DataFrame, q1: float = 0.01, q2: float = 0.99
) -> pl.DataFrame:
    """Mark rows with outlier rewards outside specified percentile thresholds.

    Args:
        df: DataFrame with a "reward" column
        q1: Lower percentile threshold
        q2: Upper percentile threshold

    Returns:
        DataFrame with added "valid" column indicating non-outlier rows
    """
    ql = df["reward"].quantile(q1)
    qu = df["reward"].quantile(q2)
    df = df.with_columns(
        ((pl.col("reward") < ql) | (pl.col("reward") > qu)).alias("outlier")
    )
    invalid_count = df["outlier"].sum()
    print(f"marked {invalid_count} / {df.height} daily rows as outliers")
    return df


# Image processing configuration
PIPELINE_URL = "http://pipeline:8000"  # On plant-network
EMBEDDINGS_URL = "http://embeddings:8000"  # On plant-network
DATA_DIR = Path("/data")  # /data/plant-rl is mounted as /data in container
OFFLINE_DIR = DATA_DIR / "offline"
ONLINE_DIR = DATA_DIR / "online"

# Pipeline parameters
POT_DETECTION_PROMPT = "pot"
POT_DETECTION_THRESHOLD = 0.03
WARP_MARGIN = 0.25

logger = logging.getLogger(__name__)


def encode_image(image: Image.Image) -> str:
    """Encode PIL Image to base64."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def decode_image(image_data: str) -> Image.Image:
    """Decode base64 to PIL Image."""
    image_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def find_image_path(experiment: int, zone: int, timestamp: datetime) -> Optional[Path]:
    """Find the image file corresponding to a dataset row.

    Args:
        experiment: Experiment ID (e.g., 11, 12, 13)
        zone: Zone ID (e.g., 1, 2, 3)
        timestamp: Timestamp from the dataset (in America/Edmonton timezone)

    Returns:
        Path to the image file, or None if not found
    """
    if timestamp.tzinfo is None:
        # If no timezone, assume America/Edmonton
        timestamp = timestamp.replace(tzinfo=ZoneInfo("America/Edmonton"))

    # Convert to UTC for filename matching
    timestamp_utc = timestamp.astimezone(ZoneInfo("UTC"))
    timestamp_str = timestamp_utc.strftime("%Y-%m-%dT%H%M00+0000")

    # Search for the image in the zone directory
    zone_glob = f"{ONLINE_DIR}/E{experiment}/P1/*/alliance-zone{zone:02}/images/{timestamp_str}_left.jpg"
    matches = glob.glob(zone_glob)

    if matches:
        return Path(matches[0])

    logger.warning(f"Image not found: E{experiment}/zone{zone} at {timestamp_str}")
    return None


def detect_pots_reference(image_path: Path, zone_key: tuple) -> Optional[dict]:
    """Detect pots in a reference image to get quadrilaterals.

    Args:
        image_path: Path to the reference image
        zone_key: Tuple of (experiment, zone) for saving visualization

    Returns:
        Dictionary with detection results (boxes, quadrilaterals, etc.), or None if failed
    """
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        image_data = encode_image(image)

        # Call full pipeline to get pot positions with visualization
        response = requests.post(
            f"{PIPELINE_URL}/pot/pipeline",
            json={
                "image_data": image_data,
                "text_prompt": POT_DETECTION_PROMPT,
                "threshold": POT_DETECTION_THRESHOLD,
                "margin": WARP_MARGIN,
                "visualize": True,  # Enable visualization for debugging
            },
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()

        # Save visualization if available
        if "visualization" in result and result["visualization"]:
            try:
                viz_dir = OFFLINE_DIR / "visualizations"
                viz_dir.mkdir(exist_ok=True)
                viz_image = decode_image(result["visualization"])
                viz_path = viz_dir / f"E{zone_key[0]}_zone{zone_key[1]}_reference.jpg"
                viz_image.save(viz_path)
                logger.info(f"Saved visualization to {viz_path}")
            except Exception as e:
                logger.warning(f"Failed to save visualization: {e}")

        return result
    except Exception as e:
        logger.error(f"Pot detection failed for {image_path}: {e}")
        return None


def warp_with_quadrilaterals(image_path: Path, quadrilaterals: list) -> Optional[list]:
    """Warp an image using pre-computed quadrilaterals.

    Args:
        image_path: Path to the image file
        quadrilaterals: List of quadrilaterals from reference image

    Returns:
        List of warped pot images (base64), or None if failed
    """
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        image_data = encode_image(image)

        # Call warp endpoint with pre-computed quadrilaterals
        response = requests.post(
            f"{PIPELINE_URL}/pot/warp",
            json={
                "image_data": image_data,
                "quadrilaterals": quadrilaterals,
                "margin": WARP_MARGIN,
                "output_size": None,
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()

        return result.get("warped_images", [])
    except Exception as e:
        logger.error(f"Warping failed for {image_path}: {e}")
        return None


def generate_embedding(image_data: str) -> Optional[list]:
    """Generate DINOv3 embedding for an image.

    Args:
        image_data: Base64-encoded image

    Returns:
        768-dimensional embedding vector, or None if failed
    """
    try:
        response = requests.post(
            f"{EMBEDDINGS_URL}/predict",
            json={
                "image_data": image_data,
                "embedding_types": ["cls_token"],
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()

        return result["cls_token"]
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return None


def process_zone_images(zone_key: tuple, zone_images: list) -> list:
    """Process all images for a single zone using reference image optimization.

    Args:
        zone_key: Tuple of (experiment, zone)
        zone_images: List of dicts with 'time' and 'image_path' keys (sorted by time)

    Returns:
        List of result dictionaries for all detected plants across all images
    """
    experiment, zone = zone_key
    results = []

    if not zone_images:
        return results

    # Find the first 9:30 AM image as reference (daylight image)
    # Use America/Edmonton timezone to ensure we pick daylight hours
    target_tz = ZoneInfo("America/Edmonton")
    reference_image = None

    for img_info in zone_images:
        timestamp = img_info["time"]
        # Convert to target timezone if needed
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=target_tz)
        else:
            timestamp = timestamp.astimezone(target_tz)

        # Check if this is a 9:30 AM image
        if timestamp.hour == 9 and timestamp.minute == 30:
            reference_image = img_info
            break

    # Fallback to first image if no 9:30 AM image found
    if reference_image is None:
        reference_image = zone_images[0]
        logger.warning(
            f"E{experiment}/zone{zone}: No 9:30 AM image found in {target_tz}, using first image at {reference_image['time']}"
        )
    else:
        logger.info(
            f"E{experiment}/zone{zone}: Using 9:30 AM reference image at {reference_image['time']} ({target_tz})"
        )

    # Detect pots in reference image
    detection_result = detect_pots_reference(reference_image["image_path"], zone_key)
    if detection_result is None:
        logger.warning(
            f"E{experiment}/zone{zone}: Failed to detect pots in reference image"
        )
        return results

    quadrilaterals = detection_result.get("quadrilaterals", [])
    if not quadrilaterals:
        logger.warning(f"E{experiment}/zone{zone}: No pots detected in reference image")
        return results

    # Log diagnostic info for investigation
    num_detected = len(quadrilaterals)
    logger.info(
        f"E{experiment}/zone{zone}: Detected {num_detected} pots (expected 18), processing {len(zone_images)} images"
    )
    if num_detected != 18:
        logger.warning(
            f"E{experiment}/zone{zone}: Pot count mismatch! Expected 18, got {num_detected}. Check visualization at /data/offline/visualizations/E{experiment}_zone{zone}_reference.jpg"
        )

    # Create output directory for processed images
    # Format: E{exp}/Z{zone:02d}/images/
    processed_dir = (
        OFFLINE_DIR / "processed" / f"E{experiment}" / f"Z{zone:02d}" / "images"
    )
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Process all images in this zone using the same quadrilaterals
    for img_info in zone_images:
        timestamp = img_info["time"]
        image_path = img_info["image_path"]

        # Format timestamp for filename
        timestamp_str = timestamp.strftime("%Y-%m-%dT%H%M%S")

        # Warp using pre-computed quadrilaterals
        warped_images = warp_with_quadrilaterals(image_path, quadrilaterals)
        if warped_images is None:
            logger.warning(f"E{experiment}/zone{zone} at {timestamp}: Warping failed")
            continue

        # Generate embeddings and save images for each pot
        for plant_id, warped_b64 in enumerate(warped_images):
            if warped_b64 is None:
                embedding = None
                image_file_path = None
            else:
                # Generate embedding
                embedding = generate_embedding(warped_b64)

                # Save warped pot image
                try:
                    warped_image = decode_image(warped_b64)
                    image_filename = f"{timestamp_str}_plant{plant_id:02d}.jpg"
                    image_file_path = processed_dir / image_filename
                    warped_image.save(image_file_path)
                    # Store relative path from /data/offline
                    image_file_path = str(image_file_path.relative_to(OFFLINE_DIR))
                except Exception as e:
                    logger.warning(
                        f"Failed to save pot image for E{experiment}/zone{zone} plant{plant_id} at {timestamp}: {e}"
                    )
                    image_file_path = None

            results.append(
                {
                    "experiment": experiment,
                    "zone": zone,
                    "time": timestamp,
                    "plant_id": plant_id,
                    "embedding": embedding,
                    "image_path": image_file_path,  # Add image path reference
                }
            )

    return results


def transform_image_embeddings(df: pl.DataFrame, max_workers: int = 8) -> pl.DataFrame:
    """Transform to add image embeddings to a daily dataset.

    This function:
    1. Finds corresponding image files for each (experiment, zone, time) combination
    2. Detects pots in reference images (9:30 AM daylight images)
    3. Warps all images using the detected pot positions
    4. Generates DINOv3 embeddings for each cropped pot
    5. Saves processed pot images
    6. Returns a new DataFrame with plant_id, embedding, and image_path columns

    Args:
        df: DataFrame with experiment, zone, and time columns
        max_workers: Number of parallel workers for zone processing

    Returns:
        DataFrame with added plant_id, embedding, and image_path columns.
        The returned DataFrame will have more rows than the input (one per detected plant).
    """
    logger.info(f"Starting image embedding transform on {len(df)} rows")

    # Get unique images (experiment, zone, time combinations)
    unique_images = (
        df.select(["experiment", "zone", "time"])
        .unique()
        .sort("experiment", "zone", "time")
    )

    # Group images by (experiment, zone) and find corresponding image paths
    zone_groups = {}
    for row in unique_images.iter_rows(named=True):
        experiment = row["experiment"]
        zone = row["zone"]
        timestamp = row["time"]

        # Find the image file
        image_path = find_image_path(experiment, zone, timestamp)
        if image_path is None:
            continue

        zone_key = (experiment, zone)
        if zone_key not in zone_groups:
            zone_groups[zone_key] = []

        zone_groups[zone_key].append(
            {
                "time": timestamp,
                "image_path": image_path,
            }
        )

    logger.info(
        f"Processing {len(zone_groups)} zones with {len(unique_images)} total images"
    )
    logger.info(f"Using {max_workers} parallel workers")

    # Store results for each detected plant
    all_results = []

    # Process zones in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all zone processing tasks
        future_to_zone = {
            executor.submit(process_zone_images, zone_key, images): zone_key
            for zone_key, images in zone_groups.items()
        }

        # Collect results as they complete
        with tqdm(total=len(zone_groups), desc="Processing zones") as pbar:
            for future in as_completed(future_to_zone):
                zone_key = future_to_zone[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    logger.info(
                        f"E{zone_key[0]}/zone{zone_key[1]}: Generated {len(results)} plant embeddings"
                    )
                except Exception as e:
                    logger.error(
                        f"Error processing E{zone_key[0]}/zone{zone_key[1]}: {e}"
                    )
                finally:
                    pbar.update(1)

    # Create new dataframe with embeddings
    logger.info(f"Creating new dataset with {len(all_results)} plant detections")
    df_new = pl.DataFrame(all_results)

    # Get the original dataset columns we want to preserve (excluding plant_id, image_path, embedding which we're adding/reassigning)
    # We'll join on (experiment, zone, time) and take the first match for shared columns
    original_cols_to_keep = [
        col
        for col in df.columns
        if col
        not in ["plant_id", "embedding", "image_path", "experiment", "zone", "time"]
    ]

    # For each (experiment, zone, time), get one representative row from original dataset
    df_metadata = df.select(
        ["experiment", "zone", "time"] + original_cols_to_keep
    ).unique(subset=["experiment", "zone", "time"], keep="first")

    # Join the new detections with metadata
    df_with_embeddings = df_new.join(
        df_metadata,
        on=["experiment", "zone", "time"],
        how="left",
    )

    # Reorder columns to put key columns first
    key_cols = ["experiment", "zone", "time", "plant_id", "image_path", "embedding"]
    other_cols = [col for col in df_with_embeddings.columns if col not in key_cols]
    df_with_embeddings = df_with_embeddings.select(key_cols + other_cols)

    # Convert embeddings to Array[Float32] for efficiency
    logger.info("Converting embeddings to Array[Float32]")
    df_with_embeddings = df_with_embeddings.with_columns(
        pl.col("embedding").list.eval(pl.element().cast(pl.Float32)).alias("embedding")
    )

    # Print statistics
    total_embeddings = df_new.filter(pl.col("embedding").is_not_null()).height
    total_plants = len(df_new)
    logger.info(f"Successfully generated {total_embeddings}/{total_plants} embeddings")
    logger.info(f"Original dataset had {len(df)} rows")
    logger.info(
        f"New dataset has {len(df_with_embeddings)} rows (based on plant-cv detections)"
    )

    # Diagnostic summary for investigation
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC SUMMARY - Image Processing")
    logger.info("=" * 60)

    # Analyze zones by pot count
    zone_stats = (
        df_new.group_by(["experiment", "zone"])
        .agg(
            [
                pl.col("plant_id").max().alias("max_plant_id"),
                pl.col("time").n_unique().alias("num_images"),
            ]
        )
        .sort("experiment", "zone")
    )

    logger.info("\nZone-by-zone pot counts:")
    for row in zone_stats.iter_rows(named=True):
        num_pots = row["max_plant_id"] + 1 if row["max_plant_id"] is not None else 0
        status = "✓" if num_pots == 18 else "⚠"
        logger.info(
            f"  {status} E{row['experiment']}/zone{row['zone']}: {num_pots} pots detected across {row['num_images']} images"
        )

    # Count zones with issues
    zones_with_issues = zone_stats.filter(
        (pl.col("max_plant_id").is_null()) | ((pl.col("max_plant_id") + 1) != 18)
    )
    logger.info(f"\nZones with issues: {len(zones_with_issues)}/{len(zone_stats)}")
    logger.info(
        f"Expected total rows if all zones had 18 pots: {len(unique_images) * 18}"
    )
    logger.info(f"Actual rows: {len(df_with_embeddings)}")
    logger.info(
        f"Difference: {len(unique_images) * 18 - len(df_with_embeddings)} missing detections"
    )
    logger.info(
        "\nCheck visualization files in /data/offline/visualizations/ for debugging"
    )
    logger.info("=" * 60 + "\n")

    return df_with_embeddings
