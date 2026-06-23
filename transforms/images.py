import base64
import glob
import io
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import polars as pl
import requests
from PIL import Image
from tqdm import tqdm

from config import VISION_VERSION
from transforms.cache import DiskCache

PIPELINE_URL = "http://localhost:8800"
DATA_DIR = Path("/data/plant-rl")
OFFLINE_DIR = DATA_DIR / "offline"
ONLINE_DIR = DATA_DIR / "online"

logger = logging.getLogger(__name__)

thread_local = threading.local()


def get_session() -> requests.Session:
    """Get a thread-local requests session."""
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session


def encode_image(image: Image.Image) -> str:
    """Encode PIL Image to base64."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def decode_image(image_data: str) -> Image.Image:
    """Decode base64 to PIL Image."""
    image_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def find_image_path(experiment: int, zone: int, image_name: str) -> Optional[Path]:
    """Find the image file corresponding to a dataset row.

    Args:
        experiment: Experiment ID (e.g., 11, 12, 13)
        zone: Zone ID (e.g., 1, 2, 3)
        image_name: Image name from the dataset (e.g., "2025-08-20T153000+0000_left.jpg")

    Returns:
        Path to the image file, or None if not found
    """
    # Search for the image in the zone directory
    zone_glob = (
        f"{ONLINE_DIR}/E{experiment}/P1/*/alliance-zone{zone:02}/images/{image_name}"
    )
    matches = glob.glob(zone_glob)

    if matches:
        return Path(matches[0])

    logger.warning(f"Image not found: E{experiment}/zone{zone} at {image_name}")
    return None


def process_zone_images(
    zone_df_subset: pl.DataFrame, output_dir: Optional[Path] = None
) -> pl.DataFrame:
    """Process all images for a single zone using sequential tracking.

    Args:
        zone_df_subset: DataFrame containing (experiment, zone, time, image_name) for one zone.
        output_dir: Optional output directory for processed images and visualizations

    Returns:
        Polars DataFrame of result dictionaries for all detected plants across all images
    """
    if zone_df_subset.is_empty():
        return pl.DataFrame()

    # Get experiment and zone from first row (assuming all rows are same zone)
    first_row = zone_df_subset.row(0, named=True)
    experiment = first_row["experiment"]
    zone = first_row["zone"]

    # Identify images to process
    images_info = []
    for row in zone_df_subset.select("time", "image_name").iter_rows(named=True):
        path = find_image_path(experiment, zone, row["image_name"])
        if path:
            images_info.append(
                {
                    "experiment": experiment,
                    "zone": zone,
                    "time": row["time"],
                    "image_path": path,
                }
            )

    if not images_info:
        return pl.DataFrame()

    # Sort images by time for sequential tracking
    images_info = sorted(images_info, key=lambda x: x["time"])
    num_images = len(images_info)

    # Load cache with strict count check
    cache = DiskCache(OFFLINE_DIR, VISION_VERSION)
    cache_df = cache.load(experiment, zone, num_images)

    if cache_df is not None:
        logger.info(
            f"E{experiment}/zone{zone}: Using full cache ({len(cache_df)} rows)"
        )
        return cache_df

    images_to_process = images_info
    logger.info(
        f"E{experiment}/zone{zone}: No valid cache found, processing all {len(images_to_process)} images"
    )

    # Determine base directory for saving images
    if output_dir:
        images_base_dir = output_dir / "images"
    else:
        images_base_dir = (
            OFFLINE_DIR / "processed" / f"E{experiment}" / f"Z{zone:02d}" / "images"
        )
    images_base_dir.mkdir(parents=True, exist_ok=True)

    pots_results = []
    state = None
    associations = {}

    logger.info(f"E{experiment}/zone{zone}: Starting sequential tracking")
    t_start = time.time()

    for i, img_info in enumerate(
        tqdm(images_to_process, desc=f"E{experiment}/Z{zone}")
    ):
        try:
            timestamp = img_info["time"]
            timestamp_str = timestamp.strftime("%Y-%m-%dT%H%M%S")
            image_path = img_info["image_path"]

            # Load and check brightness to skip dark images
            import numpy as np

            with Image.open(image_path) as img:
                img_gray = img.convert("L")
                mean_brightness = np.mean(np.array(img_gray))

            if mean_brightness < 90.0:
                logger.info(
                    f"E{experiment}/zone{zone}: Skipping dark image (brightness {mean_brightness:.2f}): {image_path.name}"
                )
                if pots_results:
                    last_time = pots_results[-1]["time"]
                    last_frame_pots = [
                        row for row in pots_results if row["time"] == last_time
                    ]
                    current_frame_pots = []
                    for row in last_frame_pots:
                        new_row = row.copy()
                        new_row["time"] = timestamp
                        current_frame_pots.append(new_row)
                    pots_results.extend(current_frame_pots)
                continue

            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            # Call Pipeline API
            if state is None:
                payload = {"image_data": image_data}
                url = f"{PIPELINE_URL}/pipeline/detect"
            else:
                payload = {
                    "image_data": image_data,
                    "state": state,
                    "associations": associations,
                }
                url = f"{PIPELINE_URL}/pipeline/propagate"

            response = get_session().post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()

            # Update tracking state
            state = result.get("state")
            associations = result.get("associations", {})
            plant_stats = result.get("plant_stats", {})
            ordered_pot_ids = result.get("ordered_pot_ids", [])

            # Parallelize embedding and saving for the current frame's pots
            current_frame_pots = []

            def process_pot(pot_id_str):
                stats = plant_stats.get(pot_id_str)
                if not stats:
                    return None

                warped_b64 = stats.get("warped_image")
                if not warped_b64:
                    return None

                # 1. Get Embedding from stats
                cls_token = stats.get("cls_token")

                # 2. Save Warped Image
                pot_id = int(pot_id_str)
                plant_dir = images_base_dir / str(pot_id)
                plant_dir.mkdir(parents=True, exist_ok=True)

                fname = f"{timestamp_str}.jpg"
                fpath = plant_dir / fname
                try:
                    with open(fpath, "wb") as f:
                        f.write(base64.b64decode(warped_b64))
                    rel_path = str(
                        fpath.relative_to(output_dir if output_dir else OFFLINE_DIR)
                    )
                except Exception as e:
                    logger.warning(f"Failed to save image for pot {pot_id}: {e}")
                    rel_path = None

                # Construct result row
                res_row = {
                    "experiment": experiment,
                    "zone": zone,
                    "time": timestamp,
                    "plant_id": pot_id,
                    "cls_token": cls_token,
                    "image_path": rel_path,
                }
                # Add stats (area, intensity, etc.)
                for k, v in stats.items():
                    if k not in ["warped_image", "plant_id"]:
                        res_row[k] = v
                return res_row

            # Process pots in parallel for this frame
            with ThreadPoolExecutor(max_workers=64) as executor:
                futures = [
                    executor.submit(process_pot, str(pid)) for pid in ordered_pot_ids
                ]
                for future in as_completed(futures):
                    res = future.result()
                    if res:
                        current_frame_pots.append(res)

            pots_results.extend(current_frame_pots)

            # Optional: Save visualization if provided
            viz_data = result.get("visualization_data")
            if viz_data:
                viz_dir = (
                    (output_dir / "visualizations")
                    if output_dir
                    else (OFFLINE_DIR / "visualizations")
                )
                viz_dir.mkdir(parents=True, exist_ok=True)
                viz_path = viz_dir / f"E{experiment}_Z{zone}_{timestamp_str}_viz.jpg"
                try:
                    with open(viz_path, "wb") as f:
                        f.write(base64.b64decode(viz_data))
                except Exception as e:
                    logger.warning(f"Failed to save visualization: {e}")

        except Exception as e:
            logger.error(f"Failed to process frame {i} for E{experiment}/Z{zone}: {e}")
            # Reset state on failure to force a new /detect on next frame
            state = None

    logger.info(
        f"E{experiment}/zone{zone}: Processed {len(images_to_process)} images in {time.time() - t_start:.2f}s"
    )

    combined_df = (
        pl.DataFrame(pots_results, infer_schema_length=100000)
        if pots_results
        else pl.DataFrame()
    )

    if not combined_df.is_empty():
        combined_df = combined_df.sort("time", "plant_id")
        # Save updated cache with count
        cache.save(experiment, zone, combined_df, num_images)

    return combined_df


def transform_image_embeddings(
    df: pl.DataFrame, output_dir: Optional[Path] = None
) -> pl.DataFrame:
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
        output_dir: Optional output directory for processed images and visualizations.
                   If provided, images will be saved to output_dir/images/ and
                   visualizations to output_dir/visualizations/

    Returns:
        DataFrame with added plant_id, embedding, and image_path columns.
        The returned DataFrame will have more rows than the input (one per detected plant).
    """
    logger.info(f"Starting image embedding transform on {len(df)} rows")

    # We partition by experiment/zone to process each zone cache independently
    # Note: partition_by returns a list of dataframes
    zone_dfs = df.partition_by(["experiment", "zone"], maintain_order=True)

    result_dfs = []
    logger.info(f"Processing {len(zone_dfs)} zones...")

    for zone_df in tqdm(zone_dfs):
        processed_zone_df = process_zone_images(zone_df, output_dir)
        if not processed_zone_df.is_empty():
            result_dfs.append(processed_zone_df)

    if not result_dfs:
        logger.warning("No image results generated.")
        # Return original (without embeddings columns) or error?
        # If truly empty, we can return df but it lacks embeddings.
        return df

    all_results_df = pl.concat(result_dfs, how="diagonal")

    # Get the original dataset columns we want to preserve
    # Exclude: join keys, columns we're adding/reassigning, and stats columns that are now per-plant
    # The stats columns in df_new are dynamically determined, so we exclude all cols present in df_new
    # EXCEPT keys.

    keys = {"experiment", "zone", "time"}
    new_cols = set(all_results_df.columns)

    # Columns to keep from ORIGINAL df: everything NOT in new_cols, plus keys
    cols_to_keep = [c for c in df.columns if c not in new_cols or c in keys]

    # For each (experiment, zone, time), get one representative row from original dataset
    df_metadata = df.select(cols_to_keep).unique(
        subset=["experiment", "zone", "time"], keep="first"
    )

    # Join the new detections with metadata
    df_with_embeddings = all_results_df.join(
        df_metadata,
        on=["experiment", "zone", "time"],
        how="left",
    )

    return df_with_embeddings
