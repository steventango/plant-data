import base64
import glob
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo
import threading

import polars as pl
import requests
from PIL import Image
from tqdm import tqdm


PIPELINE_URL = "http://localhost:8800"
EMBEDDINGS_URL = "http://localhost:8803"
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

        # 1. Detect pots
        detect_response = get_session().post(
            f"{PIPELINE_URL}/pot/detect",
            json={
                "image_data": image_data,
                "visualize": True,
            },
            timeout=30,
        )
        detect_response.raise_for_status()
        detect_result = detect_response.json()
        boxes = detect_result.get("boxes", [])

        if not boxes:
            logger.warning(f"No pots detected in {image_path}")
            return None

        logger.info(f"Detected {len(boxes)} pots in {image_path}")

        # Save detect visualization if available
        if "visualization" in detect_result and detect_result["visualization"]:
            try:
                viz_dir = OFFLINE_DIR / "visualizations"
                viz_dir.mkdir(exist_ok=True)
                viz_image = decode_image(detect_result["visualization"])
                viz_path = viz_dir / f"E{zone_key[0]}_zone{zone_key[1]}_detect.jpg"
                viz_image.save(viz_path)
                logger.info(f"Saved detect visualization to {viz_path}")
            except Exception as e:
                logger.warning(f"Failed to save detect visualization: {e}")

        # 2. Segment pots
        segment_response = get_session().post(
            f"{PIPELINE_URL}/pot/segment",
            json={
                "image_data": image_data,
                "boxes": boxes,
                "visualize": True,
            },
            timeout=60,
        )
        segment_response.raise_for_status()
        segment_result = segment_response.json()
        masks = segment_result.get("masks")

        if not masks:
            logger.warning(f"No masks generated for {image_path}")
            return None

        # Save segment visualization if available
        if "visualization" in segment_result and segment_result["visualization"]:
            try:
                viz_dir = OFFLINE_DIR / "visualizations"
                viz_dir.mkdir(exist_ok=True)
                viz_image = decode_image(segment_result["visualization"])
                viz_path = viz_dir / f"E{zone_key[0]}_zone{zone_key[1]}_segment.jpg"
                viz_image.save(viz_path)
                logger.info(f"Saved segment visualization to {viz_path}")
            except Exception as e:
                logger.warning(f"Failed to save segment visualization: {e}")

        # 3. Compute quadrilaterals
        quad_response = get_session().post(
            f"{PIPELINE_URL}/pot/quad",
            json={
                "masks": masks,
                "image_data": image_data,
                "visualize": True,
            },
            timeout=30,
        )
        quad_response.raise_for_status()
        quad_result = quad_response.json()
        quadrilaterals = quad_result.get("quadrilaterals", [])

        # Save quad visualization if available
        if "visualization" in quad_result and quad_result["visualization"]:
            try:
                viz_dir = OFFLINE_DIR / "visualizations"
                viz_dir.mkdir(exist_ok=True)
                viz_image = decode_image(quad_result["visualization"])
                viz_path = viz_dir / f"E{zone_key[0]}_zone{zone_key[1]}_quad.jpg"
                viz_image.save(viz_path)
                logger.info(f"Saved quad visualization to {viz_path}")
            except Exception as e:
                logger.warning(f"Failed to save quad visualization: {e}")

        return {"quadrilaterals": quadrilaterals}
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
        response = get_session().post(
            f"{PIPELINE_URL}/pot/warp",
            json={
                "image_data": image_data,
                "quadrilaterals": quadrilaterals,
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
        response = get_session().post(
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


def detect_plant(
    warped_image_b64: str,
    zone_key: tuple = None,
    plant_id: int = None,
    timestamp_str: str = None,
) -> Optional[dict]:
    """Detect plant in a warped pot image."""
    try:
        response = get_session().post(
            f"{PIPELINE_URL}/plant/detect",
            json={"image_data": warped_image_b64, "visualize": True},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()

        # Save detect visualization if available
        if (
            zone_key
            and plant_id is not None
            and timestamp_str
            and "visualization" in result
            and result["visualization"]
        ):
            try:
                viz_dir = OFFLINE_DIR / "visualizations" / "plant_detect"
                viz_dir.mkdir(parents=True, exist_ok=True)
                viz_image = decode_image(result["visualization"])
                viz_path = (
                    viz_dir
                    / f"E{zone_key[0]}_zone{zone_key[1]}_plant{plant_id:02d}_{timestamp_str}_detect.jpg"
                )
                viz_image.save(viz_path)
                logger.debug(f"Saved plant detect visualization to {viz_path}")
            except Exception as e:
                logger.warning(f"Failed to save plant detect visualization: {e}")

        return result
    except Exception as e:
        logger.error(f"Plant detection failed: {e}")
        return None


def segment_plant(
    warped_image_b64: str,
    boxes: list,
    confidences: list,
    zone_key: tuple = None,
    plant_id: int = None,
    timestamp_str: str = None,
) -> Optional[dict]:
    """Segment plant given detection boxes."""
    try:
        response = get_session().post(
            f"{PIPELINE_URL}/plant/segment",
            json={
                "image_data": warped_image_b64,
                "boxes": boxes,
                "confidences": confidences,
                "visualize": True,
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()

        # Save segment visualization if available
        if (
            zone_key
            and plant_id is not None
            and timestamp_str
            and "visualization" in result
            and result["visualization"]
        ):
            try:
                viz_dir = OFFLINE_DIR / "visualizations" / "plant_segment"
                viz_dir.mkdir(parents=True, exist_ok=True)
                viz_image = decode_image(result["visualization"])
                viz_path = (
                    viz_dir
                    / f"E{zone_key[0]}_zone{zone_key[1]}_plant{plant_id:02d}_{timestamp_str}_segment.jpg"
                )
                viz_image.save(viz_path)
                logger.debug(f"Saved plant segment visualization to {viz_path}")
            except Exception as e:
                logger.warning(f"Failed to save plant segment visualization: {e}")

        return result
    except Exception as e:
        logger.error(f"Plant segmentation failed: {e}")
        return None


def compute_plant_stats(
    warped_image_b64: str,
    mask_b64: str,
    zone_key: tuple = None,
    plant_id: int = None,
    timestamp_str: str = None,
) -> Optional[dict]:
    """Compute plant statistics given mask."""
    try:
        response = get_session().post(
            f"{PIPELINE_URL}/plant/stats",
            json={
                "warped_image": warped_image_b64,
                "mask": mask_b64,
                "pot_size_mm": 60.0,
                "margin": 0.25,
                "visualize": True,
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()

        # Save stats visualization if available
        if (
            zone_key
            and plant_id is not None
            and timestamp_str
            and "visualization" in result
            and result["visualization"]
        ):
            try:
                viz_dir = OFFLINE_DIR / "visualizations" / "plant_stats"
                viz_dir.mkdir(parents=True, exist_ok=True)
                viz_image = decode_image(result["visualization"])
                viz_path = (
                    viz_dir
                    / f"E{zone_key[0]}_zone{zone_key[1]}_plant{plant_id:02d}_{timestamp_str}_stats.jpg"
                )
                viz_image.save(viz_path)
                logger.debug(f"Saved plant stats visualization to {viz_path}")
            except Exception as e:
                logger.warning(f"Failed to save plant stats visualization: {e}")

        return result
    except Exception as e:
        logger.error(f"Plant stats failed: {e}")
        return None


def process_zone_images(zone_key: tuple, zone_images: list) -> list:
    """Process all images for a single zone using block-based execution.

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
    target_tz = ZoneInfo("America/Edmonton")
    reference_image = None

    for img_info in zone_images:
        timestamp = img_info["time"]
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=target_tz)
        else:
            timestamp = timestamp.astimezone(target_tz)

        if timestamp.hour == 9 and timestamp.minute == 30:
            reference_image = img_info
            break

    if reference_image is None:
        reference_image = zone_images[0]
        logger.warning(
            f"E{experiment}/zone{zone}: No 9:30 AM image found, using first image"
        )
    else:
        logger.info(
            f"E{experiment}/zone{zone}: Using 9:30 AM reference image at {reference_image['image_path']}"
        )

    # Detect pots in reference image
    t0 = time.time()
    detection_result = detect_pots_reference(reference_image["image_path"], zone_key)
    logger.info(
        f"E{experiment}/zone{zone}: Reference detection took {time.time() - t0:.2f}s"
    )
    if detection_result is None:
        logger.warning(
            f"E{experiment}/zone{zone}: Failed to detect pots in reference image"
        )
        return results

    quadrilaterals = detection_result.get("quadrilaterals", [])
    if not quadrilaterals:
        logger.warning(f"E{experiment}/zone{zone}: No pots detected in reference image")
        return results

    # Create output directory
    processed_dir = (
        OFFLINE_DIR / "processed" / f"E{experiment}" / f"Z{zone:02d}" / "images"
    )
    processed_dir.mkdir(parents=True, exist_ok=True)

    # --- BLOCK 1: WARP ALL IMAGES ---
    logger.info(f"E{experiment}/zone{zone}: Starting WARP block")
    t0_warp = time.time()
    pots = []  # List of dicts representing each pot

    def warp_task(img_info):
        timestamp = img_info["time"]
        image_path = img_info["image_path"]
        timestamp_str = timestamp.strftime("%Y-%m-%dT%H%M%S")
        warped_images = warp_with_quadrilaterals(image_path, quadrilaterals)
        if warped_images:
            return [
                {
                    "experiment": experiment,
                    "zone": zone,
                    "time": timestamp,
                    "plant_id": i,
                    "warped_b64": b64,
                    "timestamp_str": timestamp_str,
                }
                for i, b64 in enumerate(warped_images)
            ]
        return []

    with ThreadPoolExecutor(max_workers=256) as executor:
        futures = [executor.submit(warp_task, img) for img in zone_images]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"E{experiment}/Z{zone} Warping",
            leave=False,
        ):
            pots.extend(future.result())

    logger.info(
        f"E{experiment}/zone{zone}: Warped {len(pots)} pots in {time.time() - t0_warp:.2f}s"
    )

    if not pots:
        return []

    # --- BLOCK 2: EMBED ALL POTS ---
    logger.info(f"E{experiment}/zone{zone}: Starting EMBED block")
    t0_embed = time.time()

    def embed_task(pot):
        pot["embedding"] = generate_embedding(pot["warped_b64"])
        return pot

    with ThreadPoolExecutor(max_workers=8) as executor:
        # Update pots in place (or create new list)
        # We use map to keep order if needed, but list(executor.map) is fine
        pots = list(
            tqdm(
                executor.map(embed_task, pots),
                total=len(pots),
                desc=f"E{experiment}/Z{zone} Embedding",
                leave=False,
            )
        )

    logger.info(
        f"E{experiment}/zone{zone}: Embedded {len(pots)} pots in {time.time() - t0_embed:.2f}s"
    )

    # --- BLOCK 3: DETECT ALL PLANTS ---
    logger.info(f"E{experiment}/zone{zone}: Starting DETECT block")
    t0_detect = time.time()

    def detect_task(pot):
        result = detect_plant(
            pot["warped_b64"],
            zone_key=zone_key,
            plant_id=pot["plant_id"],
            timestamp_str=pot["timestamp_str"],
        )
        if result:
            pot["boxes"] = result.get("boxes", [])
            pot["confidences"] = result.get("confidences", [])
        else:
            pot["boxes"] = []
            pot["confidences"] = []
        return pot

    with ThreadPoolExecutor(max_workers=32) as executor:
        pots = list(
            tqdm(
                executor.map(detect_task, pots),
                total=len(pots),
                desc=f"E{experiment}/Z{zone} Detecting",
                leave=False,
            )
        )

    logger.info(
        f"E{experiment}/zone{zone}: Detected plants in {time.time() - t0_detect:.2f}s"
    )

    # --- BLOCK 4: SEGMENT ALL PLANTS ---
    logger.info(f"E{experiment}/zone{zone}: Starting SEGMENT block")
    t0_segment = time.time()

    def segment_task(pot):
        if pot["boxes"]:
            result = segment_plant(
                pot["warped_b64"],
                pot["boxes"],
                pot["confidences"],
                zone_key=zone_key,
                plant_id=pot["plant_id"],
                timestamp_str=pot["timestamp_str"],
            )
            if result and result.get("success"):
                pot["mask_b64"] = result.get("mask")
            else:
                pot["mask_b64"] = None
        else:
            pot["mask_b64"] = None
        return pot

    with ThreadPoolExecutor(max_workers=8) as executor:
        pots = list(
            tqdm(
                executor.map(segment_task, pots),
                total=len(pots),
                desc=f"E{experiment}/Z{zone} Segmenting",
                leave=False,
            )
        )

    logger.info(
        f"E{experiment}/zone{zone}: Segmented plants in {time.time() - t0_segment:.2f}s"
    )

    # --- BLOCK 5: STATS ALL PLANTS ---
    logger.info(f"E{experiment}/zone{zone}: Starting STATS block")
    t0_stats = time.time()

    def stats_task(pot):
        if pot["mask_b64"]:
            result = compute_plant_stats(
                pot["warped_b64"],
                pot["mask_b64"],
                zone_key=zone_key,
                plant_id=pot["plant_id"],
                timestamp_str=pot["timestamp_str"],
            )
            if result:
                pot["stats"] = result.get("stats")
            else:
                pot["stats"] = None
        else:
            pot["stats"] = None
        return pot

    with ThreadPoolExecutor(max_workers=256) as executor:
        pots = list(
            tqdm(
                executor.map(stats_task, pots),
                total=len(pots),
                desc=f"E{experiment}/Z{zone} Stats",
                leave=False,
            )
        )

    logger.info(
        f"E{experiment}/zone{zone}: Computed stats in {time.time() - t0_stats:.2f}s"
    )

    # --- BLOCK 6: SAVE IMAGES ---
    logger.info(f"E{experiment}/zone{zone}: Starting SAVE block")
    t0_save = time.time()

    def save_task(pot):
        try:
            warped_image = decode_image(pot["warped_b64"])
            image_filename = f"{pot['timestamp_str']}_plant{pot['plant_id']:02d}.jpg"
            image_file_path = processed_dir / image_filename
            warped_image.save(image_file_path)
            pot["image_path"] = str(image_file_path.relative_to(OFFLINE_DIR))
        except Exception as e:
            logger.warning(f"Failed to save image: {e}")
            pot["image_path"] = None
        return pot

    with ThreadPoolExecutor(max_workers=256) as executor:
        pots = list(
            tqdm(
                executor.map(save_task, pots),
                total=len(pots),
                desc=f"E{experiment}/Z{zone} Saving",
                leave=False,
            )
        )

    logger.info(
        f"E{experiment}/zone{zone}: Saved images in {time.time() - t0_save:.2f}s"
    )

    # --- FINALIZE RESULTS ---
    for pot in pots:
        result_row = {
            "experiment": pot["experiment"],
            "zone": pot["zone"],
            "time": pot["time"],
            "plant_id": pot["plant_id"],
            "embedding": pot["embedding"],
            "image_path": pot["image_path"],
        }
        if pot.get("stats"):
            result_row.update(pot["stats"])
        results.append(result_row)

    return results


def transform_image_embeddings(df: pl.DataFrame) -> pl.DataFrame:
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

    Returns:
        DataFrame with added plant_id, embedding, and image_path columns.
        The returned DataFrame will have more rows than the input (one per detected plant).
    """
    logger.info(f"Starting image embedding transform on {len(df)} rows")

    # Get unique images (experiment, zone, time combinations)
    unique_images = (
        df.select(["experiment", "zone", "time", "image_name"])
        .unique()
        .sort("experiment", "zone", "time")
    )

    # Group images by (experiment, zone) and find corresponding image paths
    zone_groups = {}
    for row in unique_images.iter_rows(named=True):
        experiment = row["experiment"]
        zone = row["zone"]
        timestamp = row["time"]
        image_name = row["image_name"]

        # Find the image file
        image_path = find_image_path(experiment, zone, image_name)
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

    # Store results for each detected plant
    all_results = []

    for zone_key, images in tqdm(zone_groups.items()):
        result = process_zone_images(zone_key, images)
        all_results.extend(result)

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
        status = "✓" if num_pots == 18 or num_pots == 64 else "⚠"
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
