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


def detect_pots_reference(
    image_path: Path, zone_key: tuple, output_dir: Optional[Path] = None
) -> Optional[dict]:
    """Detect pots in a reference image to get quadrilaterals.

    Args:
        image_path: Path to the reference image
        zone_key: Tuple of (experiment, zone) for saving visualization
        output_dir: Optional output directory for visualizations

    Returns:
        Dictionary with detection results (boxes, quadrilaterals, etc.), or None if failed
    """
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        image_data = encode_image(image)

        # Determine visualization directory
        viz_dir = (
            (output_dir / "visualizations")
            if output_dir
            else (OFFLINE_DIR / "visualizations")
        )

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
                viz_dir.mkdir(parents=True, exist_ok=True)
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
                viz_dir.mkdir(parents=True, exist_ok=True)
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
                viz_dir.mkdir(parents=True, exist_ok=True)
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
                "embedding_types": ["cls_token", "patch_features"],
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()

        return result["cls_token"], result["patch_features"]
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return None


def detect_plant(
    warped_image_b64: str,
) -> Optional[dict]:
    """Detect plant in a warped pot image."""
    try:
        response = get_session().post(
            f"{PIPELINE_URL}/plant/detect",
            json={"image_data": warped_image_b64},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Plant detection failed: {e}")
        return None


def segment_plant(
    warped_image_b64: str,
    boxes: list,
    confidences: list,
) -> Optional[dict]:
    """Segment plant given detection boxes."""
    try:
        response = get_session().post(
            f"{PIPELINE_URL}/plant/segment",
            json={
                "image_data": warped_image_b64,
                "boxes": boxes,
                "confidences": confidences,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Plant segmentation failed: {e}")
        return None


def compute_plant_stats(
    warped_image_b64: str,
    mask_b64: str,
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
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Plant stats failed: {e}")
        return None


def visualize_plant_pipeline(
    warped_image_b64: str,
    boxes: list,
    confidences: list,
    masks: list,
    zone_key: tuple,
    plant_id: int,
    timestamp_str: str,
    mask_scores: Optional[list] = None,
    combined_scores: Optional[list] = None,
    selected_index: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """Visualize the plant pipeline results."""
    try:
        response = get_session().post(
            f"{PIPELINE_URL}/plant/visualize",
            json={
                "image_data": warped_image_b64,
                "boxes": boxes,
                "confidences": confidences,
                "masks": masks,
                "mask_scores": mask_scores,
                "combined_scores": combined_scores,
                "selected_index": selected_index,
                "stats": True,
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()

        if "visualization" in result:
            try:
                # Save to images/{plant_id}/{timestamp}_viz.jpg
                if output_dir:
                    plant_dir = output_dir / "images" / str(plant_id)
                else:
                    experiment, zone = zone_key
                    plant_dir = (
                        OFFLINE_DIR
                        / "processed"
                        / f"E{experiment}"
                        / f"Z{zone:02d}"
                        / "images"
                        / str(plant_id)
                    )

                plant_dir.mkdir(parents=True, exist_ok=True)
                viz_image = decode_image(result["visualization"])
                viz_path = plant_dir / f"{timestamp_str}_viz.jpg"
                viz_image.save(viz_path)
                logger.debug(f"Saved plant visualization to {viz_path}")
            except Exception as e:
                logger.warning(f"Failed to save plant visualization: {e}")

    except Exception as e:
        logger.error(f"Plant visualization failed: {e}")


def process_zone_images(
    zone_df_subset: pl.DataFrame, output_dir: Optional[Path] = None
) -> pl.DataFrame:
    """Process all images for a single zone using block-based execution.

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
    zone_key = (experiment, zone)

    # Load cache first
    cache = DiskCache(OFFLINE_DIR, VISION_VERSION)
    cache_df = cache.load(experiment, zone)

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

    images_info_df = pl.DataFrame(images_info)

    # Filter out cached images
    if cache_df is not None and not cache_df.is_empty():
        # Anti-join on time
        todo_df = images_info_df.join(cache_df.select("time"), on="time", how="anti")
        images_to_process = todo_df.to_dicts()
        logger.info(
            f"E{experiment}/zone{zone}: {len(cache_df)} images cached, {len(images_to_process)} to process"
        )
    else:
        images_to_process = images_info
        # Define empty cache df with expected schema for union later
        # We'll just rely on new_results_df schema if cache is empty
        logger.info(
            f"E{experiment}/zone{zone}: No cache found, processing {len(images_to_process)} images"
        )

    # If nothing to process, return cache
    if not images_to_process:
        if cache_df is not None:
            return cache_df
        else:
            return pl.DataFrame()

    # Find reference image (9:30 AM)
    reference_image = None
    for img_info in images_info:  # Search in ALL images
        timestamp = img_info["time"]
        if timestamp.hour == 9 and timestamp.minute == 30:
            reference_image = img_info
            break

    if reference_image is None:
        reference_image = images_info[0]
        logger.warning(
            f"E{experiment}/zone{zone}: No 9:30 AM image found, using first image"
        )
    else:
        logger.info(
            f"E{experiment}/zone{zone}: Using 9:30 AM reference image at {reference_image['image_path']}"
        )

    # Detect pots in reference image
    t0 = time.time()
    detection_result = detect_pots_reference(
        reference_image["image_path"], zone_key, output_dir
    )
    logger.info(
        f"E{experiment}/zone{zone}: Reference detection took {time.time() - t0:.2f}s"
    )

    pots = []
    if detection_result and detection_result.get("quadrilaterals"):
        quadrilaterals = detection_result["quadrilaterals"]

        if output_dir:
            images_base_dir = output_dir / "images"
        else:
            images_base_dir = (
                OFFLINE_DIR / "processed" / f"E{experiment}" / f"Z{zone:02d}" / "images"
            )
        images_base_dir.mkdir(parents=True, exist_ok=True)

        # --- BLOCK 1: WARP ---
        logger.info(f"E{experiment}/zone{zone}: Starting WARP block")
        t0_warp = time.time()

        def warp_task(img_info):
            timestamp = img_info["time"]
            result_list = []
            warped_images = warp_with_quadrilaterals(
                img_info["image_path"], quadrilaterals
            )
            if warped_images:
                timestamp_str = timestamp.strftime("%Y-%m-%dT%H%M%S")
                for i, b64 in enumerate(warped_images):
                    result_list.append(
                        {
                            "experiment": experiment,
                            "zone": zone,
                            "time": timestamp,
                            "plant_id": i,
                            "warped_b64": b64,
                            "timestamp_str": timestamp_str,
                        }
                    )
            return result_list

        with ThreadPoolExecutor(max_workers=256) as executor:
            futures = [executor.submit(warp_task, img) for img in images_to_process]
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

        if pots:
            # --- BLOCK 2: EMBED ---
            logger.info(f"E{experiment}/zone{zone}: Starting EMBED block")
            t0_embed = time.time()

            def embed_task(pot):
                pot["cls_token"], pot["patch_features"] = generate_embedding(
                    pot["warped_b64"]
                )
                return pot

            with ThreadPoolExecutor(max_workers=8) as executor:
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

            # --- BLOCK 3: DETECT ---
            logger.info(f"E{experiment}/zone{zone}: Starting DETECT block")
            t0_detect = time.time()

            def detect_task(pot):
                res = detect_plant(pot["warped_b64"])
                pot["boxes"] = res.get("boxes", []) if res else []
                pot["confidences"] = res.get("confidences", []) if res else []
                return pot

            with ThreadPoolExecutor(max_workers=16) as executor:
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

            # --- BLOCK 4: SEGMENT ---
            logger.info(f"E{experiment}/zone{zone}: Starting SEGMENT block")
            t0_segment = time.time()

            def segment_task(pot):
                res = (
                    segment_plant(pot["warped_b64"], pot["boxes"], pot["confidences"])
                    if pot["boxes"]
                    else None
                )
                success = res and res.get("success")
                pot["mask_b64"] = res.get("mask") if success else None
                pot["all_masks"] = res.get("masks", []) if success else []
                # Keep other scores if needed...
                pot["mask_scores"] = res.get("mask_scores", []) if success else []
                pot["combined_scores"] = (
                    res.get("combined_scores", []) if success else []
                )
                pot["selected_index"] = res.get("selected_index") if success else None
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

            # --- BLOCK 5: STATS ---
            logger.info(f"E{experiment}/zone{zone}: Starting STATS block")
            t0_stats = time.time()

            def stats_task(pot):
                pot["stats"] = None
                if pot["mask_b64"]:
                    res = compute_plant_stats(pot["warped_b64"], pot["mask_b64"])
                    if res:
                        pot["stats"] = res.get("stats")
                        visualize_plant_pipeline(
                            pot["warped_b64"],
                            pot["boxes"],
                            pot["confidences"],
                            pot.get("all_masks") or [pot["mask_b64"]],
                            zone_key,
                            pot["plant_id"],
                            pot["timestamp_str"],
                            mask_scores=pot.get("mask_scores"),
                            combined_scores=pot.get("combined_scores"),
                            selected_index=pot.get("selected_index"),
                            output_dir=output_dir,
                        )
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

            # --- BLOCK 6: SAVE ---
            logger.info(f"E{experiment}/zone{zone}: Starting SAVE block")
            t0_save = time.time()

            def save_task(pot):
                try:
                    plant_dir = images_base_dir / str(pot["plant_id"])
                    plant_dir.mkdir(parents=True, exist_ok=True)
                    warped = decode_image(pot["warped_b64"])
                    fname = f"{pot['timestamp_str']}.jpg"
                    fpath = plant_dir / fname
                    warped.save(fpath)
                    pot["image_path"] = str(
                        fpath.relative_to(output_dir if output_dir else OFFLINE_DIR)
                    )
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

    # Combine new results
    new_results_df = pl.DataFrame()
    if pots:
        rows = []
        for pot in pots:
            r = {
                "experiment": pot["experiment"],
                "zone": pot["zone"],
                "time": pot["time"],
                "plant_id": pot["plant_id"],
                "cls_token": pot["cls_token"],
                "patch_features": pot["patch_features"],
                "image_path": pot["image_path"],
            }
            if pot.get("stats"):
                r.update(pot["stats"])
            rows.append(r)
        new_results_df = pl.DataFrame(rows)

    # Merge with cache
    if cache_df is not None and not cache_df.is_empty():
        if not new_results_df.is_empty():
            combined_df = pl.concat([cache_df, new_results_df], how="diagonal")
        else:
            combined_df = cache_df
    else:
        combined_df = new_results_df

    if not combined_df.is_empty():
        combined_df = combined_df.sort("time", "plant_id")
        # Save updated cache (only if we added something new)
        if not new_results_df.is_empty():
            cache.save(experiment, zone, combined_df)

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
