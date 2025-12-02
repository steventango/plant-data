import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import polars as pl
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path("/data")
OFFLINE_DIR = DATA_DIR / "offline"
STAGING_DIR = Path("/data/tmp/tmp/labeled/labeling")
VERSION = "v13"
PARQUET_FILENAME = (
    f"cleaned_offline_dataset_daily_continuous_{VERSION}_with_predictions.parquet"
)
PARQUET_PATH = OFFLINE_DIR / PARQUET_FILENAME


def generate_uuid(
    experiment: int, zone: int, plant_id: int, timestamp: datetime
) -> str:
    """Generate a unique ID for a pot image.
    Format: {experiment}_{zone}_{plant_id}_{isoformat_time}_{uuid length 6}
    """
    # Use ISO format but strip timezone and microseconds for cleaner filename
    time_str = timestamp.strftime("%Y%m%dT%H%M%S")
    suffix = str(uuid.uuid4())[:6]
    return f"{experiment}_{zone}_{plant_id}_{time_str}_{suffix}"


def parse_filename(filename: str) -> Optional[dict]:
    """Parse the filename to extract metadata.
    Format: {experiment}_{zone}_{plant_id}_{isoformat_time}_{uuid length 6}.jpg
    """
    try:
        # Remove extension
        name = Path(filename).stem
        parts = name.split("_")
        if len(parts) < 5:
            return None

        experiment = int(parts[0])
        zone = int(parts[1])
        plant_id = int(parts[2])
        time_str = parts[3]
        # suffix = parts[4]

        # Parse ISO format timestamp
        # Format: 20250822T093000 (from isoformat without separators)
        timestamp = datetime.strptime(time_str, "%Y%m%dT%H%M%S").replace(
            tzinfo=ZoneInfo("America/Edmonton")
        )

        return {
            "experiment": experiment,
            "zone": zone,
            "plant_id": plant_id,
            "time": timestamp,
        }
    except Exception as e:
        logger.error(f"Failed to parse filename {filename}: {e}")
        return None


def export_images(label_columns=None):
    """Export pot images to staging directory for labeling.

    Args:
        label_columns: Optional list of column names to use for iterative labeling.
                      If None, auto-detects sparse columns.
    """
    if STAGING_DIR.exists():
        logger.warning(f"Staging directory {STAGING_DIR} already exists.")
        response = input("Delete and start over? (y/n): ")
        if response.lower() == "y":
            shutil.rmtree(STAGING_DIR)
        else:
            logger.info("Resuming/Aborting export.")
            return

    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset from {PARQUET_PATH}")
    if not PARQUET_PATH.exists():
        logger.error(f"Dataset not found: {PARQUET_PATH}")
        return

    df = pl.read_parquet(PARQUET_PATH)

    # Filter rows that have image_path
    df_with_images = df.filter(pl.col("image_path").is_not_null())
    logger.info(f"Found {len(df_with_images)} rows with pot images")

    # Detect label columns (columns that might contain user-added labels)
    logger.info(f"Using specified label columns: {label_columns}")
    label_stats = {}
    for col in label_columns:
        if col not in df_with_images.columns:
            logger.warning(f"Column '{col}' not found in dataset")
            continue
        non_null_count = df_with_images.filter(pl.col(col).is_not_null()).height
        if non_null_count > 0:
            label_stats[col] = non_null_count

    if label_stats:
        logger.info(f"Found existing labels: {label_stats}")
        logger.info("Images with labels will be organized into folders")

    # Copy images with UUID filenames
    for row in tqdm(
        df_with_images.iter_rows(named=True),
        total=len(df_with_images),
        desc="Copying images",
    ):
        experiment = row["experiment"]
        zone = row["zone"]
        plant_id = row["plant_id"]
        timestamp = row["time"]
        image_path = row["image_path"]

        # Generate UUID
        img_uuid = generate_uuid(experiment, zone, plant_id, timestamp)
        # Source path
        source_path = OFFLINE_DIR / image_path

        # Determine destination folder based on existing labels
        dest_folder = STAGING_DIR

        # Check for disagreement between bolted and bolted_pred
        val_bolted = row.get("bolted")
        val_pred = row.get("bolted_pred")

        if val_bolted is not None and val_pred is not None and val_bolted != val_pred:
            # If predictions disagree with labels, put in root for review
            filename = f"{img_uuid}_DISAGREE_pred={val_pred}_label={val_bolted}.jpg"
        else:
            filename = f"{img_uuid}.jpg"
            for col in label_stats.keys():
                value = row.get(col)
                if value is not None:
                    # Create folder with label format: column=value
                    folder_name = f"{col}={value}"
                    dest_folder = STAGING_DIR / folder_name
                    dest_folder.mkdir(exist_ok=True)
                    break  # Use first non-null label

        dest_path = dest_folder / filename

        try:
            if not source_path.exists():
                logger.warning(f"Source image not found: {source_path}")
                continue

            shutil.copy2(source_path, dest_path)
        except Exception as e:
            logger.error(f"Failed to copy {source_path} to {dest_path}: {e}")

    logger.info(f"Export completed. Images are in {STAGING_DIR}")
    if label_stats:
        logger.info("Pre-labeled images have been organized into folders")
        logger.info("Unlabeled images are in the root directory")
    logger.info("Please organize images into folders (e.g., 'bolted=1', 'bolted=0').")


def import_labels():
    """Import labels from organized folders."""
    if not STAGING_DIR.exists():
        logger.error(f"Staging directory {STAGING_DIR} does not exist.")
        return

    logger.info("Scanning for labels...")

    labels = []

    # Walk through subdirectories
    for folder in STAGING_DIR.iterdir():
        if not folder.is_dir():
            continue

        label_name = folder.name
        logger.info(f"Processing folder: {label_name}")

        for image_file in folder.glob("*.jpg"):
            metadata = parse_filename(image_file.name)
            if not metadata:
                continue

            # Parse label
            if "=" in label_name:
                key, value = label_name.split("=", 1)
                # Try to convert value to int/float if possible
                try:
                    if "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # Keep as string

                metadata[key] = value
            else:
                metadata["label"] = label_name

            labels.append(metadata)

    if not labels:
        logger.warning("No labeled images found.")
        return

    logger.info(f"Found {len(labels)} labeled images.")

    # Create dataframe from labels
    df_labels = pl.DataFrame(labels)

    # Debug: Show what columns we have in df_labels
    logger.info(f"df_labels columns: {df_labels.columns}")
    logger.info(f"df_labels shape: {df_labels.shape}")
    logger.info(f"df_labels schema: {df_labels.schema}")
    logger.info(f"Sample of df_labels:\n{df_labels.head()}")

    # Load original dataset
    logger.info(f"Loading original dataset from {PARQUET_PATH}")
    df_orig = pl.read_parquet(PARQUET_PATH)
    logger.info(f"df_orig shape: {df_orig.shape}")

    # Join keys
    join_keys = ["experiment", "zone", "time", "plant_id"]

    # Check if keys exist
    for key in join_keys:
        if key not in df_orig.columns:
            logger.error(
                f"Key {key} not found in original dataset columns: {df_orig.columns}"
            )
            return
        if key not in df_labels.columns:
            logger.error(
                f"Key {key} not found in df_labels columns: {df_labels.columns}"
            )
            return

    # Debug: Check data types match
    logger.info("Checking data types for join keys:")
    for key in join_keys:
        logger.info(
            f"  {key}: df_orig={df_orig[key].dtype}, df_labels={df_labels[key].dtype}"
        )

    # Perform left join to add labels
    logger.info(f"Performing join with keys: {join_keys}")
    df_labeled = df_orig.join(df_labels, on=join_keys, how="left", suffix="_new")

    # Debug: Show what happened after join
    logger.info(f"df_labeled shape: {df_labeled.shape}")
    logger.info(f"df_labeled columns: {df_labeled.columns}")

    # Count how many rows got new labels
    label_cols = [col for col in df_labels.columns if col not in join_keys]
    logger.info(f"Label columns to merge: {label_cols}")

    # Merge the new label columns with existing ones
    for col in label_cols:
        new_col_name = f"{col}_new"
        if new_col_name in df_labeled.columns:
            non_null_count = df_labeled.filter(
                pl.col(new_col_name).is_not_null()
            ).height
            logger.info(f"  {col}: {non_null_count} rows with new labels")

            # Merge: use new labels if available, otherwise keep old labels
            if col in df_labeled.columns:
                logger.info(f"  Merging {new_col_name} into existing {col} column")
                df_labeled = df_labeled.with_columns(
                    pl.coalesce([pl.col(new_col_name), pl.col(col)]).alias(col)
                ).drop(new_col_name)
            else:
                logger.info(f"  Renaming {new_col_name} to {col}")
                df_labeled = df_labeled.rename({new_col_name: col})

    # Save
    output_path = OFFLINE_DIR / "labeled_dataset.parquet"
    logger.info(f"Saving labeled dataset to {output_path}")
    df_labeled.write_parquet(output_path)
    logger.info("Done!")

    # Summary
    logger.info("\nSummary of changes:")
    for col in label_cols:
        if col in df_labeled.columns:
            total_labeled = df_labeled.filter(pl.col(col).is_not_null()).height
            logger.info(f"  {col}: {total_labeled} total labeled rows")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Label parquet datasets by organizing pot images into folders"
    )
    parser.add_argument(
        "action",
        choices=["export", "import"],
        help="Action to perform: export images or import labels",
    )
    parser.add_argument(
        "label",
        nargs="+",
        help="Label columns to use for iterative labeling (export only). If not specified, auto-detects sparse columns.",
    )

    args = parser.parse_args()

    if args.action == "export":
        export_images(label_columns=args.label)
    elif args.action == "import":
        import_labels()


if __name__ == "__main__":
    main()
