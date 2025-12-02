import argparse
import logging
import sys
from pathlib import Path

import polars as pl
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a 16:9 collage of plant images by randomly sampling from the dataset."
    )
    parser.add_argument(
        "--parquet",
        "-p",
        default="/data/offline/cleaned_offline_dataset_continuous_v16.parquet",
        help="Path to parquet file",
    )
    parser.add_argument(
        "--out",
        "-o",
        default="results/plant_collage.jpg",
        help="Output filename for the collage",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=9,
        help="Number of rows in the grid (default: 9)",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=16,
        help="Number of columns in the grid (default: 16)",
    )
    parser.add_argument(
        "--thumb-width",
        type=int,
        default=200,
        help="Width of each thumbnail image (default: 200)",
    )
    parser.add_argument(
        "--thumb-height",
        type=int,
        default=200,
        help="Height of each thumbnail image (default: 200)",
    )

    args = parser.parse_args()

    # 1. Read Parquet
    try:
        logging.info(f"Reading parquet: {args.parquet}")
        df = pl.read_parquet(args.parquet)
    except Exception as e:
        logging.error(f"Failed to read parquet: {e}")
        sys.exit(1)

    # 2. Filter for valid image paths
    if "image_path" not in df.columns:
        logging.error("'image_path' column not found in dataset.")
        sys.exit(1)

    logging.info("Filtering for valid image paths...")
    df = df.filter(pl.col("image_path").is_not_null())

    # Optional: Filter for specific time if desired, but user asked for random sampling.
    # We might want to ensure the file actually exists before selecting, but that's expensive for all rows.
    # We will sample more than needed and discard missing ones.

    total_slots = args.rows * args.cols
    logging.info(f"Need {total_slots} images for a {args.cols}x{args.rows} grid.")

    # 3. Randomly sample
    # We sample 2x the needed amount to account for missing files
    sample_size = min(total_slots * 2, df.shape[0])
    logging.info(f"Sampling {sample_size} candidates from {df.shape[0]} rows...")

    # Polars sample
    sampled_df = df.sample(n=sample_size, shuffle=True)

    valid_images = []

    # 4. Collect valid images
    logging.info("Verifying image files...")
    for row in sampled_df.iter_rows(named=True):
        if len(valid_images) >= total_slots:
            break

        rel_path = row["image_path"]
        # Check paths
        full_path = Path("/data/offline") / rel_path
        if not full_path.exists():
            full_path = Path("/data") / rel_path
            if not full_path.exists():
                continue

        valid_images.append(full_path)

    if len(valid_images) < total_slots:
        logging.warning(
            f"Only found {len(valid_images)} valid images out of {total_slots} required. The grid will be incomplete."
        )

    # 5. Create Collage
    logging.info("Creating collage...")

    # Calculate canvas size
    canvas_width = args.cols * args.thumb_width
    canvas_height = args.rows * args.thumb_height

    collage = Image.new("RGB", (canvas_width, canvas_height), color=(0, 0, 0))

    for idx, img_path in enumerate(valid_images):
        if idx >= total_slots:
            break

        row_idx = idx // args.cols
        col_idx = idx % args.cols

        x = col_idx * args.thumb_width
        y = row_idx * args.thumb_height

        try:
            with Image.open(img_path) as img:
                # Resize/Crop to fit thumbnail size
                # We use Image.Resampling.LANCZOS if available, else Image.ANTIALIAS (deprecated in newer Pillow)
                resample_method = getattr(Image, "Resampling", Image).LANCZOS

                # Resize maintaining aspect ratio to cover the thumbnail area, then crop
                img_ratio = img.width / img.height
                thumb_ratio = args.thumb_width / args.thumb_height

                if img_ratio > thumb_ratio:
                    # Image is wider than thumb
                    new_height = args.thumb_height
                    new_width = int(new_height * img_ratio)
                else:
                    # Image is taller than thumb
                    new_width = args.thumb_width
                    new_height = int(new_width / img_ratio)

                img = img.resize((new_width, new_height), resample=resample_method)

                # Center crop
                left = (new_width - args.thumb_width) / 2
                top = (new_height - args.thumb_height) / 2
                right = (new_width + args.thumb_width) / 2
                bottom = (new_height + args.thumb_height) / 2

                img = img.crop((left, top, right, bottom))

                collage.paste(img, (x, y))
        except Exception as e:
            logging.warning(f"Failed to process image {img_path}: {e}")

    # 6. Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    collage.save(out_path, quality=90)
    logging.info(f"Collage saved to {out_path}")


if __name__ == "__main__":
    main()
