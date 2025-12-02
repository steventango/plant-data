import polars as pl
import shutil
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        "-p",
        default="/data/plant-rl/offline/cleaned_offline_dataset_continuous_v16.parquet",
        help="Path to parquet file",
    )
    parser.add_argument(
        "--target-area",
        "-t",
        default=400,
        type=float,
        help="Target area for filtering",
    )
    parser.add_argument(
        "--out",
        "-o",
        default="results/retrieved_images",
        help="Output directory for retrieved images",
    )
    args = parser.parse_args()
    parquet_file = args.parquet
    output_dir = Path(args.out)
    output_dir.mkdir(exist_ok=True)

    print(f"Reading {parquet_file}...")
    try:
        df = pl.read_parquet(parquet_file)
    except Exception as e:
        print(f"Failed to read parquet: {e}")
        return

    # Prefilter to 9:30 AM rows
    df = df.filter(pl.col("time").dt.hour() == 9)
    df = df.filter(pl.col("time").dt.minute() == 30)

    # Determine area column
    if "area" in df.columns:
        area_col = "area"
    elif "clean_area" in df.columns:
        area_col = "clean_area"
    else:
        print("No area column found")
        return

    print(f"Using area column: {area_col}")

    # Filter for valid area and image_path
    df = df.filter(pl.col(area_col).is_not_null() & pl.col("image_path").is_not_null())

    # Calculate distance to target area
    target_area = args.target_area
    df = df.with_columns((pl.col(area_col) - target_area).abs().alias("area_diff"))

    # Sort by difference and take top 30
    top_30 = df.sort("area_diff").head(30)

    print(f"Found {len(top_30)} images closest to area {target_area}")

    # Copy images
    count = 0
    for row in top_30.iter_rows(named=True):
        rel_path = row["image_path"]
        full_path = Path("/data/offline") / rel_path

        dest_name = f"area_{row[area_col]:.2f}_{Path(rel_path).name}"
        dest_path = output_dir / dest_name

        try:
            shutil.copy2(full_path, dest_path)
            print(f"Copied {full_path} to {dest_path}")
            count += 1
        except Exception as e:
            print(f"Failed to copy {full_path}: {e}")

    print(f"Successfully retrieved {count} images.")


if __name__ == "__main__":
    main()
