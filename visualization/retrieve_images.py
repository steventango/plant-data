import polars as pl
import shutil
from pathlib import Path
import argparse
import datetime

from visualization.common import default_parquet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        "-p",
        default=default_parquet(),
        help="Path to parquet file",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Retrieve raw images instead of processed ones",
    )
    parser.add_argument(
        "--experiment",
        "-e",
        type=int,
        help="Experiment number to filter for",
    )
    parser.add_argument(
        "--zone",
        "-z",
        type=int,
        help="Zone number to filter for",
    )
    parser.add_argument(
        "--out",
        "-o",
        default="results/retrieved_images",
        help="Output directory for retrieved images",
    )
    parser.add_argument(
        "--target-area",
        type=float,
        default=None,
        help="If set, include area value in output filename",
    )
    args = parser.parse_args()
    parquet_file = args.parquet
    output_dir = Path(args.out)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Reading {parquet_file}...")
    try:
        df = pl.read_parquet(parquet_file)
    except Exception as e:
        print(f"Failed to read parquet: {e}")
        return

    # Filter by experiment and zone if provided
    if args.experiment is not None:
        df = df.filter(pl.col("experiment") == args.experiment)
    if args.zone is not None:
        df = df.filter(pl.col("zone") == args.zone)

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

    # Filter for valid image_path
    df = df.filter(pl.col("image_path").is_not_null())

    results = df.unique(subset=["image_path"]).sort("time")
    print(f"Found {len(results)} unique images matching filters")

    # Copy images
    count = 0
    seen_paths = set()
    for row in results.iter_rows(named=True):
        rel_path = row["image_path"]
        # In the parquet, image_path is absolute or relative
        full_path = Path(rel_path)
        if not full_path.is_absolute():
            full_path = Path("/data/offline") / rel_path

        if args.raw:
            # The raw images are named using UTC time: YYYY-MM-DDTHHMMSS+0000_left.jpg
            # Convert the local time to UTC for the lookup
            utc_time = row["time"].astimezone(datetime.timezone.utc)
            timestamp_utc = utc_time.strftime("%Y-%m-%dT%H%M%S")

            if "/processed/" in str(full_path):
                zone_root = Path(str(full_path).split("/processed/")[0])
                raw_dir = zone_root / "images"

                # Search for the raw image with timestamp +0000_left.jpg
                potential_raw = raw_dir / f"{timestamp_utc}+0000_left.jpg"
                if potential_raw.exists():
                    full_path = potential_raw
                else:
                    # Try a glob if the exact name doesn't match
                    matches = list(raw_dir.glob(f"{timestamp_utc}*"))
                    if matches:
                        full_path = matches[0]
                    else:
                        print(
                            f"Could not find raw image for {timestamp_utc} (local {row['time']}) in {raw_dir}"
                        )
                        continue

        # Use time and zone in name if possible
        timestamp = row["time"].strftime("%Y%m%d_%H%M%S")
        prefix = f"E{row['experiment']}_Z{row['zone']}_"
        if args.target_area is not None:
            dest_name = f"{prefix}area_{row[area_col]:.2f}_{timestamp}_{full_path.name}"
        else:
            dest_name = f"{prefix}{timestamp}_{full_path.name}"

        dest_path = output_dir / dest_name

        if dest_path in seen_paths:
            continue
        seen_paths.add(dest_path)

        try:
            shutil.copy2(full_path, dest_path)
            print(f"Copied {full_path} to {dest_path}")
            count += 1
        except Exception as e:
            print(f"Failed to copy {full_path}: {e}")

    print(f"Successfully retrieved {count} images.")


if __name__ == "__main__":
    main()
