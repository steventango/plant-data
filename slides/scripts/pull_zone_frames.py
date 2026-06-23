"""
Pull day-14 (9:30 AM observation) raw and viz frames for all 12 E17 zones.

Authoritative source for each frame: the `image_name` column of each zone's
processed parquet. Don't guess from filenames or timezone offsets.

Outputs to slides/assets/photos/zones/zone{NN}_{raw,viz}.jpg
"""

import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import polars as pl

E17_ROOT = Path("/data/plant-rl/online/E17/P1")
TARGET_DATE = date(2026, 4, 14)
OUT = Path(__file__).resolve().parent.parent / "assets" / "photos" / "zones"
RESIZE = "1600x"


def find_zone_dirs() -> dict[int, Path]:
    """Return {zone_number: zone_dir} for every E17 zone."""
    zones: dict[int, Path] = {}
    for zd in E17_ROOT.glob("*/alliance-zone*"):
        zone = int(zd.name.split("zone")[-1])
        zones[zone] = zd
    return dict(sorted(zones.items()))


def lookup_raw_name(parquet: Path, day: date) -> str | None:
    """Return the raw image filename used for the 9:30 observation on `day`."""
    df = pl.read_parquet(parquet, columns=["time", "plant_id", "image_name"])
    rows = (
        df.filter(pl.col("time").dt.date() == pl.lit(day))
        .unique(subset=["time"])
        .select("image_name")
    )
    if rows.is_empty():
        return None
    return rows["image_name"][0]


def viz_path_for(zone_dir: Path, zone: int, day: date) -> Path:
    return (
        zone_dir
        / "processed"
        / "v27"
        / "visualizations"
        / f"E17_Z{zone}_{day:%Y-%m-%d}T093000_viz.jpg"
    )


def convert_resize(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["convert", str(src), "-resize", RESIZE, "-quality", "92", str(dst)],
        check=True,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    zones = find_zone_dirs()
    print(f"Found {len(zones)} zones in {E17_ROOT}")

    misses: list[str] = []
    for zone, zone_dir in zones.items():
        zz = f"{zone:02d}"
        parquet = zone_dir / "processed" / "v27" / f"E17_Z{zone}.parquet"
        if not parquet.exists():
            misses.append(f"Z{zone}: no parquet at {parquet}")
            continue

        raw_name = lookup_raw_name(parquet, TARGET_DATE)
        if not raw_name:
            misses.append(f"Z{zone}: no row in parquet for {TARGET_DATE}")
            continue

        raw_src = zone_dir / "images" / raw_name
        viz_src = viz_path_for(zone_dir, zone, TARGET_DATE)

        if not raw_src.exists():
            misses.append(f"Z{zone}: raw missing: {raw_src}")
            continue
        if not viz_src.exists():
            misses.append(f"Z{zone}: viz missing: {viz_src}")

        raw_dst = OUT / f"zone{zz}_raw.jpg"
        viz_dst = OUT / f"zone{zz}_viz.jpg"
        convert_resize(raw_src, raw_dst)
        if viz_src.exists():
            convert_resize(viz_src, viz_dst)
        print(f"Z{zone}: raw={raw_src.name}  ->  {raw_dst.name}, {viz_dst.name}")

    if misses:
        print("\nMisses:")
        for m in misses:
            print(" ", m)
        sys.exit(1 if any("no parquet" in m or "no row" in m or "raw missing" in m for m in misses) else 0)


if __name__ == "__main__":
    main()
