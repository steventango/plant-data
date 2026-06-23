import re
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from visualization.common import RESULTS_DIR

GENERATE_SCRIPT = Path("generate.sh")
OUTPUT_DIR = RESULTS_DIR / "reference_images"


def get_zone_paths():
    paths = []
    with open(GENERATE_SCRIPT, "r") as f:
        for line in f:
            if line.strip().startswith("uv run python process_zone.py"):
                # Extract path after --data-path
                match = re.search(r"--data-path\s+(\S+)", line)
                if match:
                    paths.append(Path(match.group(1)))
    return paths


def main():
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir()

    print(f"Reading zone paths from {GENERATE_SCRIPT}...")
    zone_paths = get_zone_paths()
    print(f"Found {len(zone_paths)} zone paths.")

    # Edmonton Timezone
    edmonton_tz = ZoneInfo("America/Edmonton")

    for zone_path in zone_paths:
        try:
            # Extract Exp and Zone IDs from path for naming
            parts = zone_path.parts
            exp_part = next(
                (p for p in parts if p.startswith("E") and p[1:].isdigit()), "E??"
            )

            # Extract zone number (e.g. "01" from "alliance-zone01")
            zone_dirname = zone_path.name
            match_zone = re.search(r"zone(\d+)", zone_dirname)
            if match_zone:
                zone_num = match_zone.group(1).zfill(2)
                zone_prefix = f"Z{zone_num}"
            else:
                zone_prefix = zone_dirname

            images_dir = zone_path / "images"
            if not images_dir.exists():
                print(f"WARNING: Images dir not found at {images_dir}")
                continue

            # Find valid images
            found_ref = None

            # We need to scan images to find one at 9:30 AM Edmonton time
            # Filenames are typically: YYYY-MM-DDTHHMMSS+0000_left.jpg (UTC)

            # Sorting ensures we look at days in order. We usually want the first day's reference.
            # But the user might want *any* reference from that zone, usually the first day is best.
            all_images = sorted(list(images_dir.glob("*.jpg")))

            for img_path in all_images:
                # Parse filename
                try:
                    # Remove _left.jpg suffix for parsing
                    ts_str = img_path.name.split("_")[0]
                    # Format: 2025-08-20T042000+0000
                    # Standard isoformat should handle it
                    dt_utc = datetime.fromisoformat(ts_str)

                    # Convert to Edmonton
                    dt_yeg = dt_utc.astimezone(edmonton_tz)

                    if dt_yeg.hour == 9 and dt_yeg.minute == 30:
                        found_ref = img_path
                        # print(f"Found ref: {img_path.name} -> {dt_yeg}")
                        break
                except ValueError:
                    continue

            if found_ref:
                # Destination filename: E13Z01_<original file name>
                dest_name = f"{exp_part}{zone_prefix}_{found_ref.name}"
                dest_path = OUTPUT_DIR / dest_name

                print(f"Found {found_ref.name} (9:30 YEG) -> Copying to {dest_name}")
                shutil.copy2(found_ref, dest_path)
            else:
                print(f"WARNING: No 9:30 AM (Edmonton) image found in {images_dir}")
                # Fallback: maybe list strict UTC matches if we were doing the naive thing before?
                # But since we are "being smarter", we trust the timezone logic.

        except Exception as e:
            print(f"ERROR processing {zone_path}: {e}")


if __name__ == "__main__":
    main()
