"""Create anffany-v1.1.parquet from mixed-v27.parquet (E18 only)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import polars as pl

from config import VERSION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPERIMENT = 18

DROP_COLUMNS = [
    "action_coefficients",
    "red_coef",
    "white_coef",
    "blue_coef",
    "days_since_sterilization",
    "days_since_plate",
    "days_since_transplant",
    "days_since_dome_removal",
    "days_since_watering",
    "liters_per_pot",
    "is_good_day",
    "cls_token",
    "cls_token_pca",
    "cls_token_umap",
    "sterilized_date",
    "plate_date",
    "transplant_date",
    "remove_domes_date",
    "num_pots",
    "num_pots_per_tray",
    "bolted_prob",
    "bolted_pred",
]
for i in range(6):
    for trace in (0.5, 0.7, 0.9):
        DROP_COLUMNS.append(f"action.{i}_trace_{trace}")
for color in ("red", "white", "blue"):
    for trace in (0.5, 0.7, 0.9):
        DROP_COLUMNS.append(f"{color}_coef_trace_{trace}")


def main() -> None:
    default_dir = Path(f"/data/plant-rl/offline/{VERSION}")
    parser = argparse.ArgumentParser(
        description="Filter mixed parquet to E18 and write anffany-v1.parquet."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_dir / f"mixed-{VERSION}.parquet",
        help="Input mixed parquet path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_dir / "anffany-v1.1.parquet",
        help="Output parquet path",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    logger.info("Reading %s", args.input)
    df = pl.read_parquet(args.input)

    before = df.height
    df = df.filter(pl.col("experiment") == EXPERIMENT)
    logger.info("Kept %d / %d rows (E%d)", df.height, before, EXPERIMENT)

    zones = df.select("zone").unique().sort("zone").to_series().to_list()
    logger.info("Zones in output: %s", zones)

    drop = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(drop)
    logger.info("Dropped %d columns (%d remaining)", len(drop), len(df.columns))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.output)
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
