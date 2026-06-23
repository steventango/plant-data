#!/usr/bin/env python3
"""
Shared helpers for E18 power / energy analysis.

WHY THIS EXISTS
---------------
The offline parquet (mixed-vXX) stores `power` only at the 4-hourly subsample
points (09:00 / 13:00 / 17:00 local). Summing that column is *not* a correct
energy integral for two reasons:

  1. The 09:00 sample lands during the lights-on ramp-up (the smart plug reads
     ~20 W mid-transition instead of the steady ~47 W), so it is a low outlier
     for every zone.
  2. Three points per day cannot capture within-day ramp shapes
     (Parabolic / LateRamp etc.), so the integral is badly biased.

These helpers instead integrate the *raw* (~10 s resolution) power logs over the
full photoperiod, which reproduces the designed energy fractions. Energy is
estimated as (mean photoperiod power) x (photoperiod hours), which is robust to
the irregular / duplicated raw timestamps.
"""

import json
from datetime import time
from pathlib import Path

import polars as pl

# Photoperiod for E18: lights on 09:00-21:00 local (config timezone).
PHOTOPERIOD_START = time(9, 0)
PHOTOPERIOD_END = time(21, 0)
PHOTOPERIOD_HOURS = 12.0
DEFAULT_TZ = "Etc/GMT-2"

E18_RAW_ROOT = Path("/data/plant-rl/online/E18/P1")

# zone -> raw run directory (authoritative: generate.sh P1 block).
E18_RAW_DIRS = {
    1: "SequencePowerLawRamp1/alliance-zone01",
    2: "SequenceParabolic2/alliance-zone02",
    3: "ConstantLow3/alliance-zone03",
    4: "SequenceSeventyPercentRamp4/alliance-zone04",
    5: "SequenceLateRamp5/alliance-zone05",
    6: "Schedule6/alliance-zone06",
    7: "Schedule7/alliance-zone07",
    8: "Schedule8/alliance-zone08",
    9: "Schedule9/alliance-zone09",
    10: "Schedule10/alliance-zone10",
    11: "Constant11/alliance-zone11",
}


def _zone_timezone(zone_dir: Path) -> str:
    config_path = zone_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                return json.load(f).get("timezone", DEFAULT_TZ)
        except (json.JSONDecodeError, OSError):
            pass
    return DEFAULT_TZ


def daily_power_energy(
    zones=None,
    raw_root: Path = E18_RAW_ROOT,
    start_date=None,
) -> pl.DataFrame:
    """Integrate raw power over the photoperiod for each E18 zone and day.

    Returns a long DataFrame with columns:
        zone, date, day, mean_power_W, energy_Wh
    where ``mean_power_W`` is the mean power over the photoperiod and
    ``energy_Wh = mean_power_W * PHOTOPERIOD_HOURS``. ``day`` is the integer day
    index relative to the earliest photoperiod date observed (across all zones),
    or relative to ``start_date`` if given.
    """
    if zones is None:
        zones = list(E18_RAW_DIRS)

    frames = []
    for zone in zones:
        rel = E18_RAW_DIRS.get(zone)
        if rel is None:
            continue
        zone_dir = raw_root / rel
        files = sorted(zone_dir.glob("raw_*.csv"))
        if not files:
            continue
        tz = _zone_timezone(zone_dir)
        parts = []
        for f in files:
            try:
                parts.append(
                    pl.read_csv(
                        f,
                        try_parse_dates=True,
                        infer_schema_length=10000,
                        columns=["time", "power"],
                    )
                )
            except Exception:
                continue
        if not parts:
            continue
        df = (
            pl.concat(parts, how="diagonal_relaxed")
            .unique(subset=["time"])
            .drop_nulls("power")
            .with_columns(pl.col("time").dt.convert_time_zone(tz))
        )
        df = df.filter(
            (pl.col("time").dt.time() >= PHOTOPERIOD_START)
            & (pl.col("time").dt.time() < PHOTOPERIOD_END)
        )
        if df.is_empty():
            continue
        per_day = (
            df.with_columns(pl.col("time").dt.date().alias("date"))
            .group_by("date")
            .agg(pl.col("power").mean().alias("mean_power_W"))
            .with_columns(
                pl.lit(zone).alias("zone"),
                (pl.col("mean_power_W") * PHOTOPERIOD_HOURS).alias("energy_Wh"),
            )
        )
        frames.append(per_day)

    if not frames:
        return pl.DataFrame(
            schema={
                "zone": pl.Int64,
                "date": pl.Date,
                "day": pl.Int64,
                "mean_power_W": pl.Float64,
                "energy_Wh": pl.Float64,
            }
        )

    out = pl.concat(frames, how="diagonal_relaxed")
    anchor = start_date if start_date is not None else out["date"].min()
    out = out.with_columns(
        ((pl.col("date") - pl.lit(anchor)).dt.total_days()).alias("day")
    )
    return out.sort("zone", "date").select(
        "zone", "date", "day", "mean_power_W", "energy_Wh"
    )
