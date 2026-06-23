import polars as pl

from .attributes import EXPERIMENT_EVENTS


def transform_wall_time(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("time")
        .min()
        .over(["experiment", "zone"])
        .dt.truncate("1d")
        .alias("first_day_midnight")
    )
    df = df.with_columns(
        (pl.col("first_day_midnight") + pl.duration(hours=9, minutes=30)).alias(
            "ref_time"
        )
    )
    df = df.with_columns(
        ((pl.col("time") - pl.col("ref_time")) / pl.duration(days=1)).alias("wall_time")
    )
    df = df.drop(["first_day_midnight", "ref_time"])
    return df


def transform_days_since_events(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate days since specific events.
    Handles missing base columns by setting them to None.
    """
    event_mapping = {
        "days_since_sterilization": "sterilized_date",
        "days_since_plate": "plate_date",
        "days_since_transplant": "transplant_date",
        "days_since_dome_removal": "remove_domes_date",
    }

    cols = []
    for alias, base_col in event_mapping.items():
        if base_col in df.columns:
            cols.append(
                (pl.col("time").dt.date() - pl.col(base_col))
                .dt.total_days()
                .alias(alias)
            )
        else:
            cols.append(pl.lit(None).cast(pl.Int64).alias(alias))

    if cols:
        df = df.with_columns(cols)

    return df


def transform_watering_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate days since last watering and liters per pot.
    Uses join_asof to find the most recent watering event for each row.
    """
    # Create metadata dataframe for watering events
    watering_data = []
    for exp_id, events in EXPERIMENT_EVENTS.items():
        if "watering" in events and "num_pots_per_tray" in events:
            num_pots_per_tray = events["num_pots_per_tray"]
            for w_date, liters in events["watering"].items():
                watering_data.append(
                    {
                        "experiment": exp_id,
                        "watering_date": w_date,
                        "liters_per_pot": liters / num_pots_per_tray,
                    }
                )

    if not watering_data:
        return df.with_columns(
            pl.lit(None).cast(pl.Int64).alias("days_since_watering"),
            pl.lit(None).cast(pl.Float64).alias("liters_per_pot"),
        )

    w_df = pl.DataFrame(watering_data)

    df = df.with_columns(pl.col("experiment").cast(pl.Int32))
    w_df = w_df.with_columns(pl.col("experiment").cast(pl.Int32))

    df = df.with_columns(pl.col("time").dt.date().alias("_date"))

    # Sort both dataframes as required by join_asof
    df = df.sort("experiment", "time")
    w_df = w_df.sort("experiment", "watering_date")

    # Perform asof join to find the latest watering event <= current row date
    df = df.join_asof(
        w_df,
        left_on="_date",
        right_on="watering_date",
        by="experiment",
        strategy="backward",
    )

    # Calculate days since watering
    df = df.with_columns(
        (pl.col("_date") - pl.col("watering_date"))
        .dt.total_days()
        .alias("days_since_watering")
    )

    # Clean up temporary columns
    df = df.drop(["_date", "watering_date"])

    return df


def transform_log_clean_area(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate log_clean_area = log(clean_area).

    When clean_area == 0 (e.g., plant-cv's EWM cleaning returned its 0
    initial state because the first frames were outliers), log_clean_area
    is set to NULL rather than 0. Otherwise transform_reward computes
    reward = log_clean_area_t − log_clean_area_{t-1} and the transition
    from a sentinel 0 to a real value produces a spurious log-jump (e.g.
    reward = log(111) − 0 = 4.7 at day 0).
    """
    if "clean_area" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("clean_area") > 0)
            .then(pl.col("clean_area").log())
            .otherwise(None)
            .alias("log_clean_area")
        )
    return df


def transform_energy(df: pl.DataFrame) -> pl.DataFrame:
    """
    Per-step lighting energy (Wh) over each subsampled interval.

    ``transform_power`` stores per-row energy at full resolution in ``energy``
    and the running total in ``cumulative_energy``. After subsampling, this
    differences ``cumulative_energy`` so ``energy[t]`` is the energy consumed
    over the interval following timestep t. Summing ``energy`` over an episode
    telescopes to total photoperiod energy. The final timestep has no following
    interval and is left null.
    """
    if "cumulative_energy" not in df.columns:
        return df
    df = df.sort("experiment", "zone", "plant_id", "time")
    df = df.with_columns(
        (
            pl.col("cumulative_energy")
            .shift(-1)
            .over("experiment", "zone", "plant_id")
            - pl.col("cumulative_energy")
        ).alias("energy")
    )
    return df


def transform_state(df: pl.DataFrame) -> pl.DataFrame:
    """
    Transform state-related columns including clean area computation.
    """
    df = transform_days_since_events(df)
    df = transform_watering_features(df)
    df = transform_wall_time(df)
    df = transform_energy(df)

    if "clean_area" not in df.columns and "area" in df.columns:
        df = df.with_columns(pl.col("area").alias("clean_area"))

    df = transform_log_clean_area(df)

    if "clean_area" in df.columns:
        df = df.with_columns(
            pl.col("clean_area")
            .mean()
            .over("experiment", "zone", "time")
            .alias("mean_clean_area"),
        )
    return df
