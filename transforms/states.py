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


def transform_state(df: pl.DataFrame) -> pl.DataFrame:
    """
    Transform state-related columns including clean area computation.
    """
    df = transform_days_since_events(df)
    df = transform_watering_features(df)
    df = transform_wall_time(df)

    if "clean_area" not in df.columns and "area" in df.columns:
        df = df.with_columns(pl.col("area").alias("clean_area"))

    if "clean_area" in df.columns:
        df = df.with_columns(
            pl.col("clean_area")
            .mean()
            .over("experiment", "zone", "time")
            .alias("mean_clean_area"),
        )
    return df
