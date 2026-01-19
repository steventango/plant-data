import polars as pl


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


def transform_frequencies(df: pl.DataFrame) -> pl.DataFrame:
    for col in df.columns:
        if "frequencies" in col:
            # Skip if column is not a List type (e.g. if it's Null because all values are None)
            if not isinstance(df.schema[col], pl.List):
                continue

            df = df.with_columns(
                pl.col(col)
                .list.eval(
                    pl.int_range(0, pl.len()).repeat_by(pl.element()).explode().median()
                )
                .list.first()
                .alias(f"{col}_mean")
            )
    return df


def transform_days_since_events(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate days since specific events:
    - days_since_sterilization
    - days_since_plate
    - days_since_transplant
    - days_since_dome_removal

    Should be called after transform_attributes has added the event date columns.
    """
    df = df.with_columns(
        (pl.col("time").dt.date() - pl.col("sterilized_date"))
        .dt.total_days()
        .alias("days_since_sterilization"),
        (pl.col("time").dt.date() - pl.col("plate_date"))
        .dt.total_days()
        .alias("days_since_plate"),
        (pl.col("time").dt.date() - pl.col("transplant_date"))
        .dt.total_days()
        .alias("days_since_transplant"),
        (pl.col("time").dt.date() - pl.col("remove_domes_date"))
        .dt.total_days()
        .alias("days_since_dome_removal"),
    )
    return df


def transform_state(df: pl.DataFrame) -> pl.DataFrame:
    """
    Transform state-related columns including clean area computation.
    """
    df = transform_frequencies(df)
    df = transform_days_since_events(df)
    df = transform_wall_time(df)
    df = df.with_columns(
        pl.col("clean_area")
        .mean()
        .over("experiment", "zone", "time")
        .alias("mean_clean_area"),
    )
    return df
