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

def transform_state(df: pl.DataFrame) -> pl.DataFrame:
    """
    Transform state-related columns including clean area computation.
    """
    df = transform_frequencies(df)
    df = transform_wall_time(df)
    df = df.with_columns(
        pl.col("clean_area")
        .mean()
        .over("experiment", "zone", "time")
        .alias("mean_clean_area"),
    )
    return df
