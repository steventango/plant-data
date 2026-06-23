import polars as pl

_STEP_HOURS = 5.0 / 60.0


def transform_power(df: pl.DataFrame) -> pl.DataFrame:
    if "power" not in df.columns:
        return df

    df = df.with_columns(pl.col("power").shift(-1))
    df = df.sort("experiment", "zone", "time")
    df = df.with_columns(
        (pl.col("power").fill_null(0.0) * _STEP_HOURS).alias("energy")
    )
    df = df.with_columns(
        pl.col("energy").cum_sum().over("experiment", "zone").alias("cumulative_energy")
    )
    return df
