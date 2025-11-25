import polars as pl


def transform_state(df):
    df = df.with_columns(
        pl.col("clean_area")
        .mean()
        .over("experiment", "zone", "time")
        .alias("mean_clean_area"),
    )
    return df