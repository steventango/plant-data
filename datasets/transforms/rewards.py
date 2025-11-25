import polars as pl


def transform_reward(df):
    df = df.with_columns(
        pl.col("clean_area").shift(1).over("plant_id").alias("prev_clean_area"),
    )
    df = df.with_columns(
        (
            (pl.col("clean_area") - pl.col("prev_clean_area"))
            / pl.col("prev_clean_area")
        ).alias("reward"),
    )
    df = df.drop("prev_clean_area")
    return df