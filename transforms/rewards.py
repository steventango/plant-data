import polars as pl


def transform_reward(df):
    df = df.sort("experiment", "zone", "plant_id", "time")
    df = df.with_columns(
        pl.col("clean_area")
        .filter(pl.col("clean_area") > 0)
        .first()
        .over("experiment", "zone", "plant_id")
        .alias("initial_clean_area")
    )
    df = df.with_columns(
        pl.col("clean_area")
        .shift(1)
        .over("experiment", "zone", "plant_id")
        .alias("prev_clean_area"),
    )
    df = df.with_columns(
        (
            (pl.col("clean_area") - pl.col("prev_clean_area"))
            / pl.col("initial_clean_area")
        ).alias("reward"),
    )
    df = df.drop("initial_clean_area", "prev_clean_area")
    # if bolted, reward = 0
    df = df.with_columns(
        pl.when(pl.col("bolted_pred") > 0.5)
        .then(0)
        .otherwise(pl.col("reward"))
        .alias("reward")
    )
    return df
