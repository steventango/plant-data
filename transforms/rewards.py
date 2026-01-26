import polars as pl


def transform_reward(df):
    df = df.sort("experiment", "zone", "plant_id", "time")
    df = df.with_columns(
        pl.col("log_clean_area")
        .shift(1)
        .over("experiment", "zone", "plant_id")
        .alias("prev_log_clean_area"),
    )
    df = df.with_columns(
        (pl.col("log_clean_area") - pl.col("prev_log_clean_area")).alias("reward"),
    )
    df = df.drop("prev_log_clean_area")
    # if bolted, reward = 0
    df = df.with_columns(
        pl.when(pl.col("bolted_pred"))
        .then(0)
        .otherwise(pl.col("reward"))
        .alias("reward")
    )
    return df
