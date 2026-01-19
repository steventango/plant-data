import polars as pl


def transform_outlier_detection(
    df: pl.DataFrame, q1: float = 0.01, q2: float = 0.99
) -> pl.DataFrame:
    """Mark rows with outlier rewards outside specified percentile thresholds.

    Args:
        df: DataFrame with a "reward" column
        q1: Lower percentile threshold
        q2: Upper percentile threshold

    Returns:
        DataFrame with added "nodata", "outlier", "valid" columns indicating valid rows
    """
    # if clean_area is 0, mark as nodata
    df = df.with_columns((pl.col("clean_area") == 0).alias("nodata"))
    nodata_count = df["nodata"].sum()
    print(f"marked {nodata_count} / {df.height} daily rows as nodata")
    ql = df["reward"].quantile(q1)
    qu = df["reward"].quantile(q2)
    df = df.with_columns(
        ((pl.col("reward") < ql) | (pl.col("reward") > qu))
        .fill_null(False)
        .alias("outlier")
    )
    outlier_count = df["outlier"].sum()
    print(f"marked {outlier_count} / {df.height} daily rows as outliers")
    # valid is not nodata and not outlier
    df = df.with_columns((~(pl.col("nodata") | pl.col("outlier"))).alias("valid"))
    valid_count = df["valid"].sum()
    print(f"marked {valid_count} / {df.height} daily rows as valid")
    return df
