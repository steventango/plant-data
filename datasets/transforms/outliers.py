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
        DataFrame with added "valid" column indicating non-outlier rows
    """
    ql = df["reward"].quantile(q1)
    qu = df["reward"].quantile(q2)
    df = df.with_columns(
        ((pl.col("reward") < ql) | (pl.col("reward") > qu)).alias("outlier")
    )
    invalid_count = df["outlier"].sum()
    print(f"marked {invalid_count} / {df.height} daily rows as outliers")
    return df