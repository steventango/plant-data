import polars as pl


def get_agent_name(df: pl.DataFrame) -> pl.DataFrame:
    """
    Polars implementation of agent name mapping logic.
    """
    return df.with_columns(
        pl.when(pl.col("experiment").is_in([11, 12]))
        .then(pl.lit("Uniform_Discrete"))
        .when(pl.col("experiment") == 13)
        .then(pl.lit("Uniform_Dirichlet"))
        .when(pl.col("experiment") == 14)
        .then(
            pl.when(pl.col("zone") == 1)
            .then(pl.lit("Constant_White"))
            .when(pl.col("zone") == 2)
            .then(pl.lit("InAC_Data_Det"))
            .when(pl.col("zone").is_in([3, 4, 5, 6]))
            .then(pl.lit("InAC_Data_Sto"))
            .when(pl.col("zone") == 8)
            .then(pl.lit("InAC_GP_Det_Opt0"))
            .when(pl.col("zone") == 9)
            .then(pl.lit("InAC_GP_Det_Opt0.25"))
            .when(pl.col("zone") == 10)
            .then(pl.lit("InAC_GP_Det_Opt0.5"))
            .when(pl.col("zone") == 11)
            .then(pl.lit("InAC_GP_Det_Opt0.75"))
            .when(pl.col("zone") == 12)
            .then(pl.lit("InAC_GP_Det_Opt1"))
            .otherwise(pl.lit("Other"))
        )
        .when(pl.col("experiment") == 15)
        .then(
            pl.when(pl.col("zone") == 2)
            .then(pl.lit("InAC_2"))
            .when(pl.col("zone") == 3)
            .then(pl.lit("InAC_3"))
            .when(pl.col("zone") == 4)
            .then(pl.lit("InAC_4"))
            .otherwise(pl.lit("Other"))
        )
        .otherwise(pl.lit("Other"))
        .alias("agent")
    )


def transform_experiment_attributes(
    df: pl.DataFrame, exp_id: int, zone_id: int
) -> pl.DataFrame:
    """
    Add experiment-specific attributes to the dataframe, including:
    - experiment ID
    - zone ID
    - agent name (derived from experiment and zone)
    """
    df = df.with_columns(
        pl.lit(exp_id).alias("experiment"),
        pl.lit(zone_id).alias("zone"),
    )
    return get_agent_name(df)
