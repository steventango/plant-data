from datetime import date
import polars as pl


EXPERIMENT_EVENTS = {
    8: {
        "transplant_date": date(2025, 5, 2),
        "water_transplant_l": 1.0,
    },
    9: {
        "sterilized_date": date(2025, 5, 12),
        "plate_date": date(2025, 5, 15),
        "transplant_date": date(2025, 5, 23),
        "water_transplant_l": 1.0,
        "remove_domes_date": date(2025, 5, 26),
        "water_remove_domes_l": 2.0,
    },
    10: {
        "sterilized_date": date(2025, 6, 16),
        "plate_date": date(2025, 6, 19),
        "transplant_date": date(2025, 6, 27),
        "water_transplant_l": 1.0,
        "remove_domes_date": date(2025, 6, 30),
        "water_remove_domes_l": 2.0,
    },
    11: {
        "sterilized_date": date(2025, 6, 30),
        "plate_date": date(2025, 7, 3),
        "transplant_date": date(2025, 7, 11),
        "water_transplant_l": 1.0,
        "remove_domes_date": date(2025, 7, 14),
        "water_remove_domes_l": 2.0,
    },
    12: {
        "sterilized_date": date(2025, 9, 2),
        "plate_date": date(2025, 9, 5),
        "transplant_date": date(2025, 9, 12),
        "water_transplant_l": 1.0,
        "remove_domes_date": date(2025, 9, 15),
        "water_remove_domes_l": 2.0,
    },
    13: {
        "sterilized_date": date(2025, 9, 27),
        "plate_date": date(2025, 10, 1),  # Corrected from Sept 31
        "transplant_date": date(2025, 10, 2),
        "water_transplant_l": 1.0,
        "remove_domes_date": date(2025, 10, 5),
        "water_remove_domes_l": 2.0,
    },
    14: {
        "sterilized_date": date(2025, 10, 27),
        "transplant_date": date(2025, 11, 4),
        "water_transplant_l": 1.0,
        "remove_domes_date": date(2025, 11, 7),
        "water_remove_domes_l": 2.0,
    },
    15: {
        "sterilized_date": date(2025, 12, 1),
        "plate_date": date(2025, 12, 4),
        "transplant_date": date(2025, 12, 11),
        "water_transplant_l": 1.0,
        "remove_domes_date": date(2025, 12, 15),
        "water_remove_domes_l": 2.0,
    },
}


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
    - dates and water amounts from EXPERIMENT_EVENTS
    """
    df = df.with_columns(
        pl.lit(exp_id).alias("experiment"),
        pl.lit(zone_id).alias("zone"),
    )
    df = get_agent_name(df)

    if exp_id in EXPERIMENT_EVENTS:
        events = EXPERIMENT_EVENTS[exp_id]

        cols_to_add = []
        for key, value in events.items():
            cols_to_add.append(pl.lit(value).alias(key))

        if cols_to_add:
            df = df.with_columns(cols_to_add)

    return df
