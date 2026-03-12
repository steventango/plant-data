from datetime import date
import polars as pl


EXPERIMENT_EVENTS = {
    7: {
        "transplant_date": date(2025, 5, 2),
        "watering": {
            date(2025, 5, 2): 1.0,
        },
    },
    8: {
        "sterilized_date": date(2025, 5, 12),
        "plate_date": date(2025, 5, 15),
        "transplant_date": date(2025, 5, 23),
        "remove_domes_date": date(2025, 5, 26),
        "watering": {
            date(2025, 5, 23): 1.0,
            date(2025, 5, 26): 2.0,
        },
    },
    9: {
        "sterilized_date": date(2025, 6, 16),
        "plate_date": date(2025, 6, 19),
        "transplant_date": date(2025, 6, 27),
        "remove_domes_date": date(2025, 6, 30),
        "watering": {
            date(2025, 6, 27): 1.0,
            date(2025, 6, 30): 2.0,
        },
    },
    10: {
        "sterilized_date": date(2025, 6, 30),
        "plate_date": date(2025, 7, 3),
        "transplant_date": date(2025, 7, 11),
        "remove_domes_date": date(2025, 7, 14),
        "watering": {
            date(2025, 7, 11): 1.0,
            date(2025, 7, 14): 2.0,
        },
        "num_pots": 20,
        "num_pots_per_tray": 20,
    },
    11: {
        "sterilized_date": date(2025, 8, 5),
        "plate_date": date(2025, 8, 8),
        "transplant_date": date(2025, 8, 15),
        "remove_domes_date": date(2025, 8, 18),
        "watering": {
            date(2025, 8, 15): 1.0,
            date(2025, 8, 18): 2.0,
        },
        "num_pots": 18,
        "num_pots_per_tray": 18,
    },
    12: {
        "sterilized_date": date(2025, 9, 2),
        "plate_date": date(2025, 9, 5),
        "transplant_date": date(2025, 9, 12),
        "remove_domes_date": date(2025, 9, 15),
        "watering": {
            date(2025, 9, 12): 1.0,
            date(2025, 9, 15): 2.0,
        },
        "num_pots": 18,
        "num_pots_per_tray": 18,
    },
    13: {
        "sterilized_date": date(2025, 9, 27),
        "plate_date": date(2025, 10, 1),  # Corrected from Sept 31
        "transplant_date": date(2025, 10, 2),
        "remove_domes_date": date(2025, 10, 5),
        "watering": {
            date(2025, 10, 2): 1.0,
            date(2025, 10, 5): 2.0,
        },
        "num_pots": 64,
        "num_pots_per_tray": 32,
    },
    14: {
        "sterilized_date": date(2025, 10, 27),
        "transplant_date": date(2025, 11, 4),
        "remove_domes_date": date(2025, 11, 7),
        "watering": {
            date(2025, 11, 4): 1.0,
            date(2025, 11, 17): 2.0,
        },
        "num_pots": 64,
        "num_pots_per_tray": 32,
    },
    15: {
        "sterilized_date": date(2025, 12, 1),
        "plate_date": date(2025, 12, 4),
        "transplant_date": date(2025, 12, 11),
        "remove_domes_date": date(2025, 12, 15),
        "watering": {
            date(2025, 12, 11): 1.0,
            date(2025, 12, 15): 2.0,
        },
        "num_pots": 64,
        "num_pots_per_tray": 32,
    },
    16: {
        "sterilized_date": date(2026, 1, 19),
        "plate_date": date(2026, 1, 23),
        "transplant_date": date(2026, 1, 30),
        "remove_domes_date": date(2026, 2, 2),
        "watering": {
            date(2026, 1, 30): 1.0,
            date(2026, 2, 2): 0.25,
            date(2026, 2, 5): 0.5,
            date(2026, 2, 6): 0.25,
            date(2026, 2, 8): 0.25,
            date(2026, 2, 9): 0.5,
            date(2026, 2, 11): 0.5,
            date(2026, 2, 13): 0.25,
            date(2026, 2, 15): 0.5,
            date(2026, 2, 17): 0.25,
            date(2026, 2, 19): 0.25,
        },
        "num_pots": 36,
        "num_pots_per_tray": 18,
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
        .when(pl.col("experiment") == 16)
        .then(
            pl.when(pl.col("zone").is_in([1, 5, 9]))
            .then(pl.lit("Constant_White"))
            .when(pl.col("zone").is_in([3, 7, 12]))
            .then(pl.lit("InAC_Seed6"))
            .when(pl.col("zone").is_in([2, 6, 11]))
            .then(pl.lit("InAC_Seed7"))
            .when(pl.col("zone").is_in([4, 8, 10]))
            .then(pl.lit("InAC_Seed21"))
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
            if key == "watering":
                continue
            cols_to_add.append(pl.lit(value).alias(key))

        if cols_to_add:
            df = df.with_columns(cols_to_add)

    return df
