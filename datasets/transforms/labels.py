from pathlib import Path

import polars as pl


def import_labels(df: pl.DataFrame, label_path: str) -> pl.DataFrame:
    if not Path(label_path).exists():
        print(f"Warning: Label file {label_path} not found. Skipping label import.")
        return df

    print(f"Importing labels from {label_path}...")
    labels_df = pl.read_parquet(label_path)

    if "bolted" not in labels_df.columns:
        print("Warning: 'bolted' column not found in label dataset. Skipping.")
        return df

    labels_subset = labels_df.select(
        ["time", "experiment", "zone", "plant_id", "bolted"]
    )

    df_joined = df.join(
        labels_subset, on=["time", "experiment", "zone", "plant_id"], how="left"
    )

    return df_joined