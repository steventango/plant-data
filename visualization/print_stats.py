import polars as pl
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        "-p",
        default="/data/offline/cleaned_offline_dataset_continuous_v16.parquet",
        help="Path to parquet file",
    )
    args = parser.parse_args()
    parquet_file = args.parquet

    print(f"Reading {parquet_file}...")
    try:
        df = pl.read_parquet(parquet_file)
    except Exception as e:
        print(f"Failed to read parquet: {e}")
        return

    # Prefilter to 9:30 AM rows
    df = df.filter(pl.col("time").dt.hour() == 9)
    df = df.filter(pl.col("time").dt.minute() == 30)

    # Print number of unique exp AND zone AND plant ids we have
    print(
        f"Number of unique experiment zone plant ids: {df['experiment', 'zone', 'plant_id'].n_unique()}"
    )
    print(
        f"Number of unique experiment zone ids: {df['experiment', 'zone'].n_unique()}"
    )


if __name__ == "__main__":
    main()
