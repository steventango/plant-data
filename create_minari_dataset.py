import argparse
import logging
from pathlib import Path

import polars as pl
from minari import DataCollector
from sklearn.model_selection import GroupKFold

from config import COLS, VERSION
from env import MockEnv
from transforms import transform_terminal
from transforms.normalization import load_normalization_stats

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Convert processed dataset to Minari.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=f"/data/plant-rl/offline/{VERSION}/",
        help="Directory containing the RL filtered dataset",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    input_path = input_dir / f"mixed-{VERSION}.parquet"

    logging.info(f"Loading dataset from {input_path}")

    try:
        df = pl.read_parquet(input_path)
    except FileNotFoundError:
        logging.error(f"File not found: {input_path}. Please run join_zones.py first.")
        return

    stats = load_normalization_stats(input_dir / f"normalization-stats-{VERSION}.json")

    df = df.filter(~pl.col("outlier"))
    df_filtered = df.filter(pl.col("day") < 15)
    df = transform_terminal(df)

    def create_dataset(df: pl.DataFrame, name: str):
        mock_env = MockEnv(df, stats, COLS)
        env = DataCollector(mock_env, record_infos=True)

        # Run episodes until environment indicates all data has been processed
        logging.info("Generating Minari dataset...")
        while not mock_env.is_done():
            obs, info = env.reset(seed=0)

            while not mock_env.done:
                action = env.unwrapped.action
                obs, rew, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    break

        dataset = env.create_dataset(
            dataset_id=f"plant-data/{name}-{VERSION}",
            algorithm_name="None",
            code_permalink="https://github.com/steventango/plant-data",
            author="Steven Tang",
            author_email="stang5@ualberta.ca",
        )

        print(f"Dataset created: {dataset.id}")

    gkf = GroupKFold(n_splits=5)
    groups_df = df.select(["experiment", "zone"]).with_columns(
        pl.concat_str([pl.col("experiment"), pl.col("zone")], separator="-").alias(
            "group"
        )
    )
    groups = groups_df["group"].to_numpy()

    logging.info("Generating GroupKFold datasets...")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups)):
        logging.info(f"Processing Fold {fold}...")

        df_train = df[train_idx]
        df_val = df[val_idx]

        create_dataset(df_train, f"mixed-all-fold{fold}-train")
        create_dataset(df_val, f"mixed-all-fold{fold}-val")

    create_dataset(df_filtered, "mixed")
    create_dataset(df, "mixed-all")


if __name__ == "__main__":
    main()
