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
    default_dir = Path(f"/data/plant-rl/offline/{VERSION}/")

    parser = argparse.ArgumentParser(description="Convert processed dataset to Minari.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(default_dir),
        help="Directory containing the RL filtered dataset (used as default for --input "
        "and normalization-stats lookup).",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to the input parquet file. "
        "Default: <input-dir>/mixed-{VERSION}.parquet",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="mixed",
        help="Base name for the Minari dataset IDs. "
        "Emits plant-data/<name>-{VERSION} and plant-data/<name>-all-{VERSION}. "
        "Default: 'mixed'.",
    )
    parser.add_argument(
        "--action",
        choices=["coef", "intensity"],
        default="coef",
        help="Action representation. 'coef' = 3-dim [red_coef, white_coef, blue_coef] "
        "(default). 'intensity' = 1-dim scalar from the 'intensity' column.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    input_path = (
        Path(args.input) if args.input else input_dir / f"mixed-{VERSION}.parquet"
    )

    logging.info(f"Loading dataset from {input_path}")

    try:
        df = pl.read_parquet(input_path)
    except FileNotFoundError:
        logging.error(f"File not found: {input_path}. Please run join_zones.py first.")
        return

    # Locate matching normalization-stats file
    stats_stem = input_path.stem  # e.g. "mixed-v27" or "mixed-e18-daily-v27"
    stats_path = input_dir / f"normalization-stats-{stats_stem}.json"
    if not stats_path.exists():
        stats_path = input_path.parent / f"normalization-stats-{stats_stem}.json"
    # Legacy naming: mixed-v27.parquet paired with normalization-stats-v27.json
    if not stats_path.exists() and stats_stem == f"mixed-{VERSION}":
        for candidate in (
            input_dir / f"normalization-stats-{VERSION}.json",
            input_path.parent / f"normalization-stats-{VERSION}.json",
        ):
            if candidate.exists():
                stats_path = candidate
                break
    logging.info(f"Loading normalization stats from {stats_path}")
    stats = load_normalization_stats(stats_path)

    # Resolve action columns
    action_cols = None
    if args.action == "intensity":
        action_cols = ["intensity"]

    df = df.filter(~pl.col("outlier"))
    df_filtered = df.filter(pl.col("day") < 15)
    df = transform_terminal(df)

    def create_dataset(df: pl.DataFrame, name: str):
        mock_env = MockEnv(df, stats, COLS, action_cols=action_cols)
        env = DataCollector(mock_env, record_infos=True)
        dataset_id = f"plant-data/{name}-{VERSION}"

        # Run episodes until environment indicates all data has been processed
        logging.info("Generating Minari dataset...")
        while not mock_env.is_done():
            obs, info = env.reset(seed=0)

            while not mock_env.done:
                action = env.unwrapped.action
                obs, rew, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    break

        try:
            dataset = env.create_dataset(
                dataset_id=dataset_id,
                algorithm_name="None",
                code_permalink="https://github.com/steventango/plant-data",
                author="Steven Tang",
                author_email="stang5@ualberta.ca",
            )
        except ValueError as exc:
            if "already exists" in str(exc):
                logging.info(f"Skipping existing dataset: {dataset_id}")
                return
            raise

        print(f"Dataset created: {dataset.id}")

    # GroupKFold cross-validation splits (skipped when there are too few groups)
    n_splits = 5
    groups_df = df.select(["experiment", "zone"]).with_columns(
        pl.concat_str([pl.col("experiment"), pl.col("zone")], separator="-").alias(
            "group"
        )
    )
    groups = groups_df["group"].to_numpy()
    n_groups = len(set(groups.tolist()))

    if n_groups >= n_splits:
        gkf = GroupKFold(n_splits=n_splits)
        logging.info("Generating GroupKFold datasets...")
        for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups)):
            logging.info(f"Processing Fold {fold}...")
            df_train = df[train_idx]
            df_val = df[val_idx]
            create_dataset(df_train, f"{args.name}-all-fold{fold}-train")
            create_dataset(df_val, f"{args.name}-all-fold{fold}-val")
    else:
        logging.info(
            f"Skipping GroupKFold (n_groups={n_groups} < n_splits={n_splits})."
        )

    create_dataset(df_filtered, args.name)
    create_dataset(df, f"{args.name}-all")


if __name__ == "__main__":
    main()
