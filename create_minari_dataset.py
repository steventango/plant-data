import logging
import argparse
from pathlib import Path

import polars as pl
from env import MockEnv
from minari import DataCollector
from config import VERSION
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

    # Filter for RL
    # TODO: consider keeping some of the weird days
    df = df.filter((pl.col("day") <= 13) & pl.col("is_good_day") & ~pl.col("outlier"))

    stats = load_normalization_stats(input_dir / f"normalization-stats-{VERSION}.json")

    cols = [
        "wall_time",
        "clean_area",
        "clean_convex_hull_area",
        "clean_solidity",
        "clean_perimeter",
        "clean_width",
        "clean_height",
        "clean_longest_path",
        "clean_center_of_mass_x",
        "clean_center_of_mass_y",
        "clean_convex_hull_vertices",
        "clean_ellipse_center_x",
        "clean_ellipse_center_y",
        "clean_ellipse_major_axis",
        "clean_ellipse_minor_axis",
        "clean_ellipse_angle",
        "clean_ellipse_eccentricity",
        "red_coef_trace_0.9",
        "white_coef_trace_0.9",
        "blue_coef_trace_0.9",
    ]

    # Create continuous action dataset
    mock_env = MockEnv(df, stats, cols)
    env = DataCollector(mock_env, record_infos=True)

    # Run episodes until environment indicates all data has been processed
    logging.info("Generating Minari dataset...")
    while not mock_env.is_done():
        obs, info = env.reset(seed=0)

        while True:
            action = info["action"]
            obs, rew, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

    dataset = env.create_dataset(
        dataset_id=f"plant-data/mixed-{VERSION}",
        algorithm_name="None",
        code_permalink="https://github.com/steventango/plant-data",
        author="Steven Tang",
        author_email="stang5@ualberta.ca",
    )

    print(f"Continuous action dataset created: {dataset.id}")
    print("Dataset statistics:")
    print(f"  Total episodes: {len(list(dataset.iterate_episodes()))}")
    print(f"  Observation space: {mock_env.observation_space}")
    print(f"  Action space: {mock_env.action_space}")


if __name__ == "__main__":
    main()
