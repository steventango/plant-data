import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import minari
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Plot plant area for specific experiment and zone from Minari rollouts."
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default="plant-data/mixed-v23",
        help="Name of the Minari dataset",
    )
    parser.add_argument(
        "--experiment",
        "-e",
        type=int,
        required=True,
        help="Experiment number (e.g., 14)",
    )
    parser.add_argument(
        "--zone",
        "-z",
        type=int,
        required=True,
        help="Zone number (e.g., 1)",
    )
    parser.add_argument(
        "--out",
        "-o",
        default="results/rollout_plots",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        logging.info(f"Loading dataset: {args.dataset}")
        dataset = minari.load_dataset(args.dataset)
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    found_episodes = 0

    # In these datasets, clean_area is typically observation dimension 1
    # Wall time is typically observation dimension 0

    for episode in dataset.iterate_episodes():
        # Check if experiment and zone match
        # infos is a dict of arrays, each array has length of episode
        if "experiment" in episode.infos and "zone" in episode.infos:
            # We assume experiment and zone are constant throughout the episode
            exp_val = episode.infos["experiment"][0]
            zone_val = episode.infos["zone"][0]

            # Convert to int if they are strings or floats
            try:
                exp_val = int(exp_val)
                zone_val = int(zone_val)
            except (ValueError, TypeError):
                # If they are strings like 'E14', handle that
                if isinstance(exp_val, str) and exp_val.startswith("E"):
                    try:
                        exp_val = int(exp_val[1:])
                    except ValueError:
                        pass

            if exp_val == args.experiment and zone_val == args.zone:
                found_episodes += 1

                # Observations: (T+1, D)
                # Clean area is usually index 1
                wall_time = episode.observations[:, 0]
                wall_time_filter = wall_time < 14
                area = episode.observations[wall_time_filter, 1]
                wall_time = wall_time[wall_time_filter]

                # Plotting against wall time
                plt.plot(wall_time, area, alpha=0.6, label=f"Episode {episode.id}")

    if found_episodes == 0:
        logging.warning(
            f"No episodes found for Experiment {args.experiment}, Zone {args.zone}"
        )
    else:
        logging.info(f"Found {found_episodes} episodes.")
        plt.title(f"Plant Area for Experiment {args.experiment}, Zone {args.zone}")
        plt.xlabel("Wall Time (days)")
        plt.ylabel("Clean Area")
        plt.ylim(0, 1250)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        filename = f"area_E{args.experiment}_Z{args.zone}.png"
        out_path = out_dir / filename
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        logging.info(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
