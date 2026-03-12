import minari
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm


def get_viz_path(orig_path, experiment, zone):
    """Convert original image path to visualization path."""
    if isinstance(orig_path, bytes):
        orig_path = orig_path.decode("utf-8")
    p = Path(orig_path)
    timestamp = p.stem
    # /images/0/2025-11-12T093000.jpg -> parent.parent.parent is /processed/v21/
    viz_dir = p.parent.parent.parent / "visualizations"
    viz_filename = f"E{experiment}_Z{zone}_{timestamp}_viz.jpg"
    viz_path = viz_dir / viz_filename
    return viz_path


def plot_episode(episode, episode_id, percentile, output_dir):
    rewards = episode.rewards
    timesteps = np.arange(len(rewards))
    cumulative_rewards = np.cumsum(rewards)

    experiment = episode.infos["experiment"][0]
    zone = episode.infos["zone"][0]
    plant_id = episode.infos["plant_id"][0]
    image_paths = episode.infos["image_path"]

    total_return = np.sum(rewards)

    # Create figure
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 5)  # 2 rows, 5 columns for images

    # Plot cumulative rewards
    ax_rew = fig.add_subplot(gs[0, :])
    ax_rew.plot(
        timesteps,
        cumulative_rewards,
        label=f"Return: {total_return:.2f}",
        color="green",
        linewidth=2,
    )
    ax_rew.set_title(
        f"Episode {episode_id} (P{percentile}) | E{experiment}Z{zone}P{plant_id} | Total Return: {total_return:.2f}"
    )
    ax_rew.set_xlabel("Timestep")
    ax_rew.set_ylabel("Cumulative Reward")
    ax_rew.legend()
    ax_rew.grid(True, alpha=0.3)

    # Select key frames (5 frames: start, 25%, 50%, 75%, end)
    num_frames = 5
    indices = np.linspace(0, len(image_paths) - 1, num_frames, dtype=int)

    for i, idx in enumerate(indices):
        img_path = get_viz_path(image_paths[idx], experiment, zone)
        ax_img = fig.add_subplot(gs[1, i])

        if img_path.exists():
            img = Image.open(img_path)
            ax_img.imshow(img)
            ax_img.set_title(f"Step {idx}")
        else:
            ax_img.text(0.5, 0.5, "Image not found", ha="center", va="center")
            ax_img.set_title(f"Step {idx} (Missing)")

        ax_img.axis("off")

    plt.tight_layout()
    output_path = output_dir / f"episode_{episode_id}_p{percentile}.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot rollouts at return percentiles.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="plant-data/mixed-v22",
        help="Minari dataset name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/rollout_percentiles",
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset {args.dataset}...")
    dataset = minari.load_dataset(args.dataset)

    episode_returns = []
    episode_ids = []

    print("Calculating episode returns (excluding truncated rollouts)...")
    for i, episode in enumerate(tqdm(dataset.iterate_episodes())):
        # Check if the episode is terminal (not truncated)
        # Minari episodes have terminations and truncations
        if episode.terminations[-1]:
            episode_returns.append(np.sum(episode.rewards))
            episode_ids.append(i)

    episode_returns = np.array(episode_returns)
    episode_ids = np.array(episode_ids)

    # Sort by return
    sorted_indices = np.argsort(episode_returns)
    sorted_returns = episode_returns[sorted_indices]
    sorted_ids = episode_ids[sorted_indices]

    percentiles = [0, 25, 50, 75, 100]
    selected_indices = []

    for p in percentiles:
        # Find index closest to percentile
        idx = int(np.round((p / 100.0) * (len(sorted_returns) - 1)))
        selected_indices.append(idx)

    print(f"Total episodes: {len(sorted_returns)}")
    print("Selected episodes:")
    for i, p in enumerate(percentiles):
        idx = selected_indices[i]
        ep_id = sorted_ids[idx]
        ret = sorted_returns[idx]

        # Load episode to get info
        episode = dataset[ep_id]
        experiment = episode.infos["experiment"][0]
        zone = episode.infos["zone"][0]
        plant_id = episode.infos["plant_id"][0]

        # Decode bytes if necessary
        if isinstance(experiment, bytes):
            experiment = experiment.decode("utf-8")
        if isinstance(zone, bytes):
            zone = zone.decode("utf-8")
        if isinstance(plant_id, bytes):
            plant_id = plant_id.decode("utf-8")

        print(
            f"P{p:3d}: Episode ID {ep_id:5d}, Return {ret:8.2f} | E{experiment} Z{zone} P{plant_id}"
        )
        plot_episode(episode, ep_id, p, output_dir)


if __name__ == "__main__":
    main()
