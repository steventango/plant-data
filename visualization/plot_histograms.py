import os

import matplotlib.pyplot as plt
import minari
import numpy as np
import seaborn as sns


def main():
    # Load dataset
    dataset_name = "plant-data/mixed-v22"
    print(f"Loading dataset: {dataset_name}")
    offline_dataset = minari.load_dataset(dataset_name)

    all_obs = []
    all_actions = []
    all_rewards = []

    print("Collecting data from episodes...")
    for episode in offline_dataset.iterate_episodes():
        all_obs.append(
            episode.observations[:-1]
        )  # Exclude terminal obs for alignment with actions/rewards
        all_actions.append(episode.actions)
        all_rewards.append(episode.rewards)

    observations = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)

    print(f"Observations shape: {observations.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Rewards shape: {rewards.shape}")

    # Set seaborn style
    sns.set_theme(style="darkgrid")
    os.makedirs("results", exist_ok=True)

    from config import COLS

    n_obs_dims = observations.shape[1]
    n_action_dims = actions.shape[1]

    embedding_dim = 768
    num_non_embedding = n_obs_dims - embedding_dim

    # 1. Plot histograms for non-embedding observation dimensions
    print("Plotting non-embedding observation histograms...")
    num_non_embedding = n_obs_dims - embedding_dim
    cols = 4
    rows = (num_non_embedding + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()

    for i in range(num_non_embedding):
        ax = axes[i]
        label = COLS[i] if i < len(COLS) else f"Obs Dim {i}"
        sns.histplot(observations[:, i], ax=ax, kde=True, color="skyblue")
        ax.set_title(label)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

    # Hide unused subplots
    for i in range(num_non_embedding, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("results/histograms_obs_stats.png", dpi=200)
    plt.close()

    # 2. Plot histograms for actions and rewards
    print("Plotting action and reward histograms...")
    fig, axes = plt.subplots(1, n_action_dims + 1, figsize=(20, 5))

    for i in range(n_action_dims):
        sns.histplot(actions[:, i], ax=axes[i], kde=True, color="salmon")
        axes[i].set_title(f"Action {i}")
        axes[i].set_xlabel("Value")

    sns.histplot(rewards, ax=axes[n_action_dims], kde=True, color="green")
    axes[n_action_dims].set_title("Rewards")
    axes[n_action_dims].set_xlabel("Value")

    plt.tight_layout()
    plt.savefig("results/histograms_actions_rewards.png", dpi=200)
    plt.close()

    # 3. Plot a summary of embedding dimensions (mean, std, or range of distributions)
    print("Plotting embedding dimensions summary histogram...")
    # Instead of 768 plots, we plot the distribution of all values in the embedding space
    # and maybe the distribution of means per dimension
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    embedding_data = observations[:, -embedding_dim:]

    sns.histplot(embedding_data.flatten(), ax=axes[0], kde=True, color="purple")
    axes[0].set_title("Distribution of All Embedding Values")
    axes[0].set_xlabel("Value")

    dim_means = np.mean(embedding_data, axis=0)
    sns.histplot(dim_means, ax=axes[1], kde=True, color="indigo")
    axes[1].set_title("Distribution of Per-Dimension Means")
    axes[1].set_xlabel("Mean Value")

    plt.tight_layout()
    plt.savefig("results/histograms_embeddings_summary.png", dpi=200)
    plt.close()

    # 4. Investigate Embedding Variance
    print("Analyzing embedding variance...")
    embedding_data = observations[:, -embedding_dim:]
    variances = np.var(embedding_data, axis=0)

    # Identify zero-variance dimensions
    zero_variance_indices = np.where(variances < 1e-10)[0]
    print(
        f"Found {len(zero_variance_indices)} dimensions with near-zero variance (< 1e-10)"
    )
    if len(zero_variance_indices) > 0:
        print(f"Indices: {zero_variance_indices}")

    # Plot 16 dimensions with lowest variance
    sorted_indices = np.argsort(variances)
    low_variance_indices = sorted_indices[:16]

    print(f"Plotting 16 dimensions with lowest variance: {low_variance_indices}")
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()

    for i, idx in enumerate(low_variance_indices):
        ax = axes[i]
        sns.histplot(embedding_data[:, idx], ax=ax, kde=True, color="teal")
        ax.set_title(f"Embedding Dim {idx}\nVar: {variances[idx]:.2e}")
        ax.set_xlabel("Value")

    plt.tight_layout()
    plt.savefig("results/histograms_embeddings_low_variance.png", dpi=200)
    plt.close()

    # Plot a random sample of 16 embedding dimensions for comparison
    random_indices = np.random.choice(embedding_dim, 16, replace=False)
    print(f"Plotting 16 random embedding dimensions: {random_indices}")
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()

    for i, idx in enumerate(random_indices):
        ax = axes[i]
        sns.histplot(embedding_data[:, idx], ax=ax, kde=True, color="slateblue")
        ax.set_title(f"Embedding Dim {idx}\nVar: {variances[idx]:.2e}")
        ax.set_xlabel("Value")

    plt.tight_layout()
    plt.savefig("results/histograms_embeddings_random_sample.png", dpi=200)
    plt.close()

    print("\nEmbedding variance analysis plots saved to 'results/'.")
    print("\nAll plots saved to the 'results/' directory.")


if __name__ == "__main__":
    main()
