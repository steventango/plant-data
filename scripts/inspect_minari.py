import minari
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

offline_dataset = minari.load_dataset("plant-data/mixed-v23")
episode_index = 1200

print(f"Observation space: {offline_dataset.observation_space.shape}")
print(f"Action space: {offline_dataset.action_space.shape}")

all_obs = []
all_actions = []
all_rewards = []
all_next_obs = []
all_terminations = []

for episode in offline_dataset.iterate_episodes():
    episode_length = len(episode.observations) - 1
    for t in range(episode_length):
        observations = episode.observations[t]
        # if nan or inf raise
        if np.isnan(observations).any() or np.isinf(observations).any():
            raise ValueError("NaN or inf found in observations")
        all_obs.append(observations)

        actions = episode.actions[t]
        # if nan or inf raise
        if np.isnan(actions).any() or np.isinf(actions).any():
            raise ValueError("NaN or inf found in actions")
        all_actions.append(actions)

        rewards = episode.rewards[t]
        if np.isnan(rewards).any() or np.isinf(rewards).any():
            print(rewards)
            raise ValueError("NaN or inf found in rewards")
        all_rewards.append(rewards)

        next_observations = episode.observations[t + 1]
        # if nan or inf raise
        if np.isnan(next_observations).any() or np.isinf(next_observations).any():
            raise ValueError("NaN or inf found in next_observations")
        all_next_obs.append(next_observations)

        terminations = episode.terminations[t]
        if np.isnan(terminations).any() or np.isinf(terminations).any():
            raise ValueError("NaN or inf found in terminations")
        all_terminations.append(terminations)

# numpy stack
observations = np.stack(all_obs)
print(f"Observations shape: {observations.shape}")
actions = np.stack(all_actions)
print(f"Actions shape: {actions.shape}")
rewards = np.stack(all_rewards)
print(f"Rewards shape: {rewards.shape}")
next_observations = np.stack(all_next_obs)
print(f"Next observations shape: {next_observations.shape}")
terminations = np.stack(all_terminations)
print(f"Terminations shape: {terminations.shape}")

episode = offline_dataset[episode_index]
print(episode)
for key, value in episode.infos.items():
    print(key, value[0])

# Extract data from the first episode
episode_length = len(episode.observations) - 1
timesteps = np.arange(episode_length)

# Prepare data for plotting
observations = episode.observations[:-1]  # Exclude last observation
actions = episode.actions
rewards = episode.rewards
terminations = episode.terminations
truncations = episode.truncations
infos = episode.infos

# Set seaborn style
sns.set_theme(style="darkgrid")

# Create figure with subplots
n_obs_dims = observations.shape[1]
n_action_dims = actions.shape[1]

fig, axes = plt.subplots(5, 3, figsize=(18, 15))
axes = axes.flatten()

# Plot 0: Wall time (dimension 0)
ax0 = axes[0]
ax0.plot(
    timesteps,
    observations[:, 0],
    color="purple",
    linewidth=2,
    marker="o",
    markersize=4,
    drawstyle="steps-post",
)
ax0.set_xlabel("Timestep")
ax0.set_ylabel("Wall Time")
ax0.set_title("Observation Dimension 0: Wall Time")
ax0.grid(True, alpha=0.3)
ax0.grid(True, which="major", axis="y", alpha=0.5, linestyle="-", linewidth=0.8)
ax0.yaxis.set_major_locator(plt.MultipleLocator(1))

# Plot 1: Clean area (dimension 1)
ax1 = axes[1]
ax1.plot(
    timesteps,
    observations[:, 1],
    color="orange",
    linewidth=2,
    marker="o",
    markersize=4,
    drawstyle="steps-post",
)
ax1.set_xlabel("Timestep")
ax1.set_ylabel("Clean Area")
ax1.set_title("Observation Dimension 1: Clean Area")
ax1.grid(True, alpha=0.3)

# Plot 2: Solidity (dimension 3)
ax2 = axes[2]
ax2.plot(
    timesteps,
    observations[:, 3],
    color="brown",
    linewidth=2,
    marker="o",
    markersize=4,
    drawstyle="steps-post",
)
ax2.set_xlabel("Timestep")
ax2.set_ylabel("Solidity")
ax2.set_title("Observation Dimension 3: Solidity")
ax2.grid(True, alpha=0.3)

# Plot 3: Log Clean Area (dimension 28)
ax3 = axes[3]
ax3.plot(
    timesteps,
    observations[:, 28],
    color="green",
    linewidth=2,
    marker="o",
    markersize=4,
    drawstyle="steps-post",
)
ax3.set_xlabel("Timestep")
ax3.set_ylabel("Log Clean Area")
ax3.set_title("Observation Dimension 28: Log Clean Area")
ax3.grid(True, alpha=0.3)

# Plot 4: Liters Per Pot (dimension 33)
ax4 = axes[4]
ax4.plot(
    timesteps,
    observations[:, 33],
    color="cyan",
    linewidth=2,
    marker="o",
    markersize=4,
    drawstyle="steps-post",
)
ax4.set_xlabel("Timestep")
ax4.set_ylabel("Liters Per Pot")
ax4.set_title("Observation Dimension 33: Liters Per Pot")
ax4.grid(True, alpha=0.3)

# Plot 5: LAB Colors (dimensions 24, 19, 17)
ax5 = axes[5]
lab_indices = [24, 19, 17]  # L*, a*, b*
lab_colors = ["black", "magenta", "blue"]
lab_labels = ["L (Lightness)", "a (Green-Red)", "b (Blue-Yellow)"]

for idx, color, label in zip(lab_indices, lab_colors, lab_labels):
    ax5.plot(
        timesteps,
        observations[:, idx],
        color=color,
        label=label,
        alpha=0.7,
        linewidth=2,
        drawstyle="steps-post",
    )
ax5.set_xlabel("Timestep")
ax5.set_ylabel("LAB Values")
ax5.set_title("Observation Dimensions 24, 19, 17: LAB Color Mean")
ax5.legend(loc="lower right", fontsize="small")
ax5.grid(True, alpha=0.3)

# Plot 6: Other plant stats
ax6 = axes[6]
action_trace_start = 35  # first action trace index in COLS
excluded_dims = {0, 1, 3, 17, 19, 24, 28, 33}
for i in range(38):  # length of COLS
    if i in excluded_dims:
        continue
    ax6.plot(
        timesteps,
        observations[:, i],
        label=f"Obs {i}",
        alpha=0.7,
        linewidth=1.5,
        drawstyle="steps-post",
    )
ax6.set_xlabel("Timestep")
ax6.set_ylabel("Plant Stats Value")
ax6.set_title("Other Observation Dimensions (0-37)")
ax6.grid(True, alpha=0.3)

# Plot 7: Action traces (dimensions 35-37)
ax7 = axes[7]
colors_traces = ["red", "grey", "blue"]
labels_traces = [
    "Action Trace 0 (Red)",
    "Action Trace 1 (White)",
    "Action Trace 2 (Blue)",
]
for i, (color, label) in enumerate(zip(colors_traces, labels_traces)):
    ax7.plot(
        timesteps,
        observations[:, 35 + i],
        color=color,
        label=label,
        linewidth=2,
        marker="o",
        markersize=4,
        drawstyle="steps-post",
    )
ax7.set_xlabel("Timestep")
ax7.set_ylabel("Action Trace Value")
ax7.set_title("Observation Dimensions 35-37: Action Traces")
ax7.legend(loc="upper right")
ax7.grid(True, alpha=0.3)

# Plot 8: PCA Embeddings [:4] (dimensions 38-41)
ax8 = axes[8]
pca_indices = range(38, 42)
pca_colors = plt.cm.viridis(np.linspace(0, 1, 4))
for i, color in zip(pca_indices, pca_colors):
    ax8.plot(
        timesteps,
        observations[:, i],
        color=color,
        label=f"PCA {i - 38}",
        linewidth=2,
        drawstyle="steps-post",
    )
ax8.set_xlabel("Timestep")
ax8.set_ylabel("PCA Value")
ax8.set_title("Observation Dimensions 38-41: cls_token_pca[:4]")
ax8.legend(loc="upper right", fontsize="x-small")
ax8.grid(True, alpha=0.3)

# Plot 9: All Embeddings (last 768 dimensions)
ax9 = axes[9]
embedding_start = n_obs_dims - 768
for i in range(embedding_start, n_obs_dims):
    ax9.plot(
        timesteps, observations[:, i], alpha=0.1, linewidth=0.5, drawstyle="steps-post"
    )
ax9.set_xlabel("Timestep")
ax9.set_ylabel("Embedding Value")
ax9.set_title(f"Dimensions {embedding_start}-{n_obs_dims - 1}: Embeddings")
ax9.grid(True, alpha=0.3)

# Plot 10: Action dimensions as area plot (step function)
ax10 = axes[10]
colors = ["red", "grey", "blue"]
labels = ["Action 0", "Action 1", "Action 2"]
ax10.stackplot(
    timesteps,
    actions[:, 0],
    actions[:, 1],
    actions[:, 2],
    colors=colors,
    labels=labels,
    alpha=0.7,
    step="post",
)
ax10.set_xlabel("Timestep")
ax10.set_ylabel("Action Value")
ax10.set_title("Action Dimensions Over Time")
ax10.legend(loc="upper right")
ax10.grid(True, alpha=0.3)

# Plot 11: Rewards
ax11 = axes[11]
ax11.plot(
    timesteps,
    rewards,
    color="green",
    linewidth=2,
    marker="o",
    markersize=4,
    drawstyle="steps-post",
)
ax11.fill_between(timesteps, rewards, alpha=0.3, color="green", step="post")
ax11.set_xlabel("Timestep")
ax11.set_ylabel("Reward")
ax11.set_title("Rewards Over Time")
ax11.grid(True, alpha=0.3)

# Plot 12: Terminations
ax12 = axes[12]
ax12.plot(
    timesteps,
    terminations,
    color="red",
    linewidth=2,
    marker="s",
    markersize=5,
    drawstyle="steps-post",
)
ax12.fill_between(timesteps, terminations, alpha=0.3, color="red", step="post")
ax12.set_xlabel("Timestep")
ax12.set_ylabel("Termination")
ax12.set_title("Terminations Over Time")
ax12.set_ylim(-0.1, 1.1)
ax12.grid(True, alpha=0.3)

# Plot 13: Truncations
ax13 = axes[13]
ax13.plot(
    timesteps,
    truncations,
    color="darkorange",
    linewidth=2,
    marker="d",
    markersize=5,
    drawstyle="steps-post",
)
ax13.fill_between(timesteps, truncations, alpha=0.3, color="darkorange", step="post")
ax13.set_xlabel("Timestep")
ax13.set_ylabel("Truncation")
ax13.set_title("Truncations Over Time")
ax13.set_ylim(-0.1, 1.1)
ax13.grid(True, alpha=0.3)

# Hide unused axes
for i in range(14, 15):
    axes[i].axis("off")

plt.tight_layout()
path = f"results/episode_{episode_index}_analysis.png"
plt.savefig(path, dpi=300, bbox_inches="tight")
print(f"\nPlot saved to {path}")
print(f"Episode length: {episode_length}")
print(f"Number of observation dimensions: {n_obs_dims}")
print(f"Number of action dimensions: {n_action_dims}")
print(f"Total reward: {np.sum(rewards):.4f}")
plt.show()
