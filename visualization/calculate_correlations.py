import minari
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COLS


def main():
    # Load dataset
    dataset_name = "plant-data/mixed-v23"
    print(f"Loading dataset: {dataset_name}")
    try:
        offline_dataset = minari.load_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    all_obs = []

    print("Collecting data from episodes...")
    for episode in offline_dataset.iterate_episodes():
        all_obs.append(episode.observations[:-1])

    observations = np.concatenate(all_obs, axis=0)
    print(f"Observations shape: {observations.shape}")

    # Set seaborn style
    sns.set_theme(style="white")
    os.makedirs("results", exist_ok=True)

    embedding_dim = 768
    num_non_embedding = observations.shape[1] - embedding_dim

    # 1. Non-embedding Correlation
    print("Computing correlation for non-embedding features...")
    non_embedding_data = observations[:, :num_non_embedding]

    column_names = [
        COLS[i] if i < len(COLS) else f"Obs_{i}" for i in range(num_non_embedding)
    ]
    df_non_embedding = pd.DataFrame(non_embedding_data, columns=column_names)

    corr_matrix = df_non_embedding.corr()

    # Plot heatmap
    plt.figure(figsize=(15, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0
    )
    plt.title("Correlation Matrix - Non-Embedding State Dimensions")
    plt.tight_layout()
    plt.savefig("results/correlation_matrix_stats.png", dpi=150)
    plt.close()

    # Find high correlations
    high_corr_threshold = 0.9
    print(f"\nHighly correlated non-embedding pairs (abs > {high_corr_threshold}):")
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = [
        (column, row)
        for row in upper_tri.index
        for column in upper_tri.columns
        if abs(upper_tri.loc[row, column]) > high_corr_threshold
    ]

    for col1, col2 in high_corr_pairs:
        coeff = corr_matrix.loc[col1, col2]
        print(f"- {col1} and {col2}: {coeff:.4f}")

    # 2. Embedding Correlation (Sample/Summary)
    print("\nComputing correlation for embedding dimensions...")
    embedding_data = observations[:, -embedding_dim:]

    # Since 768x768 is large, let's look at a sample of 100 dimensions first
    sample_indices = np.linspace(0, embedding_dim - 1, 100, dtype=int)
    df_embedding_sample = pd.DataFrame(
        embedding_data[:, sample_indices],
        columns=[f"Embed_{i}" for i in sample_indices],
    )
    corr_matrix_emb = df_embedding_sample.corr()

    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_matrix_emb, cmap="coolwarm", center=0)
    plt.title("Correlation Matrix - Embedding Dimensions (Sample of 100)")
    plt.tight_layout()
    plt.savefig("results/correlation_matrix_embeddings_sample.png", dpi=150)
    plt.close()

    # Find top correlated pairs in ALL embeddings
    print("Finding top correlated embedding pairs (this may take a moment)...")
    # Reduced precision or subset if needed, but 768 isn't too crazy for pandas
    df_all_emb = pd.DataFrame(embedding_data)
    all_corr = df_all_emb.corr()

    # Get upper triangle without diagonal
    all_upper = all_corr.where(np.triu(np.ones(all_corr.shape), k=1).astype(bool))

    # Stack and find largest
    stacked_corr = all_upper.stack()
    top_10_pairs = stacked_corr.abs().sort_values(ascending=False).head(10)

    print("\nTop 10 most correlated embedding dimension pairs:")
    for (idx1, idx2), val in top_10_pairs.items():
        orig_val = all_corr.iloc[idx1, idx2]
        print(f"- Dim {idx1} and Dim {idx2}: {orig_val:.4f}")

    print("\nCorrelation analysis plots and summary saved to 'results/'.")


if __name__ == "__main__":
    main()
