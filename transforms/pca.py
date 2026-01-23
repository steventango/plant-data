import logging
import polars as pl
import numpy as np
from sklearn.decomposition import PCA
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


def transform_pca(
    df: pl.DataFrame, K: int = 10, output_path: Path = None
) -> pl.DataFrame:
    """
    Fits PCA on the cls_token column and adds pca_features to the dataframe.
    Saves the PCA model to output_path if provided.
    """
    if "cls_token" not in df.columns:
        logger.warning(
            "cls_token column not found in dataframe. Skipping PCA transform."
        )
        return df

    logger.info(f"Fitting PCA (K={K}) on cls_token embeddings...")

    embeddings_list = df["cls_token"].to_list()
    embeddings = np.array(embeddings_list)

    pca = PCA(n_components=K, random_state=0)
    pca_features = pca.fit_transform(embeddings)

    df = df.with_columns(pl.Series(pca_features.tolist()).alias("cls_token_pca"))

    if output_path:
        logger.info(f"Saving PCA model to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pca, output_path)

    return df
