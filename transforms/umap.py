import logging
import polars as pl
import numpy as np
from umap import UMAP
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


def transform_umap(
    df: pl.DataFrame, K: int = 2, output_path: Path = None, random_state: int = 42
) -> pl.DataFrame:
    """
    Fits UMAP on the cls_token column and adds cls_token_umap to the dataframe.
    Saves the UMAP model to output_path if provided.
    """
    if "cls_token" not in df.columns:
        logger.warning(
            "cls_token column not found in dataframe. Skipping UMAP transform."
        )
        return df

    logger.info(f"Fitting UMAP (K={K}) on cls_token embeddings...")

    embeddings_list = df["cls_token"].to_list()
    embeddings = np.array(embeddings_list)

    # Fit UMAP
    reducer = UMAP(n_components=K, random_state=random_state)
    umap_features = reducer.fit_transform(embeddings)

    # Add to dataframe
    df = df.with_columns(pl.Series(umap_features.tolist()).alias("cls_token_umap"))

    if output_path:
        logger.info(f"Saving UMAP model to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(reducer, output_path)

    return df
