import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import polars as pl
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

BOLTED_URL = "http://localhost:8804"
thread_local = threading.local()


def get_session() -> requests.Session:
    """Get a thread-local requests session."""
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session


def predict_bolted(cls_token: list[float]) -> float:
    """Predict bolted probability for a single embedding."""
    if cls_token is None:
        return None

    try:
        if hasattr(cls_token, "tolist"):
            cls_token = cls_token.tolist()

        response = get_session().post(
            f"{BOLTED_URL}/predict",
            json={"embedding": cls_token},
            timeout=5,
        )
        response.raise_for_status()
        return response.json().get("bolted_probability", None)
    except Exception as e:
        logger.warning(f"Bolted prediction failed: {e}")
        return None


def transform_bolted(df: pl.DataFrame) -> pl.DataFrame:
    if "cls_token" not in df.columns:
        logger.warning("cls_token column missing, skipping bolted prediction")
        return df.with_columns(pl.lit(None).alias("bolted_pred"))

    cls_tokens = df["cls_token"].to_list()

    predictions = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        predictions = list(
            tqdm(
                executor.map(predict_bolted, cls_tokens),
                total=len(cls_tokens),
                desc="Predicting Bolted",
                leave=False,
            )
        )

    df = df.with_columns(pl.Series("bolted_prob", predictions, dtype=pl.Float32))
    df = df.with_columns(
        ((pl.col("bolted_prob") > 0.5) & (pl.col("wall_time") >= 7.0)).alias(
            "bolted_pred"
        )
    )
    return df


def transform_terminal(
    df: pl.DataFrame, terminal_day: int | None = None
) -> pl.DataFrame:
    df = transform_bolted(df)
    df = df.with_columns(pl.col("bolted_pred").alias("terminal"))
    if terminal_day is not None:
        df = df.with_columns(
            (pl.col("terminal") | (pl.col("day") >= terminal_day)).alias("terminal")
        )
    return df
