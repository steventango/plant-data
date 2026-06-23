from typing import Any, Tuple

import gymnasium as gym
import numpy as np
import polars as pl
from gymnasium import spaces
from gymnasium.core import ObsType


class MockEnv(gym.Env):
    def __init__(
        self,
        df: pl.DataFrame,
        stats: dict,
        cols: list,
        action_cols: list | None = None,
    ):
        super().__init__()
        self.df = df.sort("experiment", "zone", "plant_id", "time")
        self.episode_keys = (
            df.select(["experiment", "zone", "plant_id"])
            .unique()
            .sort(["experiment", "zone", "plant_id"])
            .rows()
        )
        self.current_episode_index = 0
        self.current_episode_key = None
        self.current_row_index = 0
        self.plant_df = None
        self.was_truncated = False
        self.truncated_episode_key = None
        self.truncated_row_index = 0
        self.completed_episodes = set()  # Track completed episodes
        self.stats = stats
        self.cols = cols
        self.action_cols = action_cols  # None → default 3-dim [red_coef, white_coef, blue_coef]
        self.done = False
        self.embedding_dim = 768
        self.pca_dim = 10
        self.obs_dim = len(cols) + self.pca_dim + self.embedding_dim
        self.zero_obs = np.zeros((self.obs_dim,), dtype=np.float32)

        # Keep only episodes with at least 2 rows to avoid terminal reset returning
        # invalid observations when remaining episodes cannot produce transitions.
        episode_counts = (
            self.df.group_by(["experiment", "zone", "plant_id"])
            .len()
            .filter(pl.col("len") >= 2)
            .sort(["experiment", "zone", "plant_id"])
            .select(["experiment", "zone", "plant_id"])
        )
        self.episode_keys = episode_counts.rows()

        # Build per-feature fallback values from normalization stats.
        self.col_fill_values = {
            col: float(stats.get(col, {}).get("mean", 0.0)) for col in self.cols
        }

        # Ensure scalar feature columns never contain null/NaN at runtime.
        self.df = self.df.with_columns(
            [
                pl.col(col)
                .fill_null(self.col_fill_values[col])
                .fill_nan(self.col_fill_values[col])
                .alias(col)
                for col in self.cols
                if col in self.df.columns
            ]
        )
        col_lows = np.array(
            [stats.get(c, {}).get("min", -np.inf) for c in self.cols], dtype=np.float32
        )
        col_highs = np.array(
            [stats.get(c, {}).get("max", np.inf) for c in self.cols], dtype=np.float32
        )
        pca_stats = stats.get("cls_token_pca", {})
        pca_lows = np.array(
            pca_stats.get("min", [-np.inf] * self.pca_dim), dtype=np.float32
        )
        pca_highs = np.array(
            pca_stats.get("max", [np.inf] * self.pca_dim), dtype=np.float32
        )
        emb_stats = stats.get("cls_token", {})
        emb_lows = np.array(
            emb_stats.get("min", [-np.inf] * self.embedding_dim), dtype=np.float32
        )
        emb_highs = np.array(
            emb_stats.get("max", [np.inf] * self.embedding_dim), dtype=np.float32
        )
        obs_low = np.concatenate([col_lows, pca_lows, emb_lows])
        obs_high = np.concatenate([col_highs, pca_highs, emb_highs])
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )
        if self.action_cols is not None:
            # Parameterized action: bounds from normalization stats (data min/max)
            n_act = len(self.action_cols)
            lows = np.array(
                [stats.get(c, {}).get("min", 0.0) for c in self.action_cols],
                dtype=np.float32,
            )
            highs = np.array(
                [stats.get(c, {}).get("max", np.inf) for c in self.action_cols],
                dtype=np.float32,
            )
            self.action_space = spaces.Box(low=lows, high=highs, dtype=np.float32)
        else:
            # Default action space: [red_coef, white_coef, blue_coef]
            self.action_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

    def _get_observation(self) -> Any:
        if self.plant_df is None or self.current_row_index >= self.plant_df.height:
            return self.zero_obs

        row = self.plant_df.slice(self.current_row_index, 1)
        if row.is_empty():
            return self.zero_obs

        # Get stats
        stats = row[self.cols].to_numpy().flatten()

        # Get PCA features
        cls_token_pca = row["cls_token_pca"][0]

        # Get embedding
        cls_token = row[["cls_token"]].to_numpy().flatten()[0]

        # Some upstream vision features can still contain NaNs for edge cases.
        stats = np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
        cls_token_pca = np.nan_to_num(
            np.asarray(cls_token_pca), nan=0.0, posinf=0.0, neginf=0.0
        )
        cls_token = np.nan_to_num(
            np.asarray(cls_token), nan=0.0, posinf=0.0, neginf=0.0
        )

        obs = np.concatenate([stats, cls_token_pca, cls_token], dtype=np.float32)

        if np.isnan(obs).any():
            nan_idx = np.where(np.isnan(obs.flatten()))[0]
            cols_with_nans = " ".join(
                [
                    self.cols[i] if i < len(self.cols) else f"obs_idx_{i}"
                    for i in nan_idx
                ]
            )
            raise ValueError(f"NaN found in observation: {cols_with_nans}")

        return obs

    def _get_action(self) -> int | np.ndarray:
        if self.action_cols is not None:
            # Parameterized action: read the requested columns
            n_act = len(self.action_cols)
            zeros = np.zeros(n_act, dtype=np.float32)
            if self.plant_df is None or self.current_row_index >= self.plant_df.height:
                return zeros
            row = self.plant_df.slice(self.current_row_index, 1)
            if row.is_empty():
                return zeros
            vals = []
            for col in self.action_cols:
                v = row[col][0] if col in row.columns and row[col][0] is not None else 0.0
                vals.append(float(v))
            return np.array(vals, dtype=np.float32)

        # Default: [red_coef, white_coef, blue_coef]
        if self.plant_df is None or self.current_row_index >= self.plant_df.height:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        row = self.plant_df.slice(self.current_row_index, 1)
        if row.is_empty():
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        red_coef = row["red_coef"][0] if row["red_coef"][0] is not None else 0.0
        white_coef = row["white_coef"][0] if row["white_coef"][0] is not None else 0.0
        blue_coef = row["blue_coef"][0] if row["blue_coef"][0] is not None else 0.0
        return np.array([red_coef, white_coef, blue_coef], dtype=np.float32)

    def _get_info(self) -> dict[str, Any]:
        info = {}

        if self.plant_df is None or self.current_row_index >= self.plant_df.height:
            return info

        row = self.plant_df.slice(self.current_row_index, 1)
        if row.is_empty():
            return info

        if "image_path" in row.columns:
            image_path = row["image_path"][0]
            if image_path is not None:
                info["image_path"] = image_path

        info["experiment"] = self.current_episode_key[0]
        info["zone"] = self.current_episode_key[1]
        info["plant_id"] = self.current_episode_key[2]

        return info

    def is_done(self) -> bool:
        """Check if all episodes have been completed"""
        return len(self.completed_episodes) >= len(self.episode_keys) or self.done

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:  # type: ignore
        super().reset(seed=seed)

        episode_continued = False
        # If we were truncated, continue from where we left off
        if self.was_truncated and self.truncated_episode_key is not None:
            self.current_episode_key = self.truncated_episode_key
            # truncated_row_index was saved after the +1 increment in step(), so it
            # already points to the first row of the continuation (e.g. day 3).
            self.current_row_index = self.truncated_row_index
            self.was_truncated = False
            self.truncated_episode_key = None
            self.truncated_row_index = 0
            # Only continue if there are enough rows left to form at least one transition
            remaining = self.plant_df.height - self.current_row_index
            episode_continued = remaining >= 2
        if not episode_continued:
            # Select the next episode (cycle through all unique experiment-zone-plant combinations)
            # Skip episodes that are already completed
            while self.current_episode_index < len(self.episode_keys):
                candidate_key = self.episode_keys[self.current_episode_index]
                self.current_episode_index += 1

                # ensure rows > 1:
                if (
                    self.df.filter(
                        (pl.col("experiment") == candidate_key[0])
                        & (pl.col("zone") == candidate_key[1])
                        & (pl.col("plant_id") == candidate_key[2])
                    ).height
                    < 2
                ):
                    continue

                if candidate_key not in self.completed_episodes:
                    self.current_episode_key = candidate_key
                    break
            else:
                # All episodes completed, return None to indicate done
                self.done = True
                return self.zero_obs, {"done": True}

            # Get all rows for this episode
            self.plant_df = self.df.filter(
                (pl.col("experiment") == self.current_episode_key[0])
                & (pl.col("zone") == self.current_episode_key[1])
                & (pl.col("plant_id") == self.current_episode_key[2])
            ).sort("time")
            self.current_row_index = 0

        obs = self._get_observation()
        self.action = self._get_action()
        info = self._get_info()
        return obs, info

    def step(self, action: int | np.ndarray) -> Tuple[Any, float, bool, bool, dict]:
        row = self.plant_df.slice(self.current_row_index, 1)
        terminal = bool(row["terminal"][0]) if row["terminal"][0] is not None else False
        truncated = (
            bool(row["truncated"][0]) if row["truncated"][0] is not None else False
        )

        self.current_row_index += 1
        row = self.plant_df.slice(self.current_row_index, 1)

        reward = float(row["reward"][0]) if row["reward"][0] is not None else 0.0

        # Check if we've reached the end of this plant's data
        end_of_df = self.current_row_index == self.plant_df.height - 1
        if end_of_df or terminal:
            self.completed_episodes.add(self.current_episode_key)
        if end_of_df and not terminal:
            truncated = True

        # If truncated, save state to continue from this point in next reset
        if truncated and not terminal:
            self.was_truncated = True
            self.truncated_episode_key = self.current_episode_key
            self.truncated_row_index = self.current_row_index

        obs = self._get_observation()
        self.action = self._get_action()
        info = self._get_info()

        return obs, reward, terminal, truncated, info
