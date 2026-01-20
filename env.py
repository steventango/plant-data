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
        self.done = False
        self.embedding_dim = 768

        self.low = np.array(
            [stats[col]["min"] for col in self.cols] + [-np.inf] * self.embedding_dim
        ).astype(np.float32)
        self.high = np.array(
            [stats[col]["max"] for col in self.cols] + [np.inf] * self.embedding_dim
        ).astype(np.float32)
        self.observation_space = spaces.Box(
            low=self.low, high=self.high, dtype=np.float32
        )
        # Action space: [red_coef, white_coef, blue_coef]
        self.action_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

    def _get_observation(self) -> Any:
        if self.plant_df is None or self.current_row_index >= self.plant_df.height:
            return np.zeros(
                (len(self.cols) + 3 + self.embedding_dim,), dtype=np.float32
            )

        row = self.plant_df.slice(self.current_row_index, 1)
        if row.is_empty():
            return np.zeros(
                (len(self.cols) + 3 + self.embedding_dim,), dtype=np.float32
            )

        # Get stats
        stats = row[self.cols].to_numpy().flatten()

        # Get embedding
        cls_token = row[["cls_token"]].to_numpy().flatten()[0]

        obs = np.concatenate([stats, cls_token], dtype=np.float32)
        return obs

    def _get_action(self) -> int | np.ndarray:
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
        info = {"action": self._get_action()}

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
            self.current_row_index = self.truncated_row_index + 1
            self.was_truncated = False
            self.truncated_episode_key = None
            self.truncated_row_index = 0
            if len(self.plant_df) < 2:
                episode_continued = False
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
                return None, {"done": True}

            # Get all rows for this episode
            self.plant_df = self.df.filter(
                (pl.col("experiment") == self.current_episode_key[0])
                & (pl.col("zone") == self.current_episode_key[1])
                & (pl.col("plant_id") == self.current_episode_key[2])
            ).sort("time")
            self.current_row_index = 0

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: int | np.ndarray) -> Tuple[Any, float, bool, bool, dict]:
        self.current_row_index += 1
        row = self.plant_df.slice(self.current_row_index, 1)

        reward = float(row["reward"][0]) if row["reward"][0] is not None else 0.0
        terminal = bool(row["terminal"][0]) if row["terminal"][0] is not None else False
        truncated = (
            bool(row["truncated"][0]) if row["truncated"][0] is not None else False
        )

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
        info = self._get_info()

        return obs, reward, terminal, truncated, info
