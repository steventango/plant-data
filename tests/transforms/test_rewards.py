import polars as pl
import pytest
from transforms.rewards import transform_reward


def test_transform_reward_per_plant():
    # Create data with two plants having different initial areas
    # Plant 0: initial area 10.0
    # Plant 1: initial area 20.0

    data = {
        "experiment": [13, 13, 13, 13, 13, 13],
        "zone": [1, 1, 1, 1, 1, 1],
        "plant_id": [0, 0, 0, 1, 1, 1],
        "time": [1, 2, 3, 1, 2, 3],
        "clean_area": [
            10.0,
            12.0,
            15.0,  # Plant 0
            20.0,
            24.0,
            30.0,  # Plant 1
        ],
        "bolted_pred": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    df = pl.DataFrame(data)

    result = transform_reward(df)

    # Expected rewards:
    # Plant 0:
    #   Step 1: null (no prev)
    #   Step 2: (12 - 10) / 10 = 0.2
    #   Step 3: (15 - 12) / 10 = 0.3
    # Plant 1:
    #   Step 1: null
    #   Step 2: (24 - 20) / 20 = 0.2
    #   Step 3: (30 - 24) / 20 = 0.3

    p0_rewards = result.filter(pl.col("plant_id") == 0)["reward"].to_list()
    p1_rewards = result.filter(pl.col("plant_id") == 1)["reward"].to_list()

    assert p0_rewards[0] is None
    assert pytest.approx(p0_rewards[1]) == 0.2
    assert pytest.approx(p0_rewards[2]) == 0.3

    assert p1_rewards[0] is None
    assert pytest.approx(p1_rewards[1]) == 0.2
    assert pytest.approx(p1_rewards[2]) == 0.3


def test_transform_reward_bolted():
    data = {
        "experiment": [13, 13],
        "zone": [1, 1],
        "plant_id": [0, 0],
        "time": [1, 2],
        "clean_area": [10.0, 12.0],
        "bolted_pred": [0.0, 0.6],  # > 0.5
    }
    df = pl.DataFrame(data)
    result = transform_reward(df)

    rewards = result["reward"].to_list()
    assert rewards[0] is None
    assert rewards[1] == 0.0
