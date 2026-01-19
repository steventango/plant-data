from datetime import date, datetime, timedelta

import polars as pl

from transforms.states import (
    transform_days_since_events,
    transform_state,
    transform_watering_features,
)


def test_mean_clean_area_computed():
    """Test that mean_clean_area is computed correctly."""
    base_time = datetime(2024, 1, 1, 9, 30)
    df = pl.DataFrame(
        {
            "experiment": [1, 1, 1, 1],
            "zone": [1, 1, 1, 1],
            "time": [
                base_time,
                base_time,  # Same time
                base_time + timedelta(minutes=5),
                base_time + timedelta(minutes=5),  # Same time
            ],
            "clean_area": [100.0, 200.0, 150.0, 250.0],
        }
    )
    result = transform_state(df)

    assert result["mean_clean_area"][0] == 150.0
    assert result["mean_clean_area"][1] == 150.0
    assert result["mean_clean_area"][2] == 200.0
    assert result["mean_clean_area"][3] == 200.0


def test_transform_days_since_events():
    # Setup mock dataframe with event dates
    df = pl.DataFrame(
        {
            "time": [datetime(2025, 5, 25, 12, 0, 0)],
            "sterilized_date": [date(2025, 5, 12)],
            "plate_date": [date(2025, 5, 15)],
            "transplant_date": [date(2025, 5, 23)],
            "remove_domes_date": [date(2025, 5, 26)],
        }
    )

    df_out = transform_days_since_events(df)

    # Check if columns exist
    assert "days_since_sterilization" in df_out.columns
    assert "days_since_plate" in df_out.columns
    assert "days_since_transplant" in df_out.columns
    assert "days_since_dome_removal" in df_out.columns

    # Check calculations
    # 2025-05-25 - 2025-05-12 = 13 days
    assert df_out["days_since_sterilization"][0] == 13
    # 2025-05-25 - 2025-05-15 = 10 days
    assert df_out["days_since_plate"][0] == 10
    # 2025-05-25 - 2025-05-23 = 2 days
    assert df_out["days_since_transplant"][0] == 2
    # 2025-05-25 - 2025-05-26 = -1 days (future event relative to time)
    assert df_out["days_since_dome_removal"][0] == -1


def make_times(n: int, start: datetime = None) -> list[datetime]:
    """Create a list of datetime values spaced 5 minutes apart."""
    if start is None:
        start = datetime(2024, 1, 1, 9, 30)
    return [start + timedelta(minutes=i * 5) for i in range(n)]


def test_transform_watering_features():
    # Setup mock dataframe
    # We need 'experiment' and 'time' columns
    df = pl.DataFrame(
        {
            "experiment": [9, 9, 9, 9],
            "time": [
                datetime(2025, 5, 22, 12, 0, 0),  # Before first watering (May 23)
                datetime(2025, 5, 23, 12, 0, 0),  # Day of first watering
                datetime(2025, 5, 25, 12, 0, 0),  # Between waterings
                datetime(2025, 5, 27, 12, 0, 0),  # After second watering (May 26)
            ],
        }
    )

    # E9 watering dates: May 23 (1.0L), May 26 (2.0L)
    # E9 num_pots_per_tray: 18
    # liters_per_pot: 1.0/18 ~= 0.055, 2.0/18 ~= 0.111

    df_out = transform_watering_features(df)

    # Check if columns exist
    assert "days_since_watering" in df_out.columns
    assert "liters_per_pot" in df_out.columns

    # Row 0: Before any watering
    assert df_out["days_since_watering"][0] is None
    assert df_out["liters_per_pot"][0] is None

    # Row 1: May 23 (Day 0 of first watering)
    assert df_out["days_since_watering"][1] == 0
    assert df_out["liters_per_pot"][1] == 1.0 / 18

    # Row 2: May 25 (2 days since May 23)
    assert df_out["days_since_watering"][2] == 2
    assert df_out["liters_per_pot"][2] == 1.0 / 18

    # Row 3: May 27 (1 day since May 26)
    assert df_out["days_since_watering"][3] == 1
    assert df_out["liters_per_pot"][3] == 2.0 / 18


def test_transform_watering_features_no_metadata():
    # Test with an experiment not in EXPERIMENT_EVENTS or missing metadata
    df = pl.DataFrame({"experiment": [999], "time": [datetime(2025, 1, 1)]})
    df_out = transform_watering_features(df)
    assert df_out["days_since_watering"][0] is None
    assert df_out["liters_per_pot"][0] is None
