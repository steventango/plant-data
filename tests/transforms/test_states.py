from datetime import date, datetime, timedelta

import polars as pl

from transforms.states import transform_days_since_events, transform_state


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


class TestTransformState:
    """Tests for the transform_state function."""

    def test_mean_clean_area_computed(self):
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
                "area": [100.0, 200.0, 150.0, 250.0],
            }
        )
        result = transform_state(df)

        assert "mean_clean_area" in result.columns
        assert "clean_area" in result.columns
