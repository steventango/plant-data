"""Tests for transforms/states.py clean area calculation."""

from datetime import datetime, timedelta

import polars as pl
import pytest
from transforms.states import (
    apply_tukey_outlier_detection,
    compute_clean_area_for_group,
    transform_area,
    transform_state,
    CLEAN_AREA_LOWER_THRESHOLD,
    CLEAN_AREA_UPPER_THRESHOLD,
    MINIMUM_AREA_COUNT,
    TUKEY_K,
)


def make_times(n: int, start: datetime = None) -> list[datetime]:
    """Create a list of datetime values spaced 5 minutes apart."""
    if start is None:
        start = datetime(2024, 1, 1, 9, 30)
    return [start + timedelta(minutes=i * 5) for i in range(n)]


class TestComputeCleanAreaForGroup:
    """Tests for the compute_clean_area_for_group function."""

    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        df = pl.DataFrame(
            {
                "time": pl.Series([], dtype=pl.Datetime),
                "area": pl.Series([], dtype=pl.Float64),
            }
        )
        result = compute_clean_area_for_group(df)
        assert "clean_area" in result.columns
        assert len(result) == 0

    def test_single_observation(self):
        """Test with a single observation."""
        df = pl.DataFrame(
            {
                "time": make_times(1),
                "area": [100.0],
            }
        )
        result = compute_clean_area_for_group(df)
        assert result["clean_area"][0] == 100.0
        assert result["uema_area"][0] == 100.0

    def test_normal_sequence_no_outliers(self):
        """Test with a normal sequence that has no outliers."""
        # Gradually increasing areas (should all be accepted)
        df = pl.DataFrame(
            {
                "time": make_times(10),
                "area": [100.0 + i * 5 for i in range(10)],  # 100, 105, 110, ...
            }
        )
        result = compute_clean_area_for_group(df)
        # All areas should be accepted as clean
        for i in range(10):
            assert result["clean_area"][i] == 100.0 + i * 5

    def test_outlier_too_large(self):
        """Test detection of an outlier that is too large (masked pot instead of plant)."""
        # Start with consistent values, then spike
        areas = [100.0] * 5 + [500.0] + [105.0] * 4  # Spike at index 5
        df = pl.DataFrame(
            {
                "time": make_times(10),
                "area": areas,
            }
        )
        result = compute_clean_area_for_group(df)

        # The spike at index 5 should be replaced with the previous clean area
        # Since threshold is 0.5, upper bound is 1.5 * ewm_mean
        # EWM of ~100 would reject 500 as too large
        assert result["clean_area"][5] == 100.0  # Should be replaced with prev clean

    def test_outlier_too_small(self):
        """Test detection of an outlier that is too small (failed detection)."""
        # Start with consistent values, then drop
        areas = [100.0] * 5 + [10.0] + [105.0] * 4  # Drop at index 5
        df = pl.DataFrame(
            {
                "time": make_times(10),
                "area": areas,
            }
        )
        result = compute_clean_area_for_group(df)

        # The drop at index 5 should be replaced with the previous clean area
        # Since threshold is 0.5, lower bound is 0.5 * ewm_mean
        # EWM of ~100 would reject 10 as too small
        assert result["clean_area"][5] == 100.0  # Should be replaced with prev clean

    def test_minimum_count_before_detection(self):
        """Test that outlier detection only kicks in after MINIMUM_AREA_COUNT observations."""
        # With MINIMUM_AREA_COUNT=1, outlier detection starts after first observation
        # First observation is accepted as-is (no history to compare against)
        areas = [500.0] + [100.0] * 9
        df = pl.DataFrame(
            {
                "time": make_times(10),
                "area": areas,
            }
        )
        result = compute_clean_area_for_group(df)

        # First observation should always be accepted (no history to compare)
        assert result["clean_area"][0] == 500.0
        # With MINIMUM_AREA_COUNT=1, subsequent observations are compared against EWM
        # 100.0 is way below 0.5 * 500 = 250, so it would be rejected
        # But since 100.0 < 0.5 * ewm (which is ~500), it's an outlier
        # So it gets replaced with prev_clean_area (500.0)
        assert result["clean_area"][1] == 500.0

    def test_null_areas_handled(self):
        """Test that null areas are handled gracefully."""
        areas = [100.0, None, 105.0, None, 110.0]
        df = pl.DataFrame(
            {
                "time": make_times(5),
                "area": areas,
            }
        )
        result = compute_clean_area_for_group(df)

        # Null areas should use previous clean area
        assert result["clean_area"][0] == 100.0
        assert result["clean_area"][1] == 100.0  # Previous clean
        assert result["clean_area"][2] == 105.0
        assert result["clean_area"][3] == 105.0  # Previous clean
        assert result["clean_area"][4] == 110.0

    def test_zero_areas_treated_as_invalid(self):
        """Test that zero areas are treated as invalid (failed detection)."""
        areas = [100.0, 0.0, 105.0]
        df = pl.DataFrame(
            {
                "time": make_times(3),
                "area": areas,
            }
        )
        result = compute_clean_area_for_group(df)

        assert result["clean_area"][0] == 100.0
        assert result["clean_area"][1] == 100.0  # Zero replaced with previous
        assert result["clean_area"][2] == 105.0

    def test_uema_area_calculated(self):
        """Test that uema_area (EWM) is calculated and returned."""
        df = pl.DataFrame(
            {
                "time": make_times(5),
                "area": [100.0] * 5,
            }
        )
        result = compute_clean_area_for_group(df)

        assert "uema_area" in result.columns
        # All EWM values should be close to 100
        for i in range(5):
            assert result["uema_area"][i] == pytest.approx(100.0, rel=0.1)


class TestTukeyOutlierDetection:
    """Tests for the Tukey outlier detection function."""

    def test_no_outliers_in_normal_data(self):
        """Test that normal data has no outliers flagged."""
        df = pl.DataFrame(
            {
                "experiment": [1] * 5,
                "zone": [1] * 5,
                "time": make_times(1) * 5,  # All same time (across plants)
                "area": [100.0, 102.0, 98.0, 105.0, 95.0],  # Normal variation
            }
        )
        result = apply_tukey_outlier_detection(df)

        # No outliers should be detected
        assert result["tukey_outlier"].sum() == 0
        assert result["area_after_tukey"].to_list() == [100.0, 102.0, 98.0, 105.0, 95.0]

    def test_extreme_outlier_detected(self):
        """Test that an extreme outlier is detected and set to 0."""
        df = pl.DataFrame(
            {
                "experiment": [1] * 5,
                "zone": [1] * 5,
                "time": make_times(1) * 5,  # All same time
                "area": [100.0, 102.0, 98.0, 105.0, 10000.0],  # Last is extreme outlier
            }
        )
        result = apply_tukey_outlier_detection(df)

        # Only the extreme outlier should be flagged
        assert result["tukey_outlier"].to_list() == [False, False, False, False, True]
        # Outlier area should be set to 0
        assert result["area_after_tukey"][4] == 0.0

    def test_conservative_threshold(self):
        """Test that the conservative k=3.0 doesn't flag moderate outliers."""
        # With k=3.0, need very extreme values to be flagged
        df = pl.DataFrame(
            {
                "experiment": [1] * 6,
                "zone": [1] * 6,
                "time": make_times(1) * 6,
                "area": [
                    100.0,
                    110.0,
                    90.0,
                    120.0,
                    80.0,
                    150.0,
                ],  # 150 is high but not extreme
            }
        )
        result = apply_tukey_outlier_detection(df)

        # With conservative k=3.0, 150 should NOT be an outlier
        # Q1 ~= 90, Q3 ~= 120, IQR = 30
        # Upper fence = 120 + 3*30 = 210
        # 150 < 210, so not an outlier
        assert result.filter(pl.col("area") == 150.0)["tukey_outlier"][0] == False


class TestTransformArea:
    """Tests for the transform_area function."""

    def test_groups_processed_separately(self):
        """Test that each (experiment, zone) group is processed independently."""
        times = make_times(2)
        df = pl.DataFrame(
            {
                "experiment": [1, 1, 2, 2],
                "zone": [1, 1, 1, 1],
                "time": times + times,  # Each group gets same times
                "area": [100.0, 105.0, 200.0, 210.0],
            }
        )
        result = transform_area(df)

        # Each group should have its own clean_area values
        exp1 = result.filter(pl.col("experiment") == 1).sort("time")
        exp2 = result.filter(pl.col("experiment") == 2).sort("time")

        assert exp1["clean_area"].to_list() == [100.0, 105.0]
        assert exp2["clean_area"].to_list() == [200.0, 210.0]

    def test_missing_area_column_raises_error(self):
        """Test that missing area column raises ValueError."""
        df = pl.DataFrame(
            {
                "experiment": [1],
                "zone": [1],
                "time": [datetime(2024, 1, 1, 9, 30)],
            }
        )
        with pytest.raises(ValueError, match="must have 'area' column"):
            transform_area(df)

    def test_tukey_outlier_removed_before_ewm(self):
        """Test that Tukey outliers are set to 0 and then handled by EWM logic."""
        # Create data with multiple plants at the same timestep
        # Tukey detection works ACROSS plants at each time, not within a time series
        base_time = datetime(2024, 1, 1, 9, 30)
        times = [base_time] * 6 + [base_time + timedelta(minutes=5)] * 6

        # At first timestep: 5 normal plants + 1 extreme outlier
        # At second timestep: all normal
        df = pl.DataFrame(
            {
                "experiment": [1] * 12,
                "zone": [1] * 12,
                "time": times,
                "plant_id": [0, 1, 2, 3, 4, 5] * 2,  # 6 plants, 2 timesteps
                "area": [
                    100.0,
                    102.0,
                    98.0,
                    105.0,
                    95.0,
                    10000.0,  # Last is extreme at t=0
                    100.0,
                    102.0,
                    98.0,
                    105.0,
                    95.0,
                    100.0,
                ],  # All normal at t=1
            }
        )
        result = transform_area(df)

        # The extreme outlier at the first timestep should be detected and set to 0
        # by Tukey, then replaced by EWM logic (None since it's the first observation for that plant)
        outlier_row = result.filter(
            (pl.col("time") == base_time) & (pl.col("plant_id") == 5)
        )
        # The extreme outlier should NOT be 10000.0 in clean_area
        assert outlier_row["clean_area"][0] != 10000.0

        # Tukey should have flagged it
        assert outlier_row["tukey_outlier"][0] == True


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


class TestMorphologyFeatureReplacement:
    """Tests for morphology feature replacement when outliers are detected."""

    def test_all_morphology_features_replaced_on_outlier(self):
        """Test that all morphology features are replaced when an outlier is detected."""
        from transforms.states import MORPHOLOGY_FEATURES

        times = make_times(5)
        # Create data with an outlier at index 2
        df = pl.DataFrame(
            {
                "experiment": [1] * 5,
                "zone": [1] * 5,
                "time": times,
                "area": [100.0, 105.0, 500.0, 110.0, 115.0],  # 500 is outlier
                "convex_hull_area": [
                    150.0,
                    155.0,
                    700.0,
                    160.0,
                    165.0,
                ],  # Should be replaced
                "perimeter": [40.0, 42.0, 100.0, 44.0, 46.0],  # Should be replaced
                "width": [10.0, 11.0, 25.0, 12.0, 13.0],  # Should be replaced
                "height": [8.0, 9.0, 20.0, 10.0, 11.0],  # Should be replaced
            }
        )
        result = transform_area(df)

        # The outlier at index 2 should be detected
        assert result["is_outlier"][2] == True

        # All morphology features should be replaced with previous clean values
        assert result["clean_area"][2] == 105.0  # Previous clean area
        assert result["clean_convex_hull_area"][2] == 155.0  # Previous clean value
        assert result["clean_perimeter"][2] == 42.0  # Previous clean value
        assert result["clean_width"][2] == 11.0  # Previous clean value
        assert result["clean_height"][2] == 9.0  # Previous clean value

    def test_non_outlier_features_preserved(self):
        """Test that non-outlier observations preserve their original feature values."""
        times = make_times(3)
        df = pl.DataFrame(
            {
                "experiment": [1] * 3,
                "zone": [1] * 3,
                "time": times,
                "area": [100.0, 105.0, 110.0],  # Normal growth
                "convex_hull_area": [150.0, 155.0, 160.0],
                "perimeter": [40.0, 42.0, 44.0],
            }
        )
        result = transform_area(df)

        # No outliers should be detected
        assert result["is_outlier"].sum() == 0

        # All clean values should match original values
        for i in range(3):
            assert result["clean_area"][i] == result["area"][i]
            assert result["clean_convex_hull_area"][i] == result["convex_hull_area"][i]
            assert result["clean_perimeter"][i] == result["perimeter"][i]


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_realistic_plant_growth_with_failures(self):
        """Test a realistic scenario with plant growth and vision failures."""
        # Simulate 14 days of plant growth with occasional failures
        base_time = datetime(2024, 1, 1, 9, 30)
        times = [base_time + timedelta(days=d) for d in range(14)]

        # Normal growth: starts at 100, grows ~10% per day
        base_areas = [100.0 * (1.1**d) for d in range(14)]

        # Inject failures:
        # Day 5: masked pot (10x normal)
        # Day 10: failed detection (near zero)
        areas = base_areas.copy()
        areas[5] = areas[5] * 10  # Pot masked
        areas[10] = 5.0  # Failed detection

        df = pl.DataFrame(
            {
                "experiment": [1] * 14,
                "zone": [1] * 14,
                "time": times,
                "area": areas,
            }
        )

        result = transform_state(df)

        # The outliers should be replaced
        # Day 5 clean_area should NOT be the spike
        assert result["clean_area"][5] < areas[5]
        # Day 10 clean_area should NOT be the near-zero
        assert result["clean_area"][10] > 5.0

        # The clean_area should follow a reasonable growth pattern
        clean_areas = result["clean_area"].to_list()
        # Filter out None values and check general trend
        valid_clean = [a for a in clean_areas if a is not None]
        assert len(valid_clean) > 0
        # Should generally increase over time (plant growth)
        assert valid_clean[-1] > valid_clean[0]
