import datetime
import polars as pl

from transforms.attributes import get_agent_name, transform_experiment_attributes


def test_get_agent_name():
    df = pl.DataFrame(
        {
            "experiment": [11, 13, 14, 14, 14, 15, 15, 99],
            "zone": [1, 1, 1, 2, 8, 2, 4, 1],
        }
    )

    df_out = get_agent_name(df)

    assert "agent" in df_out.columns
    expected_agents = [
        "Uniform_Discrete",
        "Uniform_Dirichlet",
        "Constant_White",
        "InAC_Data_Det",
        "InAC_GP_Det_Opt0",
        "InAC_2",
        "InAC_4",
        "Other",
    ]
    assert df_out["agent"].to_list() == expected_agents


def test_transform_experiment_attributes():
    df = pl.DataFrame({"data": [1, 2, 3]})
    exp_id = 14
    zone_id = 1

    df_out = transform_experiment_attributes(df, exp_id, zone_id)

    assert "experiment" in df_out.columns
    assert "zone" in df_out.columns
    assert "agent" in df_out.columns

    assert df_out["experiment"].unique()[0] == exp_id
    assert df_out["zone"].unique()[0] == zone_id
    assert df_out["agent"].unique()[0] == "Constant_White"


def test_transform_experiment_attributes_with_events():
    df = pl.DataFrame({"data": [1]})

    # Test Experiment 9 which has multiple events
    exp_id = 9
    zone_id = 1

    df_out = transform_experiment_attributes(df, exp_id, zone_id)

    # Check if columns exist
    assert "sterilized_date" in df_out.columns
    assert "transplant_date" in df_out.columns
    assert "water_transplant_l" in df_out.columns

    # Check values
    year = 2025
    # E9 sterilized: May 12
    assert df_out["sterilized_date"][0] == datetime.date(year, 5, 12)
    # E9 transplant: May 23
    assert df_out["transplant_date"][0] == datetime.date(year, 5, 23)
    # E9 water transplant: 1.0L
    assert df_out["water_transplant_l"][0] == 1.0
