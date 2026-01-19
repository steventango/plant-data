import pytest
from unittest.mock import MagicMock, patch
import polars as pl
import requests
from transforms.terminal import transform_terminal


@pytest.fixture
def sample_df():
    return pl.DataFrame(
        {
            "day": [0, 1],
            "cls_token": [
                [0.1] * 768,
                [0.2] * 768,
            ],
        }
    )


@patch("transforms.terminal.requests.Session")
def test_transform_terminal_success_high(mock_session_cls, sample_df):
    """Test successful transformation with high probability."""
    mock_session = mock_session_cls.return_value
    mock_response = MagicMock()
    mock_response.json.return_value = {"bolted_probability": 0.85}
    mock_response.status_code = 200
    mock_session.post.return_value = mock_response

    result_df = transform_terminal(sample_df)

    assert "bolted_pred" in result_df.columns
    assert "terminal" in result_df.columns

    assert result_df["bolted_pred"][0] == pytest.approx(0.85, abs=1e-5)
    assert result_df["terminal"][0] == 1


@patch("transforms.terminal.requests.Session")
def test_transform_terminal_success_low(mock_session_cls, sample_df):
    """Test successful transformation with low probability."""
    mock_session = mock_session_cls.return_value
    mock_response = MagicMock()
    mock_response.json.return_value = {"bolted_probability": 0.15}
    mock_response.status_code = 200
    mock_session.post.return_value = mock_response

    result_df = transform_terminal(sample_df)

    assert "bolted_pred" in result_df.columns
    assert "terminal" in result_df.columns

    assert result_df["bolted_pred"][0] == pytest.approx(0.15, abs=1e-5)
    assert result_df["terminal"][0] == 0


@patch("transforms.terminal.requests.Session")
def test_transform_terminal_mixed_probabilities(mock_session_cls, sample_df):
    """Test transformation with mixed probabilities to verify logic."""
    mock_session = mock_session_cls.return_value

    def side_effect(*args, **kwargs):
        json_body = kwargs.get("json", {})
        embedding = json_body.get("embedding", [])
        # If embedding starts with 0.1, return 0.9 (bolted), else 0.1 (not bolted)
        if embedding and embedding[0] == 0.1:
            return MagicMock(json=lambda: {"bolted_probability": 0.9}, status_code=200)
        else:
            return MagicMock(json=lambda: {"bolted_probability": 0.1}, status_code=200)

    mock_session.post.side_effect = side_effect

    result_df = transform_terminal(sample_df)

    # Row 0: 0.1 -> 0.9 -> terminal 1
    assert result_df["bolted_pred"][0] == pytest.approx(0.9, abs=1e-5)
    assert result_df["terminal"][0] == 1

    # Row 1: 0.2 -> 0.1 -> terminal 0
    assert result_df["bolted_pred"][1] == pytest.approx(0.1, abs=1e-5)
    assert result_df["terminal"][1] == 0


@patch("transforms.terminal.requests.Session")
def test_transform_terminal_api_failure(mock_session_cls, sample_df):
    """Test graceful handling of API failures."""
    mock_session = mock_session_cls.return_value
    mock_session.post.side_effect = requests.RequestException("API Error")

    result_df = transform_terminal(sample_df)

    assert result_df["bolted_pred"][0] is None
    assert result_df["terminal"][0] is None
    assert result_df["bolted_pred"][1] is None
    assert result_df["terminal"][1] is None


@patch("transforms.terminal.requests.Session")
def test_transform_terminal_day_13(mock_session_cls):
    mock_session = mock_session_cls.return_value
    mock_session.post.return_value = MagicMock(
        json=lambda: {"bolted_probability": 0.1}, status_code=200
    )
    df = pl.DataFrame(
        {
            "day": [12, 13, 14],
            "cls_token": [[0.1] * 768] * 3,
        }
    )
    result_df = transform_terminal(df)
    assert result_df["terminal"][0] == 0
    assert result_df["terminal"][1] == 1
    assert result_df["terminal"][2] == 0
