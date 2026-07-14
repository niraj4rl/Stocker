import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from models.model_io_store import ModelIORecord, ModelIOStore


def test_model_io_record_post_init():
    record = ModelIORecord(
        ticker="RELIANCE.NS",
        paradigm="regression",
        model_name="xgb",
        context="live_prediction",
        input_payload={"rsi": 55.4, "macd": 1.2},
        output_payload={"predicted_return": 0.015, "signal": "Buy"},
        regime="Bull"
    )
    assert record.predicted_at is not None
    assert record.created_at is not None
    assert record.ticker == "RELIANCE.NS"
    assert record.input_payload["rsi"] == 55.4


@patch("psycopg2.extras.execute_batch")
@patch("psycopg2.connect")
def test_model_io_store_log_io(mock_connect, mock_execute_batch):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    store = ModelIOStore(db_url="mock_db_url")
    record = ModelIORecord(
        ticker="RELIANCE.NS",
        paradigm="regression",
        model_name="xgb",
        context="live_prediction",
        input_payload={"rsi": 55.4},
        output_payload={"predicted_return": 0.015},
        regime="Bull"
    )

    store.log_io(record)

    # Check mock interaction
    assert mock_execute_batch.called
    store.close()
    mock_conn.close.assert_called_once()
