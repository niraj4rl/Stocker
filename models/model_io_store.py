"""
ModelIORecord: captures input/output details of each prediction.
ModelIOStore: persists prediction inputs and outputs to the stocker_model_io table.
"""
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


@dataclass
class ModelIORecord:
    """One record of input and output for a single prediction."""
    ticker: str
    paradigm: str                # regression / classification
    model_name: str              # xgb / rf / ridge / svc / svr / knn / mlp
    context: str                 # live_prediction / backtest
    input_payload: dict[str, Any]  # 22 engineered features
    output_payload: dict[str, Any] # prediction outputs (signals, probabilities etc.)
    run_id: Optional[str] = None  # backtest UUID if applicable
    regime: Optional[str] = None  # Bull / Bear / HighVol
    actual_price: Optional[float] = None
    predicted_at: Optional[str] = None
    created_at: Optional[str] = None

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.predicted_at:
            self.predicted_at = now
        if not self.created_at:
            self.created_at = now

    def to_dict(self) -> dict:
        return asdict(self)


class ModelIOStore:
    """Persists model input/output prediction records to PostgreSQL."""

    def __init__(self, db_url: Optional[str] = None):
        import psycopg2
        import psycopg2.extras

        self.host = os.environ.get("DB_HOST", "localhost")
        self.port = os.environ.get("DB_PORT", "5432")
        self.dbname = os.environ.get("DB_NAME", "stockobserver")
        self.user = os.environ.get("DB_USER", "postgres")
        self.password = os.environ.get("DB_PASSWORD") or os.environ.get("DB_PASS", "")
        self.connect_timeout = int(os.environ.get("DB_CONNECT_TIMEOUT", "3"))

        self.db_url = db_url or (
            f"host={self.host} port={self.port} dbname={self.dbname} "
            f"user={self.user} password={self.password} connect_timeout={self.connect_timeout}"
        )
        self._conn = psycopg2.connect(self.db_url)

    def log_io(self, record: ModelIORecord):
        """Insert a single ModelIORecord."""
        self.log_io_batch([record])

    def log_io_batch(self, records: list[ModelIORecord]):
        """Bulk insert multiple ModelIORecords."""
        if not records:
            return

        import psycopg2.extras

        payloads = []
        for r in records:
            d = r.to_dict()
            d["input_payload"] = json.dumps(d["input_payload"])
            d["output_payload"] = json.dumps(d["output_payload"])
            payloads.append(d)

        with self._conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                """
                INSERT INTO stocker_model_io (
                    run_id, ticker, regime, paradigm, model_name, context,
                    input_payload, output_payload, actual_price, predicted_at, created_at
                ) VALUES (
                    %(run_id)s, %(ticker)s, %(regime)s, %(paradigm)s, %(model_name)s, %(context)s,
                    %(input_payload)s::jsonb, %(output_payload)s::jsonb, %(actual_price)s, %(predicted_at)s, %(created_at)s
                )
                """,
                payloads
            )
        self._conn.commit()

    def close(self):
        self._conn.close()
