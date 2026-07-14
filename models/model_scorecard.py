"""
ModelScorecard: captures granular per-model evaluation results.
ScorecardStore: persists scores, runs, and searches to one PostgreSQL table.
"""
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional
from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


MODEL_DISPLAY_NAMES = {
    "xgb": "XGBoost",
    "rf": "Random Forest",
    "ridge": "Ridge",
    "svc": "SVM Classifier",
    "svr": "SVM Regressor",
    "knn": "KNN",
    "mlp": "MLP",
}


def pretty_model_name(model_name: str) -> str:
    if not model_name:
        return "Unknown"
    return MODEL_DISPLAY_NAMES.get(str(model_name).strip().lower(), str(model_name).upper())


@dataclass
class ModelScorecard:
    """One evaluation record for a single model on a single regime/fold."""

    # Identifiers
    ticker: str
    run_id: str                  # unique backtest run identifier
    fold: int
    regime: str                  # Bull / Bear / HighVol
    paradigm: str                # regression / classification
    model_name: str              # xgb / rf / ridge / svc / svr / knn / mlp
    is_winner: bool = False      # whether this was selected as best

    # Trading metrics
    unified_score: float = 0.0  # comparable score across reg/class models
    sharpe: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    hit_rate: float = 0.0
    profit_factor: float = 0.0
    n_trades: int = 0
    win_trades: int = 0
    loss_trades: int = 0

    # Regression metrics (filled for regression models)
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    directional_accuracy: Optional[float] = None

    # Classification metrics (filled for classification models)
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None

    # Metadata
    train_samples: int = 0
    val_samples: int = 0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.unified_score == 0.0 and self.sharpe != 0.0:
            # Sharpe on strategy returns is directly comparable across paradigms.
            self.unified_score = float(self.sharpe)

    def to_dict(self) -> dict:
        return asdict(self)


def decorate_model_records(rows: list[dict]) -> list[dict]:
    decorated = []
    for row in rows:
        item = dict(row)
        raw_name = item.get("model_name", "")
        item["model_key"] = raw_name
        item["model_name"] = pretty_model_name(raw_name)
        decorated.append(item)
    return decorated


class ScorecardStore:
    """Persists all app records to one PostgreSQL table."""

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
        self._create_tables()

    def _create_tables(self):
        with self._conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS stocker_records (
                    id SERIAL PRIMARY KEY,
                    record_type TEXT NOT NULL,
                    ticker TEXT,
                    run_id TEXT,
                    fold INTEGER,
                    regime TEXT,
                    paradigm TEXT,
                    model_name TEXT,
                    is_winner INTEGER,
                    unified_score REAL,
                    sharpe REAL,
                    total_return REAL,
                    max_drawdown REAL,
                    calmar REAL,
                    hit_rate REAL,
                    profit_factor REAL,
                    n_trades INTEGER,
                    win_trades INTEGER,
                    loss_trades INTEGER,
                    mae REAL,
                    rmse REAL,
                    r2 REAL,
                    directional_accuracy REAL,
                    accuracy REAL,
                    f1_score REAL,
                    precision_score REAL,
                    recall REAL,
                    train_samples INTEGER,
                    val_samples INTEGER,

                    n_folds INTEGER,
                    start_date TEXT,
                    end_date TEXT,

                    endpoint TEXT,
                    query_text TEXT,

                    timestamp TEXT
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_stocker_records_type_ts
                ON stocker_records (record_type, timestamp DESC)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_stocker_records_ticker_type
                ON stocker_records (ticker, record_type)
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS stocker_model_io (
                    id SERIAL PRIMARY KEY,
                    run_id TEXT,
                    ticker TEXT NOT NULL,
                    regime TEXT,
                    paradigm TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    context TEXT NOT NULL,
                    input_payload JSONB NOT NULL,
                    output_payload JSONB NOT NULL,
                    actual_price REAL,
                    predicted_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_stocker_model_io_ticker_ts
                ON stocker_model_io (ticker, predicted_at DESC)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_stocker_model_io_run_id
                ON stocker_model_io (run_id)
            """)
            # User requested a single-table schema only.
            cur.execute("DROP TABLE IF EXISTS model_scores")
            cur.execute("DROP TABLE IF EXISTS backtest_runs")
        self._conn.commit()

    def _scorecard_payload(self, sc: ModelScorecard) -> dict:
        d = sc.to_dict()
        d["is_winner"] = int(bool(d.get("is_winner", False)))
        return d

    def log_scorecards(self, scorecards: list[ModelScorecard]):
        if not scorecards:
            return
        payload = [self._scorecard_payload(sc) for sc in scorecards]
        with self._conn.cursor() as cur:
            cur.executemany("""
                INSERT INTO stocker_records (
                    record_type,
                    ticker, run_id, fold, regime, paradigm, model_name, is_winner,
                    unified_score,
                    sharpe, total_return, max_drawdown, calmar,
                    hit_rate, profit_factor, n_trades, win_trades, loss_trades,
                    mae, rmse, r2, directional_accuracy,
                    accuracy, f1_score, precision_score, recall,
                    train_samples, val_samples, timestamp
                ) VALUES (
                    'model_score',
                    %(ticker)s, %(run_id)s, %(fold)s, %(regime)s, %(paradigm)s, %(model_name)s, %(is_winner)s,
                    %(unified_score)s,
                    %(sharpe)s, %(total_return)s, %(max_drawdown)s, %(calmar)s,
                    %(hit_rate)s, %(profit_factor)s, %(n_trades)s, %(win_trades)s, %(loss_trades)s,
                    %(mae)s, %(rmse)s, %(r2)s, %(directional_accuracy)s,
                    %(accuracy)s, %(f1_score)s, %(precision)s, %(recall)s,
                    %(train_samples)s, %(val_samples)s, %(timestamp)s
                )
            """, payload)
        self._conn.commit()

    def log_scorecard(self, sc: ModelScorecard):
        self.log_scorecards([sc])

    def log_run(self, run_id: str, ticker: str, n_folds: int,
                start_date: str = "", end_date: str = ""):
        with self._conn.cursor() as cur:
            cur.execute("""
                INSERT INTO stocker_records
                (record_type, run_id, ticker, n_folds, start_date, end_date, timestamp)
                VALUES ('backtest_run', %s, %s, %s, %s, %s, %s)
            """, (run_id, ticker, n_folds, start_date, end_date,
                  datetime.now().isoformat()))
        self._conn.commit()

    def log_search(self, query_text: str, endpoint: str, ticker: str = "") -> dict:
        """Log a search/query event and retain only the latest 20 search records."""
        now = datetime.now().isoformat()
        force_refresh = False
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO stocker_records
                (record_type, ticker, endpoint, query_text, timestamp)
                VALUES ('search', %s, %s, %s, %s)
                """,
                (ticker or None, endpoint, query_text, now),
            )

            cur.execute("SELECT COUNT(*) FROM stocker_records WHERE record_type = 'search'")
            count = int(cur.fetchone()[0])

            if count > 20:
                force_refresh = True
                cur.execute(
                    """
                    DELETE FROM stocker_records
                    WHERE record_type = 'search'
                      AND id NOT IN (
                        SELECT id FROM stocker_records
                        WHERE record_type = 'search'
                        ORDER BY id DESC
                        LIMIT 20
                      )
                    """
                )
                count = 20

        self._conn.commit()
        return {"force_refresh": force_refresh, "recent_search_count": count}

    def get_recent_searches(self, limit: int = 20) -> list[dict]:
        import psycopg2.extras
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, ticker, endpoint, query_text, timestamp
                FROM stocker_records
                WHERE record_type = 'search'
                ORDER BY id DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def get_all_scores(self, ticker: str = None) -> list[dict]:
        """Retrieve all model scores, optionally filtered by ticker."""
        import psycopg2.extras
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if ticker:
                cur.execute(
                    "SELECT * FROM stocker_records WHERE record_type = 'model_score' AND ticker = %s ORDER BY timestamp DESC", (ticker,)
                )
            else:
                cur.execute(
                    "SELECT * FROM stocker_records WHERE record_type = 'model_score' ORDER BY timestamp DESC"
                )
            rows = cur.fetchall()
        return decorate_model_records([dict(r) for r in rows])

    def get_leaderboard(self, ticker: str = None) -> list[dict]:
        """Aggregated best scores per model_name × paradigm × regime."""
        import psycopg2.extras
        query = """
            SELECT model_name, paradigm, regime,
                     AVG(unified_score) as avg_unified_score,
                   AVG(sharpe) as avg_sharpe,
                   AVG(total_return) as avg_return,
                   AVG(hit_rate) as avg_hit_rate,
                   AVG(mae) as avg_mae,
                   AVG(rmse) as avg_rmse,
                   AVG(accuracy) as avg_accuracy,
                   AVG(f1_score) as avg_f1,
                   COUNT(*) as n_evaluations,
                   SUM(is_winner) as times_selected
            FROM stocker_records
            WHERE record_type = 'model_score'
        """
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if ticker:
                query += " AND ticker = %s"
                query += " GROUP BY model_name, paradigm, regime ORDER BY avg_unified_score DESC NULLS LAST, avg_sharpe DESC"
                cur.execute(query, (ticker,))
            else:
                query += " GROUP BY model_name, paradigm, regime ORDER BY avg_unified_score DESC NULLS LAST, avg_sharpe DESC"
                cur.execute(query)
            rows = cur.fetchall()
        return decorate_model_records([dict(r) for r in rows])

    def get_latest_run_id(self, ticker: str) -> str:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT run_id FROM stocker_records WHERE record_type = 'backtest_run' AND ticker = %s ORDER BY timestamp DESC LIMIT 1",
                (ticker,)
            )
            row = cur.fetchone()
        return row[0] if row else ""

    def close(self):
        self._conn.close()
