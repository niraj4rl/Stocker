import os
import psycopg2
import pandas as pd

from dotenv import load_dotenv

# Load .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)

db_user = os.getenv("DB_USER", "postgres")
db_pass = os.getenv("DB_PASSWORD", "postgres")
db_host = os.getenv("DB_HOST", "localhost")
db_port = os.getenv("DB_PORT", "5432")
db_name = os.getenv("DB_NAME", "stockobserver")

# Load DB URL from environment or use constructed path
DB_URL = os.getenv("DB_URL", f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}")

def generate_research_paper_tables():
    """
    Connects to the Stocker PostgreSQL database, extracts all out-of-sample metrics,
    and aggregates them into clean tables suitable for a research paper.
    """
    print(f"Connecting to database to generate research tables...")
    try:
        conn = psycopg2.connect(DB_URL)
    except Exception as e:
        print(f"Error connecting to DB: {e}")
        print("Make sure your PostgreSQL database is running and DB_URL is configured.")
        return

    # 1. Fetch all model scorecard records
    query = """
        SELECT 
            model_name, paradigm, regime, is_winner,
            sharpe, total_return, max_drawdown, hit_rate,
            mae, mse, rmse, r2, mape, directional_accuracy,
            accuracy, f1_score, precision_score, recall, roc_auc
        FROM stocker_records
    """
    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        print("No evaluation records found in the database. Please run a backtest first.")
        return

    print(f"Loaded {len(df)} evaluation records.")

    # ---------------------------------------------------------
    # TABLE 1: Regression Metrics Summary (Price Scale Errors)
    # ---------------------------------------------------------
    reg_df = df[df["paradigm"] == "regression"].copy()
    if not reg_df.empty:
        reg_summary = reg_df.groupby(["regime", "model_name"])[[
            "rmse", "mae", "mape", "r2", "directional_accuracy", "sharpe"
        ]].mean().round(4).reset_index()
        
        reg_summary.sort_values(by=["regime", "rmse"], ascending=[True, True], inplace=True)
        reg_summary.to_csv("regression_results_table.csv", index=False)
        
        print("\n--- TABLE 1: Regression Metrics (Averaged by Regime) ---")
        print(reg_summary.to_markdown(index=False))
        print("Saved to regression_results_table.csv\n")

    # ---------------------------------------------------------
    # TABLE 2: Classification Metrics Summary
    # ---------------------------------------------------------
    cls_df = df[df["paradigm"] == "classification"].copy()
    if not cls_df.empty:
        cls_summary = cls_df.groupby(["regime", "model_name"])[[
            "accuracy", "f1_score", "precision_score", "recall", "roc_auc", "sharpe"
        ]].mean().round(4).reset_index()
        
        cls_summary.sort_values(by=["regime", "accuracy"], ascending=[True, False], inplace=True)
        cls_summary.to_csv("classification_results_table.csv", index=False)

        print("\n--- TABLE 2: Classification Metrics (Averaged by Regime) ---")
        print(cls_summary.to_markdown(index=False))
        print("Saved to classification_results_table.csv\n")

    # ---------------------------------------------------------
    # TABLE 3: The Adaptive Router Advantage (Winning Models Only)
    # ---------------------------------------------------------
    winners_df = df[df["is_winner"] == 1].copy()
    if not winners_df.empty:
        router_summary = winners_df.groupby(["regime"])[[
            "model_name", "paradigm", "sharpe", "total_return", "max_drawdown"
        ]].agg({
            "model_name": lambda x: x.mode().iloc[0], # Most frequent winning model
            "paradigm": lambda x: x.mode().iloc[0],
            "sharpe": "mean",
            "total_return": "mean",
            "max_drawdown": "mean"
        }).round(4).reset_index()

        router_summary.to_csv("adaptive_router_results.csv", index=False)
        print("\n--- TABLE 3: Adaptive Router Dominance ---")
        print(router_summary.to_markdown(index=False))
        print("Saved to adaptive_router_results.csv\n")

if __name__ == "__main__":
    generate_research_paper_tables()
