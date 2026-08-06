"""
Export utilities for model analysis data.
Generates CSV files and formatted reports for research use.
"""
import pandas as pd
from pathlib import Path
from models.model_scorecard import ScorecardStore
from utils.config import ROOT


EXPORT_DIR = ROOT / "data" / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def export_model_comparison_csv(ticker: str = None, output_path: str = None) -> str:
    """Export all model scores as a CSV file (suitable for paper tables)."""
    store = ScorecardStore()
    scores = store.get_all_scores(ticker)
    store.close()

    if not scores:
        return ""

    df = pd.DataFrame(scores)

    # Reorder columns for readability
    col_order = [
        "ticker", "fold", "regime", "paradigm", "model_name", "model_key", "is_winner",
        "unified_score",
        "sharpe", "total_return", "max_drawdown", "calmar",
        "hit_rate", "profit_factor", "n_trades", "win_trades", "loss_trades",
        "mae", "rmse", "r2", "directional_accuracy",
        "accuracy", "f1_score", "precision", "precision_score", "recall",
        "train_samples", "val_samples", "timestamp",
    ]
    available_cols = [c for c in col_order if c in df.columns]
    df = df[available_cols]

    if output_path is None:
        suffix = f"_{ticker.replace('.', '_')}" if ticker else "_all"
        output_path = str(EXPORT_DIR / f"model_comparison{suffix}.csv")

    df.to_csv(output_path, index=False)
    print(f"[export] Saved model comparison to {output_path}")
    return output_path


def export_leaderboard_csv(ticker: str = None, output_path: str = None) -> str:
    """Export aggregated leaderboard as CSV."""
    store = ScorecardStore()
    leaderboard = store.get_leaderboard(ticker)
    store.close()

    if not leaderboard:
        return ""

    df = pd.DataFrame(leaderboard)

    if output_path is None:
        suffix = f"_{ticker.replace('.', '_')}" if ticker else "_all"
        output_path = str(EXPORT_DIR / f"leaderboard{suffix}.csv")

    df.to_csv(output_path, index=False)
    print(f"[export] Saved leaderboard to {output_path}")
    return output_path


def export_regime_analysis_csv(ticker: str = None, output_path: str = None) -> str:
    """Export regime-wise model performance for paper analysis."""
    store = ScorecardStore()
    scores = store.get_all_scores(ticker)
    store.close()

    if not scores:
        return ""

    df = pd.DataFrame(scores)

    # Aggregate by regime × paradigm
    summary = df.groupby(["regime", "paradigm", "model_name"]).agg({
        "sharpe": ["mean", "std", "count"],
        "hit_rate": "mean",
        "total_return": "mean",
        "max_drawdown": "mean",
    }).round(4)

    summary.columns = ["_".join(col) for col in summary.columns]
    summary = summary.reset_index()

    if output_path is None:
        suffix = f"_{ticker.replace('.', '_')}" if ticker else "_all"
        output_path = str(EXPORT_DIR / f"regime_analysis{suffix}.csv")

    summary.to_csv(output_path, index=False)
    print(f"[export] Saved regime analysis to {output_path}")
    return output_path


def get_statistical_comparison(ticker: str = None) -> dict:
    """
    Run paired statistical tests: regressor vs classifier returns.
    Returns p-values for Wilcoxon signed-rank and paired t-test.
    """
    from scipy import stats

    store = ScorecardStore()
    scores = store.get_all_scores(ticker)
    store.close()

    if not scores:
        return {}

    df = pd.DataFrame(scores)

    # Compare regression vs classification Sharpe per regime
    results = {}
    for regime in df["regime"].unique():
        reg_scores = df[(df["regime"] == regime) & (df["paradigm"] == "regression")]["sharpe"].values
        cls_scores = df[(df["regime"] == regime) & (df["paradigm"] == "classification")]["sharpe"].values

        n = min(len(reg_scores), len(cls_scores))
        if n < 3:
            results[regime] = {"n_pairs": n, "note": "insufficient data"}
            continue

        reg_s = reg_scores[:n]
        cls_s = cls_scores[:n]

        try:
            t_stat, t_pval = stats.ttest_rel(reg_s, cls_s)
        except Exception:
            t_stat, t_pval = 0, 1.0

        try:
            w_stat, w_pval = stats.wilcoxon(reg_s, cls_s)
        except Exception:
            w_stat, w_pval = 0, 1.0

        results[regime] = {
            "n_pairs": int(n),
            "reg_mean_sharpe": round(float(reg_s.mean()), 4),
            "cls_mean_sharpe": round(float(cls_s.mean()), 4),
            "paired_ttest_pval": round(float(t_pval), 4),
            "wilcoxon_pval": round(float(w_pval), 4),
            "significant_005": bool(float(t_pval) < 0.05),
        }

    return results
