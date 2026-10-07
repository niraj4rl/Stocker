import numpy as np
import pandas as pd
from utils.config import TRADING_DAYS_YEAR


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    excess = returns - risk_free / TRADING_DAYS_YEAR
    if excess.std() == 0:
        return 0.0
    return float(np.sqrt(TRADING_DAYS_YEAR) * excess.mean() / excess.std())


def max_drawdown(equity_curve: pd.Series) -> float:
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series) -> float:
    ann_return = returns.mean() * TRADING_DAYS_YEAR
    mdd = abs(max_drawdown((1 + returns).cumprod()))
    if mdd == 0:
        if ann_return > 0:
            return float("inf")
        return 0.0
    return float(ann_return / mdd)


def total_return(returns: pd.Series) -> float:
    return float((1 + returns).prod() - 1)


def hit_rate(predicted_returns: np.ndarray, actual_returns: np.ndarray) -> float:
    """Percentage of times the predicted direction was correct."""
    pred_dir = np.sign(predicted_returns)
    actual_dir = np.sign(actual_returns)
    if len(pred_dir) == 0:
        return 0.0
    return float(np.mean(pred_dir == actual_dir) * 100)


def profit_factor(strategy_returns: np.ndarray) -> float:
    """Gross gains / gross losses."""
    gains = strategy_returns[strategy_returns > 0].sum()
    losses = abs(strategy_returns[strategy_returns < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def compute_regression_metrics(y_true_ret: np.ndarray, y_pred_ret: np.ndarray, current_prices: np.ndarray = None) -> dict:
    """Compute regression-specific metrics: MAE, MSE, RMSE, R², MAPE, directional accuracy."""
    # R2 on returns (more meaningful for financial time series than prices)
    ss_res_ret = np.sum((y_true_ret - y_pred_ret) ** 2)
    ss_tot_ret = np.sum((y_true_ret - np.mean(y_true_ret)) ** 2)
    r2 = float(1 - ss_res_ret / ss_tot_ret) if ss_tot_ret > 0 else 0.0

    dir_acc = hit_rate(y_pred_ret, y_true_ret)

    if current_prices is not None:
        y_true = current_prices * (1 + y_true_ret)
        y_pred = current_prices * (1 + y_pred_ret)
    else:
        y_true = y_true_ret
        y_pred = y_pred_ret

    residuals = y_true - y_pred
    mae = float(np.mean(np.abs(residuals)))
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))

    mask = y_true != 0
    if np.any(mask):
        mape = float(np.mean(np.abs(residuals[mask] / y_true[mask])))
    else:
        mape = 0.0

    return {
        "mae": round(mae, 6),
        "mse": round(mse, 6),
        "rmse": round(rmse, 6),
        "r2": round(r2, 4),
        "mape": round(mape, 6),
        "directional_accuracy": round(dir_acc, 2),
    }


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    """Compute classification-specific metrics: accuracy, F1, precision, recall, confusion matrix, ROC-AUC."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score

    accuracy = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    roc_auc = None
    if y_prob is not None:
        try:
            # Need to handle multi-class ROC AUC
            classes = np.unique(y_true)
            if len(classes) == 2:
                # Binary classification
                if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    y_prob = y_prob[:, 1]
                roc_auc = float(roc_auc_score(y_true, y_prob))
            elif len(classes) > 2 and y_prob.ndim == 2:
                # Multi-class
                roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
        except Exception:
            pass

    return {
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "confusion_matrix": cm,
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
    }


def compute_trading_metrics(strategy_returns: np.ndarray) -> dict:
    """Compute trading-specific metrics from strategy returns."""
    rets = pd.Series(strategy_returns)
    equity = (1 + rets).cumprod()
    n_trades = int((rets != 0).sum())
    win_trades = int((rets > 0).sum())
    loss_trades = int((rets < 0).sum())

    return {
        "sharpe": round(sharpe_ratio(rets), 4),
        "max_drawdown": round(max_drawdown(equity), 4),
        "total_return": round(total_return(rets), 4),
        "calmar": round(calmar_ratio(rets), 4),
        "hit_rate": round(win_trades / n_trades * 100, 2) if n_trades > 0 else 0.0,
        "profit_factor": round(profit_factor(strategy_returns), 4),
        "n_trades": n_trades,
        "win_trades": win_trades,
        "loss_trades": loss_trades,
    }


def compute_all_metrics(returns: pd.Series) -> dict:
    equity = (1 + returns).cumprod()
    return {
        "sharpe": round(sharpe_ratio(returns), 4),
        "max_drawdown": round(max_drawdown(equity), 4),
        "total_return": round(total_return(returns), 4),
        "calmar": round(calmar_ratio(returns), 4),
        "n_trades": int((returns != 0).sum()),
    }

