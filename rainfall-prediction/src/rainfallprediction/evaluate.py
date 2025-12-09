"""Model evaluation metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    """Compute regression metrics.

    Args:
        actual: Actual rainfall values
        predicted: Predicted rainfall values

    Returns:
        Dict with rmse, mae, r2 metrics
    """
    return {
        "rmse": np.sqrt(mean_squared_error(actual, predicted)),
        "mae": mean_absolute_error(actual, predicted),
        "r2": r2_score(actual, predicted),
    }


def compute_metrics_by_month(df: pd.DataFrame, actual_col: str, predicted_col: str) -> pd.DataFrame:
    """Compute metrics grouped by month.

    Args:
        df: DataFrame with predictions and month column
        actual_col: Name of actual values column
        predicted_col: Name of predicted values column

    Returns:
        DataFrame with metrics per month
    """
    results = []
    for month in sorted(df["month"].unique()):
        month_data = df[df["month"] == month]
        metrics = compute_metrics(month_data[actual_col], month_data[predicted_col])
        metrics["month"] = month
        results.append(metrics)
    return pd.DataFrame(results)[["month", "rmse", "mae", "r2"]]

