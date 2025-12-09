"""Feature engineering for rainfall prediction."""

import pandas as pd


def create_lag_features(df: pd.DataFrame, rainfall_col: str = "rainfall_mm", max_lag: int = 7) -> pd.DataFrame:
    """Create lagged rainfall features.

    Args:
        df: DataFrame with rainfall data (must be sorted by date)
        rainfall_col: Name of the rainfall column
        max_lag: Number of lag days to create (default: 7)

    Returns:
        DataFrame with lag features added
    """
    df = df.copy()
    for i in range(1, max_lag + 1):
        df[f"rain_lag_{i}"] = df[rainfall_col].shift(i)
    return df


def create_date_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Create date-based features.

    Args:
        df: DataFrame with a date column
        date_col: Name of the date column

    Returns:
        DataFrame with month, day, dayofyear features added
    """
    df = df.copy()
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["dayofyear"] = df[date_col].dt.dayofyear
    return df


def prepare_features(df: pd.DataFrame, rainfall_col: str = "rainfall_mm", date_col: str = "date") -> pd.DataFrame:
    """Prepare all features needed for the model.

    Args:
        df: DataFrame with date and rainfall columns
        rainfall_col: Name of the rainfall column
        date_col: Name of the date column

    Returns:
        DataFrame with all features, NaN rows dropped
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    df = create_lag_features(df, rainfall_col)
    df = create_date_features(df, date_col)
    return df.dropna()

