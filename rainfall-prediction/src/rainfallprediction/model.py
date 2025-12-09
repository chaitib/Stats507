"""Model loading and prediction."""

import pickle
from pathlib import Path
import pandas as pd

# Default model path
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "rainfall.pkl"


def load_model(model_path: Path | str | None = None):
    """Load the trained LightGBM model.

    Args:
        model_path: Path to .pkl file. Defaults to models/rainfall.pkl

    Returns:
        LightGBM Booster model
    """
    path = Path(model_path) if model_path else MODEL_PATH
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(model, df: pd.DataFrame) -> pd.Series:
    """Generate predictions using the model.

    Args:
        model: Loaded LightGBM Booster
        df: DataFrame with required features

    Returns:
        Series of predicted rainfall values in mm
    """
    feature_names = model.feature_name()
    X = df[feature_names]
    return pd.Series(model.predict(X), index=df.index)


def get_feature_names(model) -> list[str]:
    """Get the feature names expected by the model."""
    return model.feature_name()


def get_feature_importance(model) -> pd.DataFrame:
    """Get feature importance as a DataFrame."""
    return pd.DataFrame({
        "feature": model.feature_name(),
        "importance": model.feature_importance(importance_type="gain")
    }).sort_values("importance", ascending=False)

