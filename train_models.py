import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path
from config_utils import TrainingConfig
import yaml
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f) or {}
except FileNotFoundError:
    config = {}



def validate_training_columns(data, features, target):
    """Ensure that all feature and target columns exist in the DataFrame."""
    missing = [c for c in features + [target] if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def train_all_models(X_train, X_test, y_train, y_test):
    """Train multiple regression models and compute an ensemble.

    Parameters
    ----------
    X_train, X_test : array-like
        Feature matrices for training and evaluation.
    y_train, y_test : array-like
        Target values for training and evaluation.

    Returns
    -------
    predictions : dict
        Predictions from each model and the ensemble on ``X_test``.
    performance : dict
        R² and RMSE metrics for each model and the ensemble.
    """
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            objective="reg:squarederror",
        ),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
    }

    predictions = {}
    performance = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        predictions[name] = preds
        performance[name] = {
            "R2": r2_score(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        }

    ensemble_preds = np.mean(np.column_stack(list(predictions.values())), axis=1)
    predictions["Ensemble"] = ensemble_preds
    performance["Ensemble"] = {
        "R2": r2_score(y_test, ensemble_preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, ensemble_preds)),
    }

    return predictions, performance


if __name__ == "__main__":
    cfg = TrainingConfig.from_yaml(Path("config.yaml"))
    X_train = np.load(cfg.train_features)
    X_test = np.load(cfg.test_features)
    y_train = np.load(cfg.train_target)
    y_test = np.load(cfg.test_target)

    _, performance = train_all_models(X_train, X_test, y_train, y_test)
    print("Model performance:")
    for name, metrics in performance.items():
        print(f"{name}: R2={metrics['R2']:.3f}, RMSE={metrics['RMSE']:.3f}")
