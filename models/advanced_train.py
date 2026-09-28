"""
Advanced model training with cross-validation and hyperparameter tuning.

Trains six regression models with GridSearchCV + 5-fold cross-validation
and reports mean ± std for R², RMSE, MAE.
"""
import os
import joblib
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb


MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)
os.makedirs(MODELS_DIR, exist_ok=True)

HOUSING_FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveOccup"]
HOUSING_TARGET = "MEDV"


def build_model_zoo() -> dict:
    """Return a mapping of model name -> (estimator, param_grid)."""
    return {
        "Linear": (LinearRegression(), {}),
        "Ridge": (Ridge(random_state=42), {"model__alpha": [0.1, 1.0, 10.0]}),
        "RandomForest": (
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {
                "model__n_estimators": [80, 150],
                "model__max_depth": [None, 12],
            },
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=42),
            {
                "model__n_estimators": [80, 150],
                "model__learning_rate": [0.05, 0.1],
            },
        ),
        "XGBoost": (
            xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 6],
            },
        ),
        "MLP": (
            MLPRegressor(
                random_state=42, max_iter=800, early_stopping=True,
            ),
            {
                "model__hidden_layer_sizes": [(64, 32), (128, 64, 32)],
                "model__alpha": [1e-4, 1e-3],
            },
        ),
    }


def evaluate_all_models(df, n_splits: int = 5, tune: bool = True) -> dict:
    """
    Run k-fold cross-validation on every model in the zoo.
    Optionally runs GridSearchCV for hyperparameter tuning.

    Returns a dict:
        { model_name: {model, r2_mean, r2_std, rmse_mean, rmse_std,
                       mae_mean, best_params}, ...,
          "best_model": name }
    """
    X = df[HOUSING_FEATURES].values
    y = df[HOUSING_TARGET].values
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {}
    zoo = build_model_zoo()

    for name, (estimator, grid) in zoo.items():
        print(f"[advanced_train] Training {name}...")
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])

        if tune and grid:
            search = GridSearchCV(
                pipe, grid, cv=kf, scoring="r2", n_jobs=-1, refit=True,
            )
            search.fit(X, y)
            best = search.best_estimator_
            best_params = search.best_params_
        else:
            best = pipe.fit(X, y)
            best_params = {}

        cv_r2 = cross_val_score(best, X, y, cv=kf, scoring="r2")
        cv_rmse = -cross_val_score(
            best, X, y, cv=kf, scoring="neg_root_mean_squared_error",
        )
        cv_mae = -cross_val_score(
            best, X, y, cv=kf, scoring="neg_mean_absolute_error",
        )

        results[name] = {
            "model": best,
            "r2_mean": float(cv_r2.mean()),
            "r2_std": float(cv_r2.std()),
            "rmse_mean": float(cv_rmse.mean()),
            "rmse_std": float(cv_rmse.std()),
            "mae_mean": float(cv_mae.mean()),
            "best_params": best_params,
        }

        joblib.dump(
            best, os.path.join(MODELS_DIR, f"adv_{name.lower()}.pkl"),
        )

    best_name = max(results, key=lambda k: results[k]["r2_mean"])
    results["best_model"] = best_name
    return results