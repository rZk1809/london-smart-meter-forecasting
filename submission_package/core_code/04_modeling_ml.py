from __future__ import annotations

import inspect
import multiprocessing as mp
import pickle
from pathlib import Path
from typing import Any

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - optional dependency guard
    xgb = None

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover - optional dependency guard
    lgb = None

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover - optional dependency guard
    CatBoostRegressor = None

try:
    from prophet import Prophet
except Exception:  # pragma: no cover - optional dependency guard
    Prophet = None


def _require_xgb() -> None:
    if xgb is None:
        raise RuntimeError("xgboost is required for XGBoost models.")


def _require_lgb() -> None:
    if lgb is None:
        raise RuntimeError("lightgbm is required for LightGBM models.")


def _require_catboost() -> None:
    if CatBoostRegressor is None:
        raise RuntimeError("catboost is required for CatBoost models.")


def _require_prophet() -> None:
    if Prophet is None:
        raise RuntimeError("prophet is required for Prophet models.")


def train_xgboost(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    random_state: int = 42,
    patience: int = 50,
    n_estimators: int = 600,
):
    _require_xgb()
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        early_stopping_rounds=patience,
        random_state=random_state,
        n_jobs=1,
    )
    fit_kwargs: dict[str, Any] = {}
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"] = False
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def train_lightgbm(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    random_state: int = 42,
    patience: int = 50,
    n_estimators: int = 600,
):
    _require_lgb()
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=31,
        random_state=random_state,
        n_jobs=1,
        verbosity=-1,
    )
    fit_kwargs: dict[str, Any] = {}
    fit_signature = inspect.signature(model.fit)
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        if "callbacks" in fit_signature.parameters:
            fit_kwargs["callbacks"] = [lgb.early_stopping(patience, verbose=False)]
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def train_catboost(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    random_state: int = 42,
    patience: int = 50,
    iterations: int = 600,
):
    _require_catboost()
    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=0.03,
        depth=6,
        loss_function="RMSE",
        random_seed=random_state,
        verbose=False,
    )
    fit_kwargs = {}
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = (X_val, y_val)
        fit_kwargs["use_best_model"] = True
        fit_kwargs["verbose"] = False
        if "early_stopping_rounds" in inspect.signature(model.fit).parameters:
            fit_kwargs["early_stopping_rounds"] = patience
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def train_prophet(train_df):
    _require_prophet()
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    try:
        model.add_country_holidays(country_name="UK")
    except Exception:
        pass
    model.fit(train_df[["ds", "y"]].copy())
    return model


def predict_prophet(model, forecast_dates):
    future = forecast_dates[["ds"]].copy()
    forecast = model.predict(future)
    return forecast["yhat"].to_numpy()


def _prophet_worker(train_records, forecast_records, model_path_str, queue):
    try:
        _require_prophet()
        import pandas as pd

        train_df = pd.DataFrame.from_records(train_records)
        future_df = pd.DataFrame.from_records(forecast_records)
        model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        try:
            model.add_country_holidays(country_name="UK")
        except Exception:
            pass
        model.fit(train_df[["ds", "y"]].copy())
        forecast = model.predict(future_df[["ds"]].copy())
        if model_path_str:
            model_path = Path(model_path_str)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with model_path.open("wb") as handle:
                pickle.dump(model, handle)
        queue.put({"predictions": [float(value) for value in forecast["yhat"].tolist()]})
    except OSError as exc:
        if getattr(exc, "winerror", None) == 5:
            queue.put({"error": "Prophet/CmdStan execution was denied by the Windows environment."})
        else:
            queue.put({"error": f"{type(exc).__name__}: {exc}"})
    except Exception as exc:
        queue.put({"error": f"{type(exc).__name__}: {exc}"})


def train_and_predict_prophet(train_df, forecast_df, model_path: str | Path | None = None, timeout_seconds: int = 300):
    _require_prophet()
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(
        target=_prophet_worker,
        args=(
            train_df[["ds", "y"]].to_dict("records"),
            forecast_df[["ds"]].to_dict("records"),
            str(model_path) if model_path is not None else None,
            queue,
        ),
    )
    try:
        process.start()
    except OSError as exc:
        if getattr(exc, "winerror", None) == 5:
            raise RuntimeError(
                "Prophet subprocess launch was denied in this Windows environment, so the model was skipped."
            ) from exc
        raise
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(30)
        raise TimeoutError(f"Prophet exceeded the {timeout_seconds}-second timeout in this environment.")
    if queue.empty():
        raise RuntimeError("Prophet worker finished without returning a result.")
    result = queue.get()
    if "error" in result:
        raise RuntimeError(result["error"])
    return result["predictions"]


def extract_feature_importances(model):
    if hasattr(model, "feature_importances_"):
        return list(model.feature_importances_)
    if hasattr(model, "get_feature_importance"):
        return list(model.get_feature_importance())
    if hasattr(model, "booster_") and hasattr(model.booster_, "feature_importance"):
        return list(model.booster_.feature_importance())
    return None


def predict_estimator(model, X):
    return model.predict(X)
