from __future__ import annotations

import math
from typing import Iterable

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency guard
    pd = None


DEFAULT_LAGS = (1, 2, 3, 7, 14, 21, 28, 30, 60, 90)
DEFAULT_WINDOWS = (3, 7, 14, 21, 30)


def _require_pandas() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for feature engineering.")


def add_calendar_features(df, date_col: str = "ds"):
    _require_pandas()
    frame = df.copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame["dayofweek"] = frame[date_col].dt.dayofweek
    frame["month"] = frame[date_col].dt.month
    frame["quarter"] = frame[date_col].dt.quarter
    frame["year"] = frame[date_col].dt.year
    frame["dayofyear"] = frame[date_col].dt.dayofyear
    frame["weekofyear"] = frame[date_col].dt.isocalendar().week.astype("int64")
    frame["is_weekend"] = (frame["dayofweek"] >= 5).astype("int64")
    frame["month_sin"] = (2 * math.pi * frame["month"] / 12).apply(math.sin)
    frame["month_cos"] = (2 * math.pi * frame["month"] / 12).apply(math.cos)
    frame["dow_sin"] = (2 * math.pi * frame["dayofweek"] / 7).apply(math.sin)
    frame["dow_cos"] = (2 * math.pi * frame["dayofweek"] / 7).apply(math.cos)
    return frame


def add_lag_features(df, target_col: str = "y", lags: Iterable[int] = DEFAULT_LAGS):
    _require_pandas()
    frame = df.copy()
    for lag in lags:
        frame[f"lag_{lag}"] = frame[target_col].shift(lag)
    return frame


def add_rolling_features(df, target_col: str = "y", windows: Iterable[int] = DEFAULT_WINDOWS):
    _require_pandas()
    frame = df.copy()
    history = frame[target_col].shift(1)
    for window in windows:
        rolling = history.rolling(window=window)
        frame[f"rolling_mean_{window}"] = rolling.mean()
        frame[f"rolling_std_{window}"] = rolling.std()
        frame[f"rolling_min_{window}"] = rolling.min()
        frame[f"rolling_max_{window}"] = rolling.max()
        frame[f"rolling_median_{window}"] = rolling.median()
    return frame


def add_change_features(df, target_col: str = "y"):
    _require_pandas()
    frame = df.copy()
    history = frame[target_col].shift(1)
    frame["diff_1"] = history.diff(1)
    frame["diff_7"] = history.diff(7)
    frame["diff_30"] = history.diff(30)
    frame["pct_change_1"] = history.pct_change(1)
    frame["pct_change_7"] = history.pct_change(7)
    return frame


def build_feature_frame(df, target_col: str = "y"):
    _require_pandas()
    frame = add_calendar_features(df)
    frame = add_lag_features(frame, target_col=target_col)
    frame = add_rolling_features(frame, target_col=target_col)
    frame = add_change_features(frame, target_col=target_col)
    return frame.dropna().reset_index(drop=True)


def feature_columns(df, target_col: str = "y", date_col: str = "ds") -> list[str]:
    return [col for col in df.columns if col not in {date_col, target_col}]
