from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency guard
    pd = None


def _require_pandas() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for preprocessing.")


@dataclass(frozen=True)
class SplitFrames:
    train: Any
    val: Any
    test: Any


def sort_by_date(df, date_col: str = "ds"):
    _require_pandas()
    frame = df.copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    return frame.sort_values(date_col).reset_index(drop=True)


def chronological_split(df, train_ratio: float, val_ratio: float, test_ratio: float, date_col: str = "ds") -> SplitFrames:
    _require_pandas()
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    frame = sort_by_date(df, date_col=date_col)
    n = len(frame)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return SplitFrames(
        train=frame.iloc[:train_end].copy(),
        val=frame.iloc[train_end:val_end].copy(),
        test=frame.iloc[val_end:].copy(),
    )


def chronological_holdout(df, train_ratio: float = 0.8, date_col: str = "ds"):
    _require_pandas()
    frame = sort_by_date(df, date_col=date_col)
    split_idx = int(len(frame) * train_ratio)
    return frame.iloc[:split_idx].copy(), frame.iloc[split_idx:].copy()


def supervised_matrix(df, target_col: str = "y", drop_cols: Iterable[str] = ("ds",)):
    _require_pandas()
    drop_cols = set(drop_cols) | {target_col}
    feature_cols = [col for col in df.columns if col not in drop_cols]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y, feature_cols


def split_supervised(df, target_col: str = "y", train_ratio: float = 0.8, date_col: str = "ds"):
    _require_pandas()
    frame = sort_by_date(df, date_col=date_col)
    split_idx = int(len(frame) * train_ratio)
    return frame.iloc[:split_idx].copy(), frame.iloc[split_idx:].copy()
