from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import AppConfig, get_default_config
from .utils import ensure_dir, exists_nonempty

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency guard
    pd = None


DATE_COL_CANDIDATES = ("DateTime", "Date", "datetime", "timestamp")
TARGET_COL_CANDIDATES = (
    "KWH/hh (per half hour) ",
    "KWH/hh (per half hour)",
    "KWH",
    "kwh",
    "consumption",
)


def _require_pandas() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for data loading and aggregation.")


def detect_columns(file_path: str | Path) -> tuple[str, str]:
    _require_pandas()
    header = pd.read_csv(file_path, nrows=0)
    cols = list(header.columns)
    normalized = {col.strip().lower(): col for col in cols}

    date_col = None
    for candidate in DATE_COL_CANDIDATES:
        key = candidate.strip().lower()
        if key in normalized:
            date_col = normalized[key]
            break

    target_col = None
    for candidate in TARGET_COL_CANDIDATES:
        key = candidate.strip().lower()
        if key in normalized:
            target_col = normalized[key]
            break

    if date_col is None:
        for col in cols:
            lowered = col.strip().lower()
            if "date" in lowered and "time" in lowered:
                date_col = col
                break

    if target_col is None:
        for col in cols:
            if "kwh" in col.strip().lower():
                target_col = col
                break

    if date_col is None or target_col is None:
        raise ValueError(f"Could not detect required columns from: {cols}")

    return date_col, target_col


def _resolve_raw_source(config: AppConfig, raw_path: str | Path | None = None) -> Path:
    candidate = Path(raw_path) if raw_path is not None else config.raw_csv
    if exists_nonempty(candidate):
        return candidate
    if exists_nonempty(config.legacy_raw_csv):
        return config.legacy_raw_csv
    raise FileNotFoundError(
        "Could not find the raw CSV in data/raw or the legacy root location."
    )


def load_daily_dataset(
    config: AppConfig | None = None,
    refresh: bool = False,
    raw_path: str | Path | None = None,
) -> Any:
    _require_pandas()
    config = config or get_default_config()
    processed_path = config.daily_parquet

    if exists_nonempty(processed_path) and not refresh:
        df = pd.read_parquet(processed_path)
        if "ds" in df.columns:
            df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        return df.sort_values("ds").reset_index(drop=True)

    raw_csv = _resolve_raw_source(config, raw_path=raw_path)
    return build_daily_aggregate(raw_csv, processed_path)


def build_daily_aggregate(
    raw_csv: str | Path,
    cache_path: str | Path,
    chunksize: int = 2_000_000,
) -> Any:
    _require_pandas()
    raw_csv = Path(raw_csv)
    cache_path = Path(cache_path)
    ensure_dir(cache_path.parent)

    date_col, target_col = detect_columns(raw_csv)
    aggregated = None

    reader = pd.read_csv(
        raw_csv,
        usecols=[date_col, target_col],
        dtype={date_col: "string"},
        na_values=["Null", "null", "NULL", "nan", "NaN"],
        chunksize=chunksize,
        engine="c",
        low_memory=True,
        memory_map=True,
    )

    for chunk in reader:
        chunk[target_col] = pd.to_numeric(chunk[target_col], errors="coerce")
        chunk = chunk.dropna(subset=[date_col, target_col])
        chunk["ds"] = chunk[date_col].str.slice(0, 10)
        daily = chunk.groupby("ds", sort=False)[target_col].sum()
        aggregated = daily if aggregated is None else aggregated.add(daily, fill_value=0)

    daily_df = aggregated.sort_index().reset_index()
    daily_df.columns = ["ds", "y"]
    daily_df["ds"] = pd.to_datetime(daily_df["ds"], errors="coerce")
    daily_df = daily_df.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)
    daily_df["y"] = daily_df["y"].astype("float64")

    try:
        daily_df.to_parquet(cache_path, index=False)
    except Exception:
        daily_df.to_csv(cache_path.with_suffix(".csv"), index=False)

    return daily_df


def validate_daily_dataset(df: Any) -> dict[str, Any]:
    _require_pandas()
    if df is None or len(df) == 0:
        return {"valid": False, "reason": "empty frame"}

    frame = df.copy()
    if "ds" in frame.columns:
        frame["ds"] = pd.to_datetime(frame["ds"], errors="coerce")
    metrics = {
        "valid": True,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "date_min": frame["ds"].min() if "ds" in frame.columns else None,
        "date_max": frame["ds"].max() if "ds" in frame.columns else None,
        "missing_values": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
    }
    if "y" in frame.columns:
        metrics["target_min"] = float(frame["y"].min())
        metrics["target_max"] = float(frame["y"].max())
        metrics["target_mean"] = float(frame["y"].mean())
    return metrics
