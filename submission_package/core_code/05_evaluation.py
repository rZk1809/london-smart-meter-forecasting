from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency guard
    pd = None


def _to_float_list(values: Sequence[float] | Iterable[float]) -> list[float]:
    return [float(value) for value in values]


def rmse(y_true, y_pred) -> float:
    actual = _to_float_list(y_true)
    predicted = _to_float_list(y_pred)
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / max(len(actual), 1))


def mae(y_true, y_pred) -> float:
    actual = _to_float_list(y_true)
    predicted = _to_float_list(y_pred)
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / max(len(actual), 1)


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    actual = _to_float_list(y_true)
    predicted = _to_float_list(y_pred)
    values = [abs((a - p) / max(abs(a), eps)) for a, p in zip(actual, predicted)]
    return 100.0 * sum(values) / max(len(values), 1)


def r2(y_true, y_pred) -> float:
    actual = _to_float_list(y_true)
    predicted = _to_float_list(y_pred)
    if not actual:
        return float("nan")
    mean_actual = sum(actual) / len(actual)
    ss_tot = sum((a - mean_actual) ** 2 for a in actual)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    actual = _to_float_list(y_true)
    predicted = _to_float_list(y_pred)
    values = [abs(p - a) / max(abs(a) + abs(p), eps) for a, p in zip(actual, predicted)]
    return 200.0 * sum(values) / max(len(values), 1)


def medae(y_true, y_pred) -> float:
    actual = _to_float_list(y_true)
    predicted = _to_float_list(y_pred)
    return float(median(abs(a - p) for a, p in zip(actual, predicted)))


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    actual = _to_float_list(y_true)
    predicted = _to_float_list(y_pred)
    metrics = {
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "MAPE": mape(actual, predicted),
        "R2": r2(actual, predicted),
        "SMAPE": smape(actual, predicted),
        "MedAE": medae(actual, predicted),
    }
    if len(actual) > 1:
        true_direction = [1 if actual[i + 1] - actual[i] >= 0 else -1 for i in range(len(actual) - 1)]
        pred_direction = [1 if predicted[i + 1] - predicted[i] >= 0 else -1 for i in range(len(predicted) - 1)]
        metrics["Direction_Accuracy"] = 100.0 * sum(
            1 for t, p in zip(true_direction, pred_direction) if t == p
        ) / max(len(true_direction), 1)
    return metrics


def build_result_row(
    model_name: str,
    y_true,
    y_pred,
    train_seconds: float | None = None,
    inference_seconds: float | None = None,
) -> dict[str, float | str | None]:
    row = {"Model": model_name, **calculate_metrics(y_true, y_pred)}
    row["Train_Time"] = train_seconds
    row["Inference_Time"] = inference_seconds
    return row


def sort_results(rows: list[dict]) -> list[dict]:
    def _sort_key(row):
        try:
            return float(row.get("RMSE", float("inf")))
        except Exception:
            return float("inf")

    return sorted(rows, key=_sort_key)


def save_results_table(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if pd is not None:
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path
