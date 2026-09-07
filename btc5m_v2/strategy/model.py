from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

FORBIDDEN_LABEL_FEATURES = frozenset(
    {
        "resolved",
        "winning_side",
        "label_up",
        "target_up",
        "selected_won",
        "selected_side_by_mid_won",
    }
)


def ensure_no_label_features(feature_columns: Sequence[str]) -> tuple[str, ...]:
    columns = tuple(str(value) for value in feature_columns)
    overlap = sorted(set(columns) & FORBIDDEN_LABEL_FEATURES)
    if overlap:
        raise ValueError(f"label leakage features are forbidden: {', '.join(overlap)}")
    return columns


@dataclass(frozen=True)
class LogisticProbabilityModel:
    feature_columns: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    model_version: str = "btc5m-logit-v1"

    def __post_init__(self) -> None:
        columns = ensure_no_label_features(self.feature_columns)
        if columns != self.feature_columns:
            object.__setattr__(self, "feature_columns", columns)
        length = len(self.feature_columns)
        if not length:
            raise ValueError("model requires at least one feature")
        if len(self.means) != length or len(self.scales) != length or len(self.coefficients) != length:
            raise ValueError("feature_columns, means, scales, and coefficients must have equal length")
        if any(scale <= 0 or not math.isfinite(scale) for scale in self.scales):
            raise ValueError("all feature scales must be positive finite values")
        if any(not math.isfinite(value) for value in self.means):
            raise ValueError("all feature means must be finite")
        if any(not math.isfinite(value) for value in self.coefficients):
            raise ValueError("all model coefficients must be finite")
        if not math.isfinite(float(self.intercept)):
            raise ValueError("model intercept must be finite")

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

    @staticmethod
    def _finite_number(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def missing_features(self, row: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(
            name
            for name in self.feature_columns
            if self._finite_number(row.get(name)) is None
        )

    def vectorize(self, row: Mapping[str, Any]) -> tuple[float, ...] | None:
        if self.missing_features(row):
            return None
        return tuple(float(row[name]) for name in self.feature_columns)

    def predict_up(self, row: Mapping[str, Any]) -> float | None:
        values = self.vectorize(row)
        if values is None:
            return None
        linear = float(self.intercept)
        for value, mean, scale, coefficient in zip(
            values,
            self.means,
            self.scales,
            self.coefficients,
            strict=True,
        ):
            linear += coefficient * ((value - mean) / scale)
        return self._sigmoid(linear)

    def to_payload(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "model_type": "standardized_logistic_regression",
            "feature_columns": list(self.feature_columns),
            "means": list(self.means),
            "scales": list(self.scales),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LogisticProbabilityModel":
        if str(payload.get("model_type") or "") != "standardized_logistic_regression":
            raise ValueError("unsupported model_type")
        return cls(
            feature_columns=tuple(str(value) for value in payload["feature_columns"]),
            means=tuple(float(value) for value in payload["means"]),
            scales=tuple(float(value) for value in payload["scales"]),
            coefficients=tuple(float(value) for value in payload["coefficients"]),
            intercept=float(payload["intercept"]),
            model_version=str(payload.get("model_version") or "btc5m-logit-v1"),
        )

    def save_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_payload(), indent=2), encoding="utf-8")
        return destination

    @classmethod
    def load_json(cls, path: str | Path) -> "LogisticProbabilityModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("model JSON must contain an object")
        return cls.from_payload(payload)
