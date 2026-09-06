from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class V2Config:
    raw: dict[str, Any]

    @property
    def expected_resolution_source(self) -> str:
        return str(self.raw["resolution"]["expected_source"]).strip()

    @property
    def fail_if_resolution_source_missing(self) -> bool:
        return bool(self.raw["resolution"].get("fail_if_missing", True))

    @property
    def fail_if_resolution_source_unexpected(self) -> bool:
        return bool(self.raw["resolution"].get("fail_if_unexpected", True))

    @property
    def entry_min(self) -> float:
        return float(self.raw["entry"]["seconds_left_min"])

    @property
    def entry_max(self) -> float:
        return float(self.raw["entry"]["seconds_left_max"])

    @property
    def max_spread(self) -> float:
        return float(self.raw["market_quality"]["max_selected_side_spread"])

    @property
    def min_depth(self) -> float:
        return float(self.raw["market_quality"]["min_top_3_ask_notional_usd"])

    @property
    def clob_max_age_sec(self) -> float:
        return float(self.raw["freshness"]["clob_max_age_sec"])

    @property
    def max_consecutive_data_errors(self) -> int:
        return int(self.raw["freshness"]["max_consecutive_data_errors"])


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "btc_5m_v2.yaml"


def load_config(path: str | Path | None = None) -> V2Config:
    cfg_path = Path(path) if path is not None else default_config_path()
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("V2 config must be a YAML mapping")
    if int(data.get("version", 0)) != 2:
        raise ValueError("Expected V2 config version 2")
    resolution = data.get("resolution")
    if not isinstance(resolution, dict) or not str(resolution.get("expected_source") or "").strip():
        raise ValueError("V2 config requires resolution.expected_source")
    return V2Config(data)
