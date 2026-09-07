from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow.parquet as pq

from btc5m_v2.research.features import FeatureRow, build_feature_row


@dataclass(frozen=True)
class ReplayRow:
    feature: FeatureRow
    winning_side: str | None
    resolved: bool

    @property
    def label_up(self) -> int | None:
        if not self.resolved or self.winning_side not in {"UP", "DOWN"}:
            return None
        return 1 if self.winning_side == "UP" else 0

    @property
    def selected_won(self) -> bool | None:
        if not self.resolved or self.feature.selected_side is None or self.winning_side is None:
            return None
        return self.feature.selected_side == self.winning_side


def read_resolution(market_dir: str | Path) -> tuple[bool, str | None]:
    root = Path(market_dir)
    candidates = [root / "resolution.json", root / "metadata.json"]
    for path in candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if path.name == "metadata.json":
            payload = payload.get("resolution") or {}
        resolved = bool(payload.get("resolved") is True)
        side = str(payload.get("winning_side") or "").upper() or None
        if resolved:
            return True, side if side in {"UP", "DOWN"} else None
    return False, None


def iter_snapshot_rows(market_dir: str | Path) -> Iterator[dict[str, Any]]:
    root = Path(market_dir)
    paths = sorted(root.glob("clob_snapshots_part-*.parquet"))
    for path in paths:
        table = pq.read_table(path)
        for row in table.to_pylist():
            yield row


def iter_replay_rows(
    market_dir: str | Path,
    *,
    seconds_left_min: float | None = None,
    seconds_left_max: float | None = None,
) -> Iterator[ReplayRow]:
    resolved, winning_side = read_resolution(market_dir)
    for raw in iter_snapshot_rows(market_dir):
        feature = build_feature_row(raw)
        if seconds_left_min is not None and feature.seconds_left < seconds_left_min:
            continue
        if seconds_left_max is not None and feature.seconds_left > seconds_left_max:
            continue
        yield ReplayRow(feature=feature, winning_side=winning_side, resolved=resolved)


def load_replay_rows(
    market_dirs: Iterable[str | Path],
    *,
    seconds_left_min: float | None = None,
    seconds_left_max: float | None = None,
) -> list[ReplayRow]:
    rows: list[ReplayRow] = []
    for market_dir in market_dirs:
        rows.extend(
            iter_replay_rows(
                market_dir,
                seconds_left_min=seconds_left_min,
                seconds_left_max=seconds_left_max,
            )
        )
    rows.sort(key=lambda item: item.feature.received_ts_ns)
    return rows
