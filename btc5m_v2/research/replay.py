from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from btc5m_v2.research.btc_features import BTCSample, attach_btc_features
from btc5m_v2.research.features import (
    FeatureRow,
    attach_resolution_labels,
    build_causal_features,
    build_feature_row,
)


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


def read_btc_samples_parquet(
    path: str | Path,
    *,
    timestamp_column: str = "ts_ns",
    price_column: str = "price",
) -> list[BTCSample]:
    table = pq.read_table(path, columns=[timestamp_column, price_column])
    samples: list[BTCSample] = []
    for row in table.to_pylist():
        ts_ns = row.get(timestamp_column)
        price = row.get(price_column)
        if ts_ns is None or price is None:
            continue
        samples.append(BTCSample(ts_ns=int(ts_ns), price=float(price)))
    samples.sort(key=lambda sample: sample.ts_ns)
    return samples


def build_feature_dataset_rows(
    market_dir: str | Path,
    *,
    seconds_left_min: float | None = None,
    seconds_left_max: float | None = None,
    btc_samples: Iterable[BTCSample | tuple[int, float]] | None = None,
    btc_reference_price: float | None = None,
) -> list[dict[str, Any]]:
    """Build causal research rows and attach terminal labels only at the end.

    Feature construction never reads resolution.json. The final result is loaded
    only after all timestamp-local predictors have been calculated, which keeps
    the replay dataset safe from direct label leakage.
    """

    raw_rows = list(iter_snapshot_rows(market_dir))
    feature_rows = build_causal_features(raw_rows)

    filtered: list[dict[str, Any]] = []
    for row in feature_rows:
        seconds_left = float(row.get("seconds_left") or 0.0)
        if seconds_left_min is not None and seconds_left < seconds_left_min:
            continue
        if seconds_left_max is not None and seconds_left > seconds_left_max:
            continue
        filtered.append(row)

    if btc_samples is not None:
        filtered = attach_btc_features(
            filtered,
            samples=btc_samples,
            reference_price=btc_reference_price,
        )

    resolved, winning_side = read_resolution(market_dir)
    if resolved and winning_side in {"UP", "DOWN"}:
        labeled = attach_resolution_labels(filtered, winning_side=winning_side)
        for row in labeled:
            row["resolved"] = True
        return labeled

    unresolved_rows: list[dict[str, Any]] = []
    for feature in filtered:
        row = dict(feature)
        row.update(
            {
                "resolved": False,
                "winning_side": None,
                "target_up": None,
                "selected_side_by_mid_won": None,
            }
        )
        unresolved_rows.append(row)
    return unresolved_rows


def write_feature_dataset(
    market_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    seconds_left_min: float | None = None,
    seconds_left_max: float | None = None,
    btc_samples: Iterable[BTCSample | tuple[int, float]] | None = None,
    btc_reference_price: float | None = None,
) -> Path:
    root = Path(market_dir)
    destination = Path(output_path) if output_path is not None else root / "features.parquet"
    rows = build_feature_dataset_rows(
        root,
        seconds_left_min=seconds_left_min,
        seconds_left_max=seconds_left_max,
        btc_samples=btc_samples,
        btc_reference_price=btc_reference_price,
    )
    if not rows:
        raise ValueError("no replay rows available for requested market/window")

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, destination, compression="zstd")
    return destination
