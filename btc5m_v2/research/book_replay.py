from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from btc5m_v2.feeds.polymarket_ws import BookState
from btc5m_v2.research.recorder import event_timestamp_ms


@dataclass(frozen=True)
class ReplayedBookSnapshot:
    ordinal: int
    received_ts_ns: int
    event_type: str
    event_exchange_ts_ms: int | None
    up_bids: tuple[tuple[float, float], ...]
    up_asks: tuple[tuple[float, float], ...]
    down_bids: tuple[tuple[float, float], ...]
    down_asks: tuple[tuple[float, float], ...]

    @property
    def up_best_bid(self) -> float | None:
        return self.up_bids[0][0] if self.up_bids else None

    @property
    def up_best_ask(self) -> float | None:
        return self.up_asks[0][0] if self.up_asks else None

    @property
    def down_best_bid(self) -> float | None:
        return self.down_bids[0][0] if self.down_bids else None

    @property
    def down_best_ask(self) -> float | None:
        return self.down_asks[0][0] if self.down_asks else None

    def asks(self, side: str) -> tuple[tuple[float, float], ...]:
        label = str(side).strip().upper()
        if label == "UP":
            return self.up_asks
        if label == "DOWN":
            return self.down_asks
        raise ValueError("side must be UP or DOWN")

    def bids(self, side: str) -> tuple[tuple[float, float], ...]:
        label = str(side).strip().upper()
        if label == "UP":
            return self.up_bids
        if label == "DOWN":
            return self.down_bids
        raise ValueError("side must be UP or DOWN")


def _sorted_bids(state: BookState) -> tuple[tuple[float, float], ...]:
    return tuple((price, state.bids[price]) for price in sorted(state.bids, reverse=True))


def _sorted_asks(state: BookState) -> tuple[tuple[float, float], ...]:
    return tuple((price, state.asks[price]) for price in sorted(state.asks))


def _raw_rows(path: str | Path) -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"raw CLOB row {line_number} is not an object")
        event = payload.get("event")
        if not isinstance(event, dict):
            raise ValueError(f"raw CLOB row {line_number} is missing event object")
        try:
            received_ts_ns = int(payload["received_ts_ns"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"raw CLOB row {line_number} has invalid received_ts_ns") from exc
        yield {
            "received_ts_ns": received_ts_ns,
            "event_type": str(payload.get("event_type") or event.get("event_type") or ""),
            "event": event,
        }


def reconstruct_book_snapshots(
    raw_path: str | Path,
    *,
    up_token_id: str,
    down_token_id: str,
) -> list[ReplayedBookSnapshot]:
    """Replay raw events in file order using the same BookState semantics as recording.

    File order matters: multiple websocket events can share one receive timestamp.
    Replaying solely by timestamp would leak later events from the same payload into
    an earlier snapshot.
    """

    up = BookState(str(up_token_id))
    down = BookState(str(down_token_id))
    snapshots: list[ReplayedBookSnapshot] = []

    for raw in _raw_rows(raw_path):
        received_ts_ns = int(raw["received_ts_ns"])
        event = raw["event"]
        up_changed = up.apply(event, received_ts_ns)
        down_changed = down.apply(event, received_ts_ns)
        if not (up_changed or down_changed):
            continue
        if up.best_bid is None or up.best_ask is None:
            continue
        if down.best_bid is None or down.best_ask is None:
            continue

        snapshots.append(
            ReplayedBookSnapshot(
                ordinal=len(snapshots),
                received_ts_ns=received_ts_ns,
                event_type=str(event.get("event_type") or ""),
                event_exchange_ts_ms=event_timestamp_ms(event),
                up_bids=_sorted_bids(up),
                up_asks=_sorted_asks(up),
                down_bids=_sorted_bids(down),
                down_asks=_sorted_asks(down),
            )
        )
    return snapshots


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def validate_snapshot_alignment(
    recorded_rows: Sequence[dict[str, Any]],
    replayed: Sequence[ReplayedBookSnapshot],
) -> None:
    """Fail closed if raw event replay cannot reproduce the recorded snapshot sequence."""

    if len(recorded_rows) != len(replayed):
        raise ValueError(
            f"book_snapshot_count_mismatch recorded={len(recorded_rows)} replayed={len(replayed)}"
        )

    for index, (recorded, book) in enumerate(zip(recorded_rows, replayed)):
        recorded_ts = int(recorded["received_ts_ns"])
        recorded_type = str(recorded.get("event_type") or "")
        recorded_exchange_ts = _optional_int(recorded.get("event_exchange_ts_ms"))
        if (
            recorded_ts != book.received_ts_ns
            or recorded_type != book.event_type
            or recorded_exchange_ts != book.event_exchange_ts_ms
        ):
            raise ValueError(
                "book_snapshot_alignment_mismatch "
                f"index={index} recorded_ts={recorded_ts} replayed_ts={book.received_ts_ns} "
                f"recorded_type={recorded_type!r} replayed_type={book.event_type!r}"
            )


def reconstruct_and_align_books(
    raw_path: str | Path,
    *,
    recorded_rows: Sequence[dict[str, Any]],
    up_token_id: str,
    down_token_id: str,
) -> list[ReplayedBookSnapshot]:
    replayed = reconstruct_book_snapshots(
        raw_path,
        up_token_id=up_token_id,
        down_token_id=down_token_id,
    )
    validate_snapshot_alignment(recorded_rows, replayed)
    return replayed
