from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_recording_reconnects(market_dir: str | Path, *, require_field: bool = True) -> int:
    """Return recorder reconnect count, failing closed on malformed quality metadata."""

    metadata_path = Path(market_dir) / "metadata.json"
    if not metadata_path.exists():
        raise ValueError("missing_market_metadata")
    payload: Any = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("invalid_market_metadata")
    if "reconnects" not in payload:
        if require_field:
            raise ValueError("missing_recording_quality")
        return 0
    try:
        reconnects = int(payload["reconnects"])
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_recording_reconnects") from exc
    if reconnects < 0:
        raise ValueError("invalid_recording_reconnects")
    return reconnects


def require_contiguous_recording(market_dir: str | Path) -> None:
    reconnects = read_recording_reconnects(market_dir, require_field=True)
    if reconnects != 0:
        raise ValueError(f"recording_reconnected:{reconnects}")
