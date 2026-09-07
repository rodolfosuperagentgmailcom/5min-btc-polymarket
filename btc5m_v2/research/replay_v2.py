from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from btc5m_v2.research.features import attach_resolution_labels, build_causal_features


@dataclass(frozen=True)
class ReplayBuildResult:
    market_dir: Path
    output_path: Path
    rows: int
    winning_side: str | None
    resolved: bool


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def load_snapshot_rows(market_dir: str | Path) -> list[dict[str, Any]]:
    directory = Path(market_dir)
    parts = sorted(directory.glob("clob_snapshots_part-*.parquet"))
    if not parts:
        return []

    rows: list[dict[str, Any]] = []
    for path in parts:
        rows.extend(pq.read_table(path).to_pylist())
    rows.sort(key=lambda row: int(row["received_ts_ns"]))
    return rows


def load_resolution(market_dir: str | Path) -> dict[str, Any] | None:
    directory = Path(market_dir)
    resolution_path = directory / "resolution.json"
    if resolution_path.exists():
        payload = _read_json(resolution_path)
        return payload if payload.get("resolved") is True else None

    metadata_path = directory / "metadata.json"
    if not metadata_path.exists():
        return None
    metadata = _read_json(metadata_path)
    resolution = metadata.get("resolution")
    if isinstance(resolution, dict) and resolution.get("resolved") is True:
        return resolution
    return None


def build_market_replay(
    market_dir: str | Path,
    *,
    output_name: str = "replay_features.parquet",
    require_resolved: bool = True,
) -> ReplayBuildResult:
    directory = Path(market_dir)
    snapshots = load_snapshot_rows(directory)
    if not snapshots:
        raise ValueError(f"no CLOB snapshot parquet files found in {directory}")

    features = build_causal_features(snapshots)
    resolution = load_resolution(directory)
    winning_side: str | None = None
    resolved = resolution is not None

    if resolution is not None:
        raw_side = resolution.get("winning_side")
        winning_side = None if raw_side is None else str(raw_side).strip().upper()
        if winning_side not in {"UP", "DOWN"}:
            raise ValueError(f"resolved market has invalid winning_side in {directory}")
        features = attach_resolution_labels(features, winning_side=winning_side)
    elif require_resolved:
        raise ValueError(f"market is not resolved: {directory}")

    output_path = directory / output_name
    table = pa.Table.from_pylist(features)
    pq.write_table(table, output_path, compression="zstd")

    return ReplayBuildResult(
        market_dir=directory,
        output_path=output_path,
        rows=len(features),
        winning_side=winning_side,
        resolved=resolved,
    )


def build_output_root(
    output_root: str | Path,
    *,
    require_resolved: bool = True,
) -> list[ReplayBuildResult]:
    root = Path(output_root)
    results: list[ReplayBuildResult] = []
    for metadata_path in sorted(root.glob("*/*/metadata.json")):
        market_dir = metadata_path.parent
        try:
            result = build_market_replay(market_dir, require_resolved=require_resolved)
        except ValueError:
            if require_resolved:
                continue
            raise
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Build causal V2 replay feature datasets")
    parser.add_argument("path", help="Market directory or recorder output root")
    parser.add_argument("--root", action="store_true", help="Treat path as an output root")
    parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="Build features without terminal labels for unresolved markets",
    )
    args = parser.parse_args()

    if args.root:
        results = build_output_root(args.path, require_resolved=not args.allow_unresolved)
        payload: Any = [
            {
                "market_dir": str(result.market_dir),
                "output_path": str(result.output_path),
                "rows": result.rows,
                "winning_side": result.winning_side,
                "resolved": result.resolved,
            }
            for result in results
        ]
    else:
        result = build_market_replay(args.path, require_resolved=not args.allow_unresolved)
        payload = {
            "market_dir": str(result.market_dir),
            "output_path": str(result.output_path),
            "rows": result.rows,
            "winning_side": result.winning_side,
            "resolved": result.resolved,
        }

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
