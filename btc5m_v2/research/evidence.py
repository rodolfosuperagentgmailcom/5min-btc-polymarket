from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


def load_paper_journal(root: str | Path) -> list[dict[str, Any]]:
    """Load paper journal entries in deterministic chronological order."""
    journal_root = Path(root)
    rows: list[dict[str, Any]] = []
    for path in sorted(journal_root.glob("*/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        payload = dict(payload)
        payload["_journal_path"] = str(path)
        rows.append(payload)
    rows.sort(
        key=lambda row: (
            int(row.get("signal_ts_ns") or 0),
            str(row.get("slug") or ""),
            str(row.get("_journal_path") or ""),
        )
    )
    return rows


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _max_drawdown(pnls: Iterable[float]) -> float:
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for pnl in pnls:
        equity += float(pnl)
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return drawdown


def _side_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = {"UP": 0, "DOWN": 0}
    for row in rows:
        side = str(row.get("side") or "").upper()
        if side in counts:
            counts[side] += 1
    return counts


def summarize_paper_evidence(
    rows: Iterable[dict[str, Any]],
    *,
    target_settled_signals: int = 500,
) -> dict[str, Any]:
    entries = list(rows)
    paper_entries = [row for row in entries if str(row.get("mode") or "").lower() == "paper"]
    settled = [row for row in paper_entries if str(row.get("status") or "").upper() == "SETTLED"]
    open_rows = [row for row in paper_entries if str(row.get("status") or "").upper() == "OPEN"]

    # A forward-paper evidence record must prove that it never used a live order path.
    unsafe = [
        row
        for row in paper_entries
        if bool(row.get("order_path_used")) or bool(row.get("credentials_loaded"))
    ]

    settled_by_slug: dict[str, dict[str, Any]] = {}
    duplicate_slugs: list[str] = []
    for row in settled:
        slug = str(row.get("slug") or "").strip()
        if not slug:
            continue
        if slug in settled_by_slug:
            duplicate_slugs.append(slug)
        settled_by_slug[slug] = row

    independent = sorted(
        settled_by_slug.values(),
        key=lambda row: int(row.get("signal_ts_ns") or 0),
    )

    pnl_values: list[float] = []
    cost_values: list[float] = []
    fee_values: list[float] = []
    wins = 0
    for row in independent:
        pnl = _finite(row.get("net_pnl_usd"))
        cost = _finite(row.get("all_in_cost_usd"))
        fee = _finite(row.get("fee_usd"))
        if pnl is not None:
            pnl_values.append(pnl)
            if pnl > 0:
                wins += 1
        if cost is not None and cost >= 0:
            cost_values.append(cost)
        if fee is not None and fee >= 0:
            fee_values.append(fee)

    count = len(independent)
    net_pnl = sum(pnl_values)
    total_cost = sum(cost_values)
    total_fees = sum(fee_values)
    losses = count - wins
    target = max(1, int(target_settled_signals))
    progress = min(1.0, count / target)

    model_versions = sorted(
        {str(row.get("model_version")) for row in independent if row.get("model_version")}
    )

    return {
        "mode": "forward-paper-evidence",
        "live_execution_approved": False,
        "target_settled_signals": target,
        "settled_independent_signals": count,
        "remaining_to_target": max(0, target - count),
        "progress_fraction": progress,
        "evidence_target_reached": count >= target,
        "open_positions": len(open_rows),
        "wins": wins,
        "losses": losses,
        "win_rate": None if count == 0 else wins / count,
        "net_pnl_usd": net_pnl,
        "mean_pnl_per_signal_usd": None if count == 0 else net_pnl / count,
        "total_all_in_cost_usd": total_cost,
        "return_on_cost": None if total_cost <= 0 else net_pnl / total_cost,
        "fees_paid_usd": total_fees,
        "max_drawdown_usd": _max_drawdown(pnl_values),
        "side_counts": _side_counts(independent),
        "model_versions": model_versions,
        "unsafe_journal_entries": len(unsafe),
        "duplicate_settled_slugs": sorted(set(duplicate_slugs)),
        "evidence_integrity_ok": len(unsafe) == 0 and len(duplicate_slugs) == 0,
    }


def evidence_report(
    journal_root: str | Path,
    *,
    target_settled_signals: int = 500,
) -> dict[str, Any]:
    rows = load_paper_journal(journal_root)
    report = summarize_paper_evidence(
        rows,
        target_settled_signals=target_settled_signals,
    )
    report["journal_root"] = str(Path(journal_root))
    report["journal_entries_seen"] = len(rows)
    return report
