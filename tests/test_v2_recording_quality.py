from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from btc5m_v2.research.dataset import build_market_dataset_rows
from btc5m_v2.research.train_model import load_dataset_rows


def test_reconnected_market_is_rejected_before_dataset_build(tmp_path):
    market_dir = tmp_path / "2030-01-01" / "btc-updown-5m-reconnected"
    market_dir.mkdir(parents=True)
    (market_dir / "metadata.json").write_text(
        json.dumps({"reconnects": 1}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="recording_reconnected:1"):
        build_market_dataset_rows(market_dir)


def test_dataset_loader_requires_recording_quality_column(tmp_path):
    path = tmp_path / "old-dataset.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"slug": "btc-updown-5m-old", "label_up": 1}]),
        path,
    )

    with pytest.raises(ValueError, match="dataset_missing_recording_quality"):
        load_dataset_rows(path)


def test_dataset_loader_rejects_reconnected_rows(tmp_path):
    path = tmp_path / "bad-dataset.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "slug": "btc-updown-5m-bad",
                    "label_up": 1,
                    "recording_reconnects": 2,
                }
            ]
        ),
        path,
    )

    with pytest.raises(ValueError, match="dataset_contains_reconnected_recording"):
        load_dataset_rows(path)


def test_dataset_loader_accepts_contiguous_rows(tmp_path):
    path = tmp_path / "good-dataset.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "slug": "btc-updown-5m-good",
                    "label_up": 0,
                    "recording_reconnects": 0,
                }
            ]
        ),
        path,
    )

    rows = load_dataset_rows(path)
    assert len(rows) == 1
    assert rows[0]["recording_reconnects"] == 0
