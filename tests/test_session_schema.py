"""Tests for the canonical session schema and its writer.

The schema exists because five historical writers produced five per-sample
column sets, and 'timestamp' meant wall-clock time in two of them and
seconds-since-start in a third. These tests pin the contract.
"""
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from session_schema import (  # noqa: E402
    COLUMNS,
    SCHEMA_VERSION,
    SessionMetadata,
    SessionWriter,
    row_from,
)


@dataclass
class FakeComfort:
    overall_comfort: float = 82.5
    posture_quality: float = 88.0
    ergonomic_risk: float = 15.0
    fatigue_indicator: float = 4.0
    comfort_category: str = "Good"


@dataclass
class FakeQuality:
    overall_score: float = 77.5
    risk_level: str = "Low"


ANGLES = {
    "trunk_signed": -12.34,
    "neck_signed": 5.67,
    "trunk_from_vertical": 12.34,
    "neck_from_vertical": 5.67,
    "left_hip_angle": 101.2,
    "right_hip_angle": 99.8,
    "left_knee_angle": 118.4,
    "right_knee_angle": 117.9,
    "left_shoulder_elev": 14.2,
    "right_shoulder_elev": 15.1,
}


def read_csv(path) -> (List[str], List[Dict[str, str]]):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


# --------------------------------------------------------------------------
# Column contract
# --------------------------------------------------------------------------

def test_timestamp_and_t_sec_are_distinct_columns():
    """The whole point: one is a clock, the other is elapsed seconds."""
    assert "timestamp" in COLUMNS
    assert "t_sec" in COLUMNS


def test_schema_version_is_the_first_column():
    assert COLUMNS[0] == "schema_version"


def test_columns_are_unique():
    assert len(COLUMNS) == len(set(COLUMNS))


def test_row_from_fills_every_column():
    row = row_from(t_sec=1.0)
    assert set(row) == set(COLUMNS)


def test_unknown_fields_are_empty_not_missing():
    """Uniform width means pandas never produces a ragged frame."""
    row = row_from(t_sec=1.0, angles={"trunk_signed": -5.0})
    assert row["comfort_score"] == ""
    assert row["quality_score"] == ""
    assert row["hip_L"] == ""


def test_timestamp_is_absolute_and_t_sec_is_relative():
    row = row_from(t_sec=42.5)
    assert row["t_sec"] == 42.5
    # Parses as a real datetime, i.e. is not elapsed seconds.
    datetime.fromisoformat(row["timestamp"])


def test_signed_and_magnitude_columns_both_written():
    row = row_from(t_sec=1.0, angles=ANGLES)
    assert row["trunk_signed"] == pytest.approx(-12.3)
    assert row["trunk_deg"] == pytest.approx(12.3)


def test_comfort_and_quality_models_coexist():
    row = row_from(t_sec=1.0, angles=ANGLES,
                   comfort=FakeComfort(), quality=FakeQuality())
    assert row["comfort_score"] == pytest.approx(82.5)
    assert row["comfort_category"] == "Good"
    assert row["quality_score"] == pytest.approx(77.5)
    assert row["risk_level"] == "Low"


def test_missing_lean_side_becomes_empty_string_not_none():
    row = row_from(t_sec=1.0, extras={"lean_side": None, "lower_visible": True})
    assert row["lean_side"] == ""
    assert row["lower_visible"] is True


# --------------------------------------------------------------------------
# SessionWriter
# --------------------------------------------------------------------------

def test_writer_emits_the_canonical_header(tmp_path):
    path = tmp_path / "driver_comfort_20250101_120000.csv"
    SessionWriter(path)
    header, _ = read_csv(path)
    assert header == COLUMNS


def test_writer_rows_match_header_width(tmp_path):
    path = tmp_path / "s_20250101_120000.csv"
    w = SessionWriter(path)
    w.write(t_sec=1.0, angles=ANGLES, body_state="Sitting", comfort=FakeComfort())
    w.write(t_sec=2.0, angles=ANGLES, body_state="Sitting", quality=FakeQuality())
    header, rows = read_csv(path)
    assert len(rows) == 2
    for row in rows:
        assert len(row) == len(header)
        assert int(row["schema_version"]) == SCHEMA_VERSION


def test_writer_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "deeper" / "s_20250101_120000.csv"
    SessionWriter(path)
    assert path.exists()


def test_writer_truncates_an_existing_file(tmp_path):
    path = tmp_path / "s_20250101_120000.csv"
    SessionWriter(path).write(t_sec=1.0)
    SessionWriter(path)  # reopening starts a fresh session
    _, rows = read_csv(path)
    assert rows == []


# --------------------------------------------------------------------------
# Metadata sidecar
# --------------------------------------------------------------------------

def test_sidecar_is_written_on_construction(tmp_path):
    """Written up front so a session is self-describing even if it crashes."""
    path = tmp_path / "driver_comfort_20250101_120000.csv"
    meta = SessionMetadata(session_id="s", started_at="2025-01-01T12:00:00",
                           app="test", seat_id="seat-A", driver_id="d1")
    SessionWriter(path, metadata=meta)
    sidecar = SessionMetadata.sidecar_path(path)
    assert sidecar.exists()
    assert json.loads(sidecar.read_text())["seat_id"] == "seat-A"


def test_sidecar_round_trips(tmp_path):
    path = tmp_path / "s_20250101_120000.csv"
    meta = SessionMetadata(session_id="s", started_at="2025-01-01T12:00:00",
                           app="test", seat_id="seat-B", vehicle="XUV700",
                           camera_position="left-side", forward_is_image_right=True)
    meta.save(path)
    loaded = SessionMetadata.load(path)
    assert loaded.seat_id == "seat-B"
    assert loaded.vehicle == "XUV700"
    assert loaded.forward_is_image_right is True


def test_load_returns_none_when_there_is_no_sidecar(tmp_path):
    assert SessionMetadata.load(tmp_path / "nope_20250101_120000.csv") is None


def test_unknown_sidecar_fields_are_preserved_in_extra(tmp_path):
    """Forward compatibility: a newer writer's fields must not crash loading."""
    path = tmp_path / "s_20250101_120000.csv"
    SessionMetadata.sidecar_path(path).write_text(json.dumps({
        "session_id": "s", "started_at": "", "app": "test",
        "seat_id": "seat-C", "some_future_field": 7,
    }))
    loaded = SessionMetadata.load(path)
    assert loaded.seat_id == "seat-C"
    assert loaded.extra["some_future_field"] == 7


def test_sidecar_path_sits_next_to_the_csv(tmp_path):
    path = tmp_path / "driver_comfort_20250101_120000.csv"
    assert SessionMetadata.sidecar_path(path).name == \
        "driver_comfort_20250101_120000.meta.json"


# --------------------------------------------------------------------------
# The writer's output must be readable by the analytics reader
# --------------------------------------------------------------------------

def test_written_session_is_readable_by_posture_analytics(tmp_path):
    from posture_analytics import PostureAnalytics

    path = tmp_path / "driver_comfort_20250101_120000.csv"
    w = SessionWriter(path)
    for i in range(1, 6):
        w.write(t_sec=float(i), angles=ANGLES, body_state="Sitting",
                upper_body="Neutral", comfort=FakeComfort())

    analytics = PostureAnalytics(data_dir=str(tmp_path))
    df = analytics.load_all_sessions()
    assert len(df) == 5
    assert df["timestamp"].notna().all()

    analysis = analytics.analyze_session(path.name)
    assert analysis["total_samples"] == 5
    assert analysis["duration_minutes"] > 0
