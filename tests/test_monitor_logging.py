"""Tests for EnhancedPostureMonitor's CSV logging cadence and schema.

Exercises _log_data directly with synthetic results, so no camera is needed.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posture_monitor_enhanced import LOG_INTERVAL_S, EnhancedPostureMonitor  # noqa: E402
from posture_scoring import PostureScore  # noqa: E402

ANGLE_KEYS = ("trunk_from_vertical", "neck_from_vertical", "left_knee_angle",
              "right_knee_angle", "left_hip_angle", "right_hip_angle",
              "trunk_signed", "neck_signed")


def results(trunk_signed=12.0):
    return {
        "body_state": "Sitting",
        "upper_body": "Neutral",
        "extras": {"lower_visible": True, "lean_side": ""},
        "angles": {k: 100.0 for k in ANGLE_KEYS} | {
            "trunk_from_vertical": abs(trunk_signed),
            "trunk_signed": trunk_signed,
            "neck_signed": 5.0,
            "neck_from_vertical": 5.0,
        },
        "quality_score": PostureScore(
            overall_score=80.0,
            category_scores={"upper_body": 80.0, "lower_body": 80.0,
                             "shoulders": 80.0, "symmetry": 80.0},
            risk_level="Low",
            recommendations=[],
            duration_warnings=[],
        ),
    }


@pytest.fixture
def monitor(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = EnhancedPostureMonitor()
    m.start_session()
    return m


def read_rows(path):
    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    return lines[0].split(","), lines[1:]


def test_session_log_header_carries_the_signed_columns(monitor):
    header, _ = read_rows(monitor.session_log_path)
    assert "trunk_signed" in header
    assert "neck_signed" in header
    # Magnitude columns are retained for existing readers.
    assert "trunk_deg" in header and "neck_deg" in header


def test_signed_values_are_written(monitor):
    monitor.session_start = __import__("time").monotonic()
    monitor._log_data(results(trunk_signed=-22.0))
    header, rows = read_rows(monitor.session_log_path)
    assert len(rows) == 1
    row = dict(zip(header, rows[0].split(",")))
    assert float(row["trunk_signed"]) == pytest.approx(-22.0)
    # The magnitude column stays positive - that is the point of having both.
    assert float(row["trunk_deg"]) == pytest.approx(22.0)


def test_row_width_matches_header_width(monitor):
    monitor.session_start = __import__("time").monotonic()
    monitor._log_data(results())
    header, rows = read_rows(monitor.session_log_path)
    assert len(rows[0].split(",")) == len(header)


def test_logging_is_throttled_not_once_per_frame(monitor):
    """Regression: `int(t) != int(t-1)` is true for every t, so the old
    condition logged every frame despite claiming once per second."""
    import time

    # Simulate 30 frames arriving inside a single second.
    monitor.session_start = time.monotonic()
    for _ in range(30):
        monitor._log_data(results())

    _, rows = read_rows(monitor.session_log_path)
    assert len(rows) == 1, f"expected 1 row for a sub-second burst, got {len(rows)}"


def test_throttle_admits_a_row_once_the_interval_elapses(monitor):
    import time

    monitor.session_start = time.monotonic()
    monitor._log_data(results())
    # Pretend the session started further in the past, advancing session_time.
    monitor.session_start -= LOG_INTERVAL_S * 3
    monitor._log_data(results())

    _, rows = read_rows(monitor.session_log_path)
    assert len(rows) == 2
