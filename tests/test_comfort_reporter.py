"""Tests for DriverComfortReporter's reading of the canonical schema.

The reporter sits between the canonical session CSV and the driver comfort
model, and every bug it has had came from reading a column as something it is
not: 'timestamp' as elapsed seconds, 'trunk_deg' as a signed angle. These tests
pin the column contract from the reading side.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from driver_comfort_reporter import DriverComfortReporter  # noqa: E402
from driver_model import DESK, DRIVING  # noqa: E402
from session_schema import SessionMetadata, SessionWriter  # noqa: E402


def write_session(dir_path, name, rows, *, metadata=None):
    """Write a canonical session CSV plus sidecar, return its filename."""
    path = Path(dir_path) / name
    meta = metadata or SessionMetadata(
        session_id=path.stem, started_at="2026-01-01T00:00:00",
        app="posture_live_full.py", posture_reference=DRIVING.name,
        driver_id="D1", seat_id="SEAT-A",
    )
    writer = SessionWriter(path, metadata=meta)
    for row in rows:
        writer.write(**row)
    return name


def driving_rows(n, trunk_signed):
    """n canonical samples at a fixed torso angle, one second apart."""
    return [
        {
            "t_sec": float(i),
            "angles": {
                "trunk_signed": trunk_signed,
                "trunk_from_vertical": abs(trunk_signed),
                "neck_signed": 0.0,
                "neck_from_vertical": 0.0,
                "left_hip_angle": 105.0, "right_hip_angle": 105.0,
                "left_knee_angle": 115.0, "right_knee_angle": 115.0,
            },
            "body_state": "Sitting",
            "upper_body": "Neutral",
            "comfort": _Comfort(),
        }
        for i in range(n)
    ]


class _Comfort:
    """Minimal stand-in for DriverComfortScore, as SessionWriter consumes it."""
    overall_comfort = 80.0
    comfort_category = "Good"
    posture_quality = 85.0
    ergonomic_risk = 10.0
    fatigue_indicator = 5.0


@pytest.fixture
def reporter(tmp_path):
    return DriverComfortReporter(str(tmp_path))


# ---------------------------------------------------------------------------
# Elapsed seconds vs wall clock
# ---------------------------------------------------------------------------

def test_time_series_bins_on_elapsed_seconds_not_the_wall_clock(reporter, tmp_path):
    """Regression: this binned pd.cut on 'timestamp', which the canonical schema
    defines as an ISO wall-clock STRING, so every canonical and migrated session
    died with numpy.exceptions.DTypePromotionError."""
    name = write_session(tmp_path, "driver_comfort_20260101_000000.csv",
                         driving_rows(40, -20.0))
    report = reporter.generate_comprehensive_report(name)
    bins = report['time_series_analysis']['time_based_comfort']
    assert bins, "no time bins produced"
    assert all(isinstance(k, int) for k in bins)


def test_duration_is_elapsed_seconds(reporter, tmp_path):
    name = write_session(tmp_path, "driver_comfort_20260101_000001.csv",
                         driving_rows(15, -20.0))
    _, session_info = reporter.load_session_data(name)
    assert session_info['duration'] == pytest.approx(14.0)


def test_a_single_sample_does_not_raise_on_binning(reporter, tmp_path):
    name = write_session(tmp_path, "driver_comfort_20260101_000002.csv",
                         driving_rows(1, -20.0))
    report = reporter.generate_comprehensive_report(name)
    assert report['time_series_analysis']['time_based_comfort'] == {}
    assert 'time_based_comfort_note' in report['time_series_analysis']


# ---------------------------------------------------------------------------
# Which model the session is read under
# ---------------------------------------------------------------------------

def test_reference_comes_from_the_sidecar(reporter, tmp_path):
    name = write_session(
        tmp_path, "enhanced_20260101_000003.csv", driving_rows(12, -20.0),
        metadata=SessionMetadata(session_id="s", started_at="t",
                                 app="posture_monitor_enhanced.py",
                                 posture_reference=DESK.name))
    _, session_info = reporter.load_session_data(name)
    assert session_info['posture_reference'] == DESK.name
    assert session_info['posture_reference_source'] == 'sidecar'


def test_a_session_without_a_reference_reports_the_fallback_as_a_fallback(
        reporter, tmp_path):
    name = write_session(
        tmp_path, "driver_comfort_20260101_000004.csv", driving_rows(12, -20.0),
        metadata=SessionMetadata(session_id="s", started_at="t",
                                 app="posture_live_full.py"))
    _, session_info = reporter.load_session_data(name)
    assert session_info['posture_reference'] == DRIVING.name
    assert session_info['posture_reference_source'] == 'default'


# ---------------------------------------------------------------------------
# Signed lean
# ---------------------------------------------------------------------------

def test_signed_columns_are_passed_to_the_model_when_recorded(reporter, tmp_path):
    name = write_session(tmp_path, "driver_comfort_20260101_000005.csv",
                         driving_rows(12, -20.0))
    report = reporter.generate_comprehensive_report(name)
    assert report['session_info']['signed_lean_recorded'] is True
    # -20 is inside DRIVING's ideal trunk range, so the model must be content.
    assert report['summary']['average_comfort'] > 90


def test_a_supported_recline_is_not_reported_as_a_fault(reporter, tmp_path):
    """Reading the magnitude would make this look like a 20-degree forward lean."""
    name = write_session(tmp_path, "driver_comfort_20260101_000006.csv",
                         driving_rows(12, -20.0))
    report = reporter.generate_comprehensive_report(name)
    advice = report['ergonomic_insights']['seat_adjustment_recommendations']
    assert not any('forward lean' in a.lower() for a in advice)


def test_over_reclined_session_is_reported_as_backward_lean(reporter, tmp_path):
    """The branch that was dead: it tested trunk_deg < -5, and trunk_deg is a
    0..90 magnitude."""
    name = write_session(tmp_path, "driver_comfort_20260101_000007.csv",
                         driving_rows(12, -50.0))
    report = reporter.generate_comprehensive_report(name)
    advice = report['ergonomic_insights']['seat_adjustment_recommendations']
    assert any('backward lean' in a.lower() for a in advice)


def test_hunched_session_is_reported_as_forward_lean(reporter, tmp_path):
    name = write_session(tmp_path, "driver_comfort_20260101_000008.csv",
                         driving_rows(12, 25.0))
    report = reporter.generate_comprehensive_report(name)
    advice = report['ergonomic_insights']['seat_adjustment_recommendations']
    assert any('forward lean' in a.lower() for a in advice)


def test_magnitude_only_session_says_the_direction_was_not_recorded(
        reporter, tmp_path):
    """A migrated legacy session: trunk_deg present, trunk_signed empty. The
    reporter must not assert a direction it cannot know."""
    rows = driving_rows(12, -40.0)
    for row in rows:
        row['angles'] = {k: v for k, v in row['angles'].items()
                         if not k.endswith('_signed')}
    name = write_session(tmp_path, "driver_comfort_20260101_000009.csv", rows)
    report = reporter.generate_comprehensive_report(name)
    assert report['session_info']['signed_lean_recorded'] is False
    advice = ' '.join(report['ergonomic_insights']['seat_adjustment_recommendations'])
    assert 'not recorded' in advice.lower()
    assert 'backward lean' not in advice.lower()
    assert 'forward lean' not in advice.lower()


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------

def test_attribution_comes_from_the_sidecar_never_from_placeholders(
        reporter, tmp_path):
    name = write_session(tmp_path, "driver_comfort_20260101_000010.csv",
                         driving_rows(12, -20.0))
    _, session_info = reporter.load_session_data(name)
    assert session_info['driver_id'] == 'D1'
    assert session_info['seat_type'] == 'SEAT-A'
    assert 'test_driver' not in session_info.values()


def test_a_session_with_no_sidecar_reports_unknown(reporter, tmp_path):
    name = write_session(tmp_path, "driver_comfort_20260101_000011.csv",
                         driving_rows(12, -20.0))
    SessionMetadata.sidecar_path(tmp_path / name).unlink()
    _, session_info = reporter.load_session_data(name)
    assert session_info['driver_id'] == 'unknown'
    assert session_info['seat_type'] == 'unknown'
    assert session_info['metadata_source'] == 'none'
