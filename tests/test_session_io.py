"""Tests for session log loading and time-schema normalisation.

Five historical writers produced five different time schemas (see
posture_logs/). PostureAnalytics must normalise all of them to 't_sec'
(seconds from session start) plus 'timestamp' (absolute datetime).

Before the fix, load_all_sessions parsed the filename stamp with
`stem.split('_')[1:3]`, which raised for every real filename and was swallowed
by a per-file `except` - so the method returned an empty DataFrame and
`main.py report` printed "No data found" against 71 log files.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posture_analytics import PostureAnalytics, session_timestamp_from_name  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Filename stamp parsing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("stem, expected", [
    ("angles_20250811_082025", "2025-08-11 08:20:25"),
    ("fullbody_20250813_181749", "2025-08-13 18:17:49"),
    ("quality_20250816_011953", "2025-08-16 01:19:53"),
    ("enhanced_20250818_211435", "2025-08-18 21:14:35"),
    # Two underscores in the prefix: the old positional split broke here.
    ("driver_comfort_20250829_153444", "2025-08-29 15:34:44"),
])
def test_session_timestamp_parses_multiword_prefixes(stem, expected):
    assert session_timestamp_from_name(stem) == pd.Timestamp(expected)


@pytest.mark.parametrize("stem", ["no_stamp_here", "angles", "angles_2025", ""])
def test_session_timestamp_returns_none_without_a_stamp(stem):
    assert session_timestamp_from_name(stem) is None


# --------------------------------------------------------------------------
# Per-schema normalisation
# --------------------------------------------------------------------------

SCHEMAS = {
    # posture_live.py / posture_live_full older writer: t_sec only.
    "angles_20250101_120000.csv":
        "t_sec,label,trunk_from_vertical\n1.0,Neutral,5.0\n2.0,Neutral,6.0\n",
    # posture_live_full.py: numeric seconds stored under the name 'timestamp'.
    "driver_comfort_20250101_120000.csv":
        "timestamp,body_state,trunk_deg\n1.0,Sitting,5.0\n61.0,Sitting,6.0\n",
    # posture_monitor_enhanced.py: ISO timestamp plus t_sec.
    "enhanced_20250101_120000.csv":
        "timestamp,t_sec,body_state\n2025-01-01T12:00:01,1.0,Sitting\n"
        "2025-01-01T12:00:31,31.0,Sitting\n",
}


@pytest.fixture
def analytics(tmp_path):
    for name, body in SCHEMAS.items():
        (tmp_path / name).write_text(body, encoding="utf-8")
    return PostureAnalytics(data_dir=str(tmp_path))


@pytest.mark.parametrize("filename", list(SCHEMAS))
def test_load_session_data_always_yields_both_time_columns(analytics, filename):
    df = analytics.load_session_data(filename)
    assert "t_sec" in df.columns
    assert "timestamp" in df.columns
    assert pd.api.types.is_numeric_dtype(df["t_sec"])
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert df["timestamp"].notna().all()


def test_numeric_timestamp_column_is_recovered_as_elapsed_seconds(analytics):
    """driver_comfort files name their elapsed-seconds column 'timestamp'."""
    df = analytics.load_session_data("driver_comfort_20250101_120000.csv")
    assert list(df["t_sec"]) == [1.0, 61.0]
    assert df["timestamp"].iloc[0] == pd.Timestamp("2025-01-01 12:00:01")
    assert df["timestamp"].iloc[-1] == pd.Timestamp("2025-01-01 12:01:01")


def test_iso_timestamp_is_preferred_over_the_filename_stamp(analytics):
    df = analytics.load_session_data("enhanced_20250101_120000.csv")
    assert df["timestamp"].iloc[0] == pd.Timestamp("2025-01-01 12:00:01")


def test_duration_is_computed_for_every_schema(analytics):
    """Regression: driver_comfort sessions reported 0.0 minutes."""
    for filename in SCHEMAS:
        analysis = analytics.analyze_session(filename)
        assert "error" not in analysis, filename
        assert analysis["duration_minutes"] > 0, filename


def test_load_all_sessions_combines_every_schema(analytics):
    df = analytics.load_all_sessions()
    assert len(df) == 6
    assert df["session_file"].nunique() == 3
    assert df["timestamp"].is_monotonic_increasing


def test_load_all_sessions_reports_unusable_files(analytics, capsys):
    data_dir = Path(analytics.data_dir)
    (data_dir / "no_stamp.csv").write_text("t_sec\n1.0\n", encoding="utf-8")
    (data_dir / "headers_only_20250101_120000.csv").write_text("t_sec\n", encoding="utf-8")

    df = analytics.load_all_sessions()
    out = capsys.readouterr().out

    assert len(df) == 6, "unusable files must not contaminate the frame"
    assert "no_stamp.csv" in out, "a skipped file must be named, not swallowed"
    assert "headers_only_20250101_120000.csv" in out


def test_empty_directory_returns_empty_frame_without_raising(tmp_path):
    assert PostureAnalytics(data_dir=str(tmp_path)).load_all_sessions().empty


# --------------------------------------------------------------------------
# Regression corpus: the real logs in the repo
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    not (REPO_ROOT / "posture_logs").exists(),
    reason="posture_logs/ not present",
)
def test_real_log_corpus_still_loads():
    """Guards against a future schema change orphaning the recorded sessions."""
    df = PostureAnalytics(data_dir=str(REPO_ROOT / "posture_logs")).load_all_sessions()
    assert not df.empty
    assert df["timestamp"].notna().all()
    assert df["timestamp"].is_monotonic_increasing
    # 36 of the 55 CSVs carry rows; the other 19 are header-only aborted runs.
    assert df["session_file"].nunique() >= 36
