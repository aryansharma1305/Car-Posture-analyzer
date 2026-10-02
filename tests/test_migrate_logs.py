"""Tests for the legacy log migration.

Covers all five legacy families, the three time conventions, the fold
correction for the pre-fold magnitude convention, and the guarantee that
originals are never modified.
"""
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import migrate_logs  # noqa: E402
from session_schema import COLUMNS, SCHEMA_VERSION, SessionMetadata  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# One file per legacy family, with its real header.
LEGACY = {
    # t_sec only, and the pre-fold magnitude convention (upright ~= 179)
    "angles_20250101_120000.csv":
        "t_sec,label,neck_from_vertical,trunk_from_vertical,left_hip_angle,"
        "right_hip_angle,left_knee_angle,right_knee_angle,left_shoulder_elev,"
        "right_shoulder_elev\n"
        "1.01,NotSeated,179.55,179.20,167.31,177.57,179.53,176.41,18.69,20.38\n",
    # t_sec only, folded magnitudes
    "fullbody_20250101_120000.csv":
        "t_sec,body_state,upper_body,lower_visible,lean_side,trunk_deg,neck_deg,"
        "knee_L,knee_R,hip_L,hip_R\n"
        "1.03,Sitting,Neutral,False,,1.6,9.3,117.9,119.5,102.7,104.8\n",
    # numeric 'timestamp' that is really elapsed seconds
    "driver_comfort_20250101_120000.csv":
        "timestamp,body_state,upper_body,trunk_deg,neck_deg,knee_L,knee_R,hip_L,"
        "hip_R,comfort_score,comfort_category,posture_quality,ergonomic_risk,"
        "fatigue_indicator\n"
        "1.0,Sitting,Neutral,1.7,1.5,119.8,119.5,102.6,107.8,86.3,Excellent,72.7,0,0\n",
    # ISO timestamp plus t_sec
    "enhanced_20250101_120000.csv":
        "timestamp,t_sec,body_state,upper_body,lower_visible,lean_side,trunk_deg,"
        "neck_deg,knee_L,knee_R,hip_L,hip_R,overall_quality_score,risk_level\n"
        "2025-01-01T12:00:01,1.0,Sitting,Neutral,True,,3.1,9.2,118.2,117.0,104.6,"
        "107.3,63.2,Medium\n",
    "quality_20250101_120000.csv":
        "timestamp,t_sec,overall_score,upper_body_score,lower_body_score,"
        "shoulders_score,symmetry_score,risk_level,recommendations_count\n"
        "2025-01-01T12:00:01,1.0,63.2,100.0,0.0,100.0,54.6,Medium,4\n",
    # header-only aborted session
    "driver_comfort_20250101_130000.csv":
        "timestamp,body_state,upper_body,trunk_deg,neck_deg,knee_L,knee_R,hip_L,"
        "hip_R,comfort_score,comfort_category,posture_quality,ergonomic_risk,"
        "fatigue_indicator\n",
    # no filename stamp to anchor against
    "stray.csv": "t_sec,trunk_deg\n1.0,5.0\n",
}


@pytest.fixture
def logs(tmp_path):
    src = tmp_path / "posture_logs"
    src.mkdir()
    for name, body in LEGACY.items():
        (src / name).write_text(body, encoding="utf-8")
    return src


def rows_of(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def migrate(logs, apply=True):
    out = logs / "migrated"
    results = {}
    for src in sorted(logs.glob("*.csv")):
        results[src.name] = migrate_logs.migrate_file(src, out, apply)
    return out, results


# --------------------------------------------------------------------------
# Non-destructive
# --------------------------------------------------------------------------

def test_originals_are_never_modified(logs):
    before = {p.name: p.read_bytes() for p in logs.glob("*.csv")}
    migrate(logs)
    after = {p.name: p.read_bytes() for p in logs.glob("*.csv")}
    assert before == after


def test_dry_run_writes_nothing(logs):
    out, _ = migrate(logs, apply=False)
    assert not out.exists()


# --------------------------------------------------------------------------
# Every family lands on the canonical schema
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "angles_20250101_120000.csv",
    "fullbody_20250101_120000.csv",
    "driver_comfort_20250101_120000.csv",
    "enhanced_20250101_120000.csv",
    "quality_20250101_120000.csv",
])
def test_every_family_gets_the_canonical_header(logs, name):
    out, _ = migrate(logs)
    header, rows = rows_of(out / name)
    assert header == COLUMNS
    assert len(rows) == 1
    assert int(rows[0]["schema_version"]) == SCHEMA_VERSION


def test_all_migrated_files_have_identical_width(logs):
    out, _ = migrate(logs)
    widths = {len(rows_of(p)[0]) for p in out.glob("*.csv")}
    assert widths == {len(COLUMNS)}


# --------------------------------------------------------------------------
# Time conventions
# --------------------------------------------------------------------------

def test_numeric_timestamp_is_read_as_elapsed_seconds(logs):
    out, _ = migrate(logs)
    _, rows = rows_of(out / "driver_comfort_20250101_120000.csv")
    assert float(rows[0]["t_sec"]) == pytest.approx(1.0)
    assert rows[0]["timestamp"] == "2025-01-01T12:00:01"


def test_iso_timestamp_is_preserved(logs):
    out, _ = migrate(logs)
    _, rows = rows_of(out / "enhanced_20250101_120000.csv")
    assert rows[0]["timestamp"] == "2025-01-01T12:00:01"
    assert float(rows[0]["t_sec"]) == pytest.approx(1.0)


def test_t_sec_only_gets_an_absolute_timestamp(logs):
    out, _ = migrate(logs)
    _, rows = rows_of(out / "fullbody_20250101_120000.csv")
    assert rows[0]["timestamp"].startswith("2025-01-01T12:00:01")


# --------------------------------------------------------------------------
# The fold correction
# --------------------------------------------------------------------------

def test_prefold_magnitudes_are_folded_into_range(logs):
    """The earliest writer logged from the downward vertical unfolded, so an
    upright torso recorded ~179 instead of ~1."""
    out, results = migrate(logs)
    _, rows = rows_of(out / "angles_20250101_120000.csv")
    assert float(rows[0]["trunk_deg"]) == pytest.approx(0.8, abs=0.05)
    assert float(rows[0]["neck_deg"]) == pytest.approx(0.45, abs=0.05)
    assert "folded" in results["angles_20250101_120000.csv"][0]


def test_fold_is_recorded_in_the_sidecar(logs):
    """A silent data transformation would be worse than none."""
    out, _ = migrate(logs)
    meta = SessionMetadata.load(out / "angles_20250101_120000.csv")
    assert meta.extra["folded_columns"]["trunk_deg"] == 1
    assert meta.extra["out_of_range_after_migration"] == {}


def test_already_folded_files_are_left_alone(logs):
    out, results = migrate(logs)
    assert "folded" not in results["fullbody_20250101_120000.csv"][0]
    meta = SessionMetadata.load(out / "fullbody_20250101_120000.csv")
    assert meta.extra["folded_columns"] == {}


def test_migrated_values_are_all_in_canonical_range(logs):
    out, _ = migrate(logs)
    for path in out.glob("*.csv"):
        _, rows = rows_of(path)
        for row in rows:
            for column, (lo, hi) in migrate_logs.CANONICAL_RANGES.items():
                if row[column] == "":
                    continue
                assert lo <= float(row[column]) <= hi, f"{path.name}:{column}"


# --------------------------------------------------------------------------
# Skips, metadata honesty
# --------------------------------------------------------------------------

def test_header_only_sessions_are_skipped(logs):
    _, results = migrate(logs)
    assert "skipped" in results["driver_comfort_20250101_130000.csv"][0]


def test_unstamped_filenames_are_skipped(logs):
    _, results = migrate(logs)
    assert "skipped" in results["stray.csv"][0]


def test_sidecar_does_not_invent_driver_or_seat(logs):
    """These were never captured; guessing them would corrupt the study."""
    out, _ = migrate(logs)
    meta = SessionMetadata.load(out / "driver_comfort_20250101_120000.csv")
    assert meta.driver_id is None
    assert meta.seat_id is None
    assert meta.provenance == "migrated"


def test_sidecar_records_the_originating_app(logs):
    out, _ = migrate(logs)
    meta = SessionMetadata.load(out / "enhanced_20250101_120000.csv")
    assert "posture_monitor_enhanced" in meta.app


def test_signed_lean_is_left_empty_when_unrecoverable(logs):
    """Legacy files recorded magnitudes only; assuming a sign would be a lie."""
    out, _ = migrate(logs)
    _, rows = rows_of(out / "driver_comfort_20250101_120000.csv")
    assert rows[0]["trunk_signed"] == ""
    assert rows[0]["neck_signed"] == ""


# --------------------------------------------------------------------------
# The real corpus
# --------------------------------------------------------------------------

@pytest.mark.skipif(not (REPO_ROOT / "posture_logs" / "migrated").exists(),
                    reason="migration has not been run")
def test_real_migrated_corpus_is_uniform_and_in_range():
    import pandas as pd

    out = REPO_ROOT / "posture_logs" / "migrated"
    files = sorted(out.glob("*.csv"))
    assert len(files) >= 36

    for path in files:
        df = pd.read_csv(path)
        assert list(df.columns) == COLUMNS, path.name
        assert set(df["schema_version"].unique()) == {SCHEMA_VERSION}, path.name
        assert SessionMetadata.sidecar_path(path).exists(), path.name
        for column, (lo, hi) in migrate_logs.CANONICAL_RANGES.items():
            series = pd.to_numeric(df[column], errors="coerce").dropna()
            if len(series):
                assert series.between(lo, hi).all(), f"{path.name}:{column}"
