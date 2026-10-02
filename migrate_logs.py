#!/usr/bin/env python3
"""Migrate legacy session logs to the canonical schema (session_schema.py).

Five historical writers produced five per-sample schemas, and 'timestamp' meant
wall-clock time in two of them and seconds-since-start in a third. This script
rewrites them all into the canonical column set, stamps a schema_version, and
emits a metadata sidecar per session.

NON-DESTRUCTIVE: originals are never touched. Migrated copies go to a separate
directory (default posture_logs/migrated/), so the 36 sessions of real recorded
data remain exactly as captured.

Usage:
    python3 migrate_logs.py                      # dry run, prints a plan
    python3 migrate_logs.py --apply              # write migrated/
    python3 migrate_logs.py --apply --out DIR    # custom destination
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from geometry_utils import fold_to_vertical_magnitude
from driver_model import get_reference
from session_schema import (
    ANGLE_COLUMNS,
    COLUMNS,
    SCHEMA_VERSION,
    UNMEASURED_SEPARATOR,
    SessionMetadata,
    SessionWriter,
)

# Canonical range per column, used to validate migrated output. Magnitudes
# from vertical are 0..90 by construction; joint angles are 0..180.
CANONICAL_RANGES = {
    "trunk_deg": (0.0, 90.0),
    "neck_deg": (0.0, 90.0),
    "shoulder_elev_L": (0.0, 90.0),
    "shoulder_elev_R": (0.0, 90.0),
    "hip_L": (0.0, 180.0), "hip_R": (0.0, 180.0),
    "knee_L": (0.0, 180.0), "knee_R": (0.0, 180.0),
}

# Columns whose out-of-range values are recoverable by folding. The earliest
# writer measured these from the DOWNWARD vertical without folding, so an
# upright torso logged ~179 instead of ~1. fold_to_vertical_magnitude is the
# project's own documented transform for exactly this, so applying it is a
# convention correction rather than a guess - but it is always reported and
# recorded in the session sidecar.
FOLDABLE_COLUMNS = ("trunk_deg", "neck_deg", "shoulder_elev_L", "shoulder_elev_R")

STAMP_RE = re.compile(r"_(\d{8})_(\d{6})$")

# Legacy column name -> canonical column name. Columns absent from a given
# family are simply left empty in the output.
COLUMN_ALIASES = {
    "trunk_deg": "trunk_deg",
    "neck_deg": "neck_deg",
    "trunk_from_vertical": "trunk_deg",
    "neck_from_vertical": "neck_deg",
    "left_hip_angle": "hip_L",
    "right_hip_angle": "hip_R",
    "left_knee_angle": "knee_L",
    "right_knee_angle": "knee_R",
    "left_shoulder_elev": "shoulder_elev_L",
    "right_shoulder_elev": "shoulder_elev_R",
    "hip_L": "hip_L", "hip_R": "hip_R",
    "knee_L": "knee_L", "knee_R": "knee_R",
    "body_state": "body_state",
    "upper_body": "upper_body",
    "label": "posture_label",
    "lower_visible": "lower_visible",
    "lean_side": "lean_side",
    "comfort_score": "comfort_score",
    "comfort_category": "comfort_category",
    "posture_quality": "posture_quality",
    "ergonomic_risk": "ergonomic_risk",
    "fatigue_indicator": "fatigue_indicator",
    "overall_quality_score": "quality_score",
    "overall_score": "quality_score",
    "risk_level": "risk_level",
    "trunk_signed": "trunk_signed",
    "neck_signed": "neck_signed",
}

# Which monitor wrote each filename family, recorded in the sidecar so the
# provenance of every migrated session stays traceable.
FAMILY_APPS = {
    "angles": "posture_live.py",
    "fullbody": "posture_live_full.py (pre-comfort)",
    "driver_comfort": "posture_live_full.py",
    "enhanced": "posture_monitor_enhanced.py",
    "quality": "posture_monitor_enhanced.py (quality log)",
}


# The seating context of each family, as a driver_model reference name. This is
# the CONTEXT the session was recorded in, not the model that produced its
# stored label strings - every legacy writer labelled with the desk thresholds
# regardless of where the camera was, which is the defect Phase 4 fixed. The
# labels carried over from those files should be recomputed, not trusted.
#
# 'angles' (posture_live.py) is left None on purpose: that prototype's seated
# detection assumed a 90-degree knee and desk-style thresholds, and nothing in
# the file or the filename says whether it was recorded in a car. Guessing
# would be worse than reporting it unknown.
FAMILY_REFERENCES = {
    "angles": None,
    "fullbody": "driving",
    "driver_comfort": "driving",
    "enhanced": "desk",
    "quality": "desk",
}


def family_of(stem: str) -> str:
    return STAMP_RE.sub("", stem)


# Canonical column -> the angle name a PostureReference knows it by, so coverage
# can be computed from a row that has already been written.
ANGLE_FROM_COLUMN = {column: angle for angle, column in ANGLE_COLUMNS.items()}


def _stamp_coverage(rows: List[Dict[str, object]], reference) -> Dict[str, int]:
    """Fill measured_fraction / unmeasured / observation on migrated rows.

    frame_confidence stays empty: no legacy writer recorded landmark
    visibility, and a confident-looking 0 or 1 would be a fabrication.

    For the in-car families this marks every historical row as observation
    False, which is correct and is the point: DRIVING requires trunk_signed, and
    no legacy writer recorded lean direction. measured_fraction then says how
    much of the model those rows CAN support, which is what makes them usable
    for limb trend work and unusable for seat comparison.
    """
    if reference is None:
        return {}
    counts: Dict[str, int] = {"observations": 0, "excluded": 0}
    for row in rows:
        angles = {}
        for column, angle in ANGLE_FROM_COLUMN.items():
            value = row.get(column, "")
            angles[angle] = float(value) if str(value).strip() else float("nan")
        row["measured_fraction"] = round(reference.measured_fraction(angles), 3)
        row["unmeasured"] = UNMEASURED_SEPARATOR.join(reference.unmeasured(angles))
        observation = reference.is_observation(angles)
        row["observation"] = observation
        counts["observations" if observation else "excluded"] += 1
    return counts


def session_start(stem: str) -> Optional[str]:
    """ISO-8601 session start from the trailing _YYYYMMDD_HHMMSS stamp."""
    m = STAMP_RE.search(stem)
    if not m:
        return None
    d, t = m.group(1), m.group(2)
    return f"{d[:4]}-{d[4:6]}-{d[6:]}T{t[:2]}:{t[2:4]}:{t[4:]}"


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def resolve_time(row: Dict[str, str], start_iso: str) -> Tuple[str, float]:
    """Return (absolute ISO timestamp, t_sec) for a legacy row.

    Handles the three legacy conventions:
      * 't_sec' only                      -> timestamp derived from the stamp
      * numeric 'timestamp'               -> it is really elapsed seconds
      * ISO 'timestamp' (with/without t_sec)
    """
    from datetime import datetime, timedelta

    start = datetime.fromisoformat(start_iso)
    raw_ts = (row.get("timestamp") or "").strip()
    raw_tsec = (row.get("t_sec") or "").strip()

    if raw_ts and not _is_number(raw_ts):
        stamp = datetime.fromisoformat(raw_ts)
        t_sec = float(raw_tsec) if _is_number(raw_tsec) else (stamp - start).total_seconds()
        return stamp.isoformat(), round(t_sec, 2)

    if _is_number(raw_tsec):
        t_sec = float(raw_tsec)
    elif _is_number(raw_ts):
        # The 'timestamp' column held elapsed seconds, not a clock.
        t_sec = float(raw_ts)
    else:
        raise ValueError("row has neither t_sec nor a usable timestamp")

    return (start + timedelta(seconds=t_sec)).isoformat(), round(t_sec, 2)


def migrate_file(src: Path, out_dir: Path, apply: bool) -> Tuple[str, int]:
    """Migrate one CSV. Returns (status, rows_written)."""
    stem = src.stem
    start_iso = session_start(stem)
    if start_iso is None:
        return "skipped: no _YYYYMMDD_HHMMSS stamp", 0

    with open(src, newline="", encoding="utf-8") as f:
        legacy_rows = list(csv.DictReader(f))

    if not legacy_rows:
        return "skipped: header only (aborted session)", 0

    canonical: List[Dict[str, object]] = []
    for legacy in legacy_rows:
        try:
            timestamp, t_sec = resolve_time(legacy, start_iso)
        except ValueError as e:
            return f"skipped: {e}", 0

        row = {column: "" for column in COLUMNS}
        row["schema_version"] = SCHEMA_VERSION
        row["timestamp"] = timestamp
        row["t_sec"] = t_sec
        for legacy_key, value in legacy.items():
            if legacy_key in ("timestamp", "t_sec") or legacy_key is None:
                continue
            column = COLUMN_ALIASES.get(legacy_key)
            if column:
                row[column] = (value or "").strip()
        canonical.append(row)

    folded = _fold_out_of_range(canonical)
    violations = _validate(canonical)

    if not apply:
        note = f"would migrate {len(canonical)} rows"
        if folded:
            note += " (fold: " + ", ".join(f"{c} x{n}" for c, n in folded.items()) + ")"
        if violations:
            note += " WARNING out of range: " + ", ".join(
                f"{c} x{n}" for c, n in violations.items())
        return note, len(canonical)

    family = family_of(stem)
    reference = get_reference(FAMILY_REFERENCES.get(family)) \
        if FAMILY_REFERENCES.get(family) else None
    coverage = _stamp_coverage(canonical, reference)
    meta = SessionMetadata(
        session_id=stem,
        started_at=start_iso,
        app=FAMILY_APPS.get(family, f"unknown ({family})"),
        posture_reference=FAMILY_REFERENCES.get(family),
        provenance="migrated",
        extra={
            "folded_columns": folded,
            "out_of_range_after_migration": violations,
            "coverage": coverage,
        },
        notes=(
            "Migrated from the legacy schema. driver_id/seat_id/vehicle/"
            "camera_position were never captured by the original writer and "
            "are genuinely unknown - do not guess them. Lean DIRECTION is "
            "also unrecoverable: the source recorded magnitudes only, so "
            "trunk_signed/neck_signed are empty rather than assumed positive. "
            "See extra.folded_columns for any magnitude columns corrected from "
            "the pre-fold convention, and extra.out_of_range_after_migration "
            "for values the migration could not reconcile. posture_reference "
            "is the seating CONTEXT inferred from the writer family; the "
            "body_state/upper_body/posture_label strings in these rows were "
            "produced by the old desk thresholds whatever that context was, "
            "so recompute them rather than trusting them."
        ),
    )
    writer = SessionWriter(out_dir / src.name, metadata=meta)
    for row in canonical:
        writer.write_row(row)

    status = f"migrated {len(canonical)} rows"
    if folded:
        status += " (folded: " + ", ".join(f"{c} x{n}" for c, n in folded.items()) + ")"
    if violations:
        status += " WARNING out of range: " + ", ".join(
            f"{c} x{n}" for c, n in violations.items())
    return status, len(canonical)


def _fold_out_of_range(rows: List[Dict[str, object]]) -> Dict[str, int]:
    """Fold recoverable magnitude columns in place; return per-column counts."""
    counts: Dict[str, int] = {}
    for row in rows:
        for column in FOLDABLE_COLUMNS:
            value = row.get(column, "")
            if value == "" or not _is_number(str(value)):
                continue
            number = float(value)
            lo, hi = CANONICAL_RANGES[column]
            if lo <= number <= hi:
                continue
            row[column] = round(fold_to_vertical_magnitude(number), 1)
            counts[column] = counts.get(column, 0) + 1
    return counts


def _validate(rows: List[Dict[str, object]]) -> Dict[str, int]:
    """Count values still outside their canonical range after migration."""
    counts: Dict[str, int] = {}
    for row in rows:
        for column, (lo, hi) in CANONICAL_RANGES.items():
            value = row.get(column, "")
            if value == "" or not _is_number(str(value)):
                continue
            if not (lo <= float(value) <= hi):
                counts[column] = counts.get(column, 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", default="posture_logs", help="source directory")
    parser.add_argument("--out", default=None,
                        help="destination (default: <logs>/migrated)")
    parser.add_argument("--apply", action="store_true",
                        help="actually write files (default is a dry run)")
    args = parser.parse_args()

    logs = Path(args.logs)
    if not logs.is_dir():
        print(f"No such directory: {logs}")
        return 1
    out_dir = Path(args.out) if args.out else logs / "migrated"

    sources = sorted(p for p in logs.glob("*.csv") if p.is_file())
    if not sources:
        print(f"No CSV files in {logs}")
        return 1

    print(f"{'DRY RUN - ' if not args.apply else ''}{len(sources)} file(s) in {logs}")
    print(f"destination: {out_dir}\n")

    migrated = skipped = total_rows = 0
    for src in sources:
        status, rows = migrate_file(src, out_dir, args.apply)
        if status.startswith("skipped"):
            skipped += 1
        else:
            migrated += 1
            total_rows += rows
        print(f"  {src.name:<44} {status}")

    print(f"\n{migrated} session(s), {total_rows} rows; {skipped} skipped")
    if not args.apply:
        print("\nRe-run with --apply to write. Originals are never modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
