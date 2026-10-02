"""The one session-log schema.

Before this module there were five incompatible per-sample schemas, written by
three monitors, and the column named 'timestamp' meant wall-clock time in two
of them and seconds-since-start in a third. Anything reading more than one
family had to guess.

This module is the single definition of record:

  * COLUMNS           canonical column order, written by every writer
  * SCHEMA_VERSION    stamped into every row so old files stay readable
  * SessionWriter     the only thing that should open a session CSV for writing
  * SessionMetadata   the per-session sidecar (who, which seat, which camera)

Column contract
---------------
timestamp   ISO-8601 wall clock. Always absolute, never elapsed.
t_sec       float seconds since session start. Always relative, never a clock.
*_signed    signed degrees from vertical; NEGATIVE MEANS RECLINED. The sign is
            only anatomically meaningful for a side-mounted camera - see
            SessionMetadata.forward_is_image_right and
            geometry_utils.signed_angle_with_vertical.
*_deg       direction-invariant magnitudes, 0..90, kept because the threshold
            rules and RULA/REBA consume them.

Columns a given monitor does not compute are written empty rather than omitted,
so every file has the same width and pandas never produces a ragged frame.

Confidence (schema 3)
---------------------
MediaPipe reports a landmark for every joint whether or not it saw one, so an
angle can be a measurement or an extrapolation and the number looks the same.
Four columns make the difference visible, and `observation` is the one to filter
on:

frame_confidence    0..1, the LOWEST landmark visibility among the landmarks any
                    angle was built from. Empty when the writer had no
                    visibility information.
measured_fraction   0..1, the share of the posture model's weight that was
                    actually measured this row.
unmeasured          '|'-joined names of the angles that were not measured.
observation         True when the row can be scored against the model at all -
                    enough weight measured AND every required angle present.
                    Rows with False must be excluded from session statistics.

An angle that was not measured is written as an EMPTY cell, never as the string
"nan": "nan" round-trips through pandas as a float NaN but through csv and
spreadsheet readers as text, and the schema already uses empty for "this writer
did not produce a value".
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from geometry_utils import is_measured
from landmark_confidence import frame_confidence as _frame_confidence

SCHEMA_VERSION = 3

# Canonical column order. Append-only: adding a column at the end keeps older
# readers working, reordering or renaming does not.
COLUMNS = [
    "schema_version",
    "timestamp",
    "t_sec",
    # Classification
    "body_state",       # Standing|Sitting|Lying|Squatting|Unknown
    "upper_body",       # Neutral|Slouch|ForwardHead|RaisedShoulders
    "posture_label",    # posture_rules.classify_posture primary label
    "lower_visible",
    "lean_side",        # LeanLeft|LeanRight|''
    # Torso, signed (canonical) and magnitude (derived)
    "trunk_signed",
    "neck_signed",
    "trunk_deg",
    "neck_deg",
    # Limbs
    "hip_L", "hip_R",
    "knee_L", "knee_R",
    "shoulder_elev_L", "shoulder_elev_R",
    # Driver comfort model
    "comfort_score", "comfort_category",
    "posture_quality", "ergonomic_risk", "fatigue_indicator",
    # Desk/office quality model
    "quality_score", "risk_level",
    # Measurement confidence (schema 3). Appended, so a schema-2 reader that
    # addresses columns by name keeps working.
    "frame_confidence", "measured_fraction", "unmeasured", "observation",
]

# Maps an angles dict (as produced by posture_angles.compute_angles) onto the
# canonical column names.
ANGLE_COLUMNS = {
    "trunk_signed": "trunk_signed",
    "neck_signed": "neck_signed",
    "trunk_from_vertical": "trunk_deg",
    "neck_from_vertical": "neck_deg",
    "left_hip_angle": "hip_L",
    "right_hip_angle": "hip_R",
    "left_knee_angle": "knee_L",
    "right_knee_angle": "knee_R",
    "left_shoulder_elev": "shoulder_elev_L",
    "right_shoulder_elev": "shoulder_elev_R",
}

# Degrees are logged to this many decimals; more is false precision given
# MediaPipe's landmark jitter.
ANGLE_DECIMALS = 1


@dataclass
class SessionMetadata:
    """Per-session sidecar, written next to the CSV as <stem>.meta.json.

    The seat/driver/vehicle fields are what make
    DriverComfortAnalyzer.analyze_seat_comparison usable: without them a log is
    an anonymous angle series and cannot be attributed to a seat under test.
    They are Optional so a monitor can still run without them, but a session
    recorded for seat comparison should always set them.
    """
    session_id: str
    started_at: str
    app: str                                  # which monitor wrote this
    schema_version: int = SCHEMA_VERSION
    driver_id: Optional[str] = None
    seat_id: Optional[str] = None
    vehicle: Optional[str] = None
    camera_position: Optional[str] = None     # e.g. "left-side", "front"
    forward_is_image_right: Optional[bool] = None
    # The seating context this session belongs to, as a
    # driver_model.PostureReference name ("driving" or "desk"). It is what a
    # reader needs in order to interpret posture_label / upper_body / comfort_*
    # or to re-score the angles, because the same angles mean opposite things
    # under the two models: 25 degrees behind vertical is the designed-for
    # driving posture and a slouch at a desk. None means the context was never
    # recorded; do not assume one.
    posture_reference: Optional[str] = None
    provenance: str = "recorded"              # or "migrated"
    notes: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def sidecar_path(cls, csv_path) -> Path:
        csv_path = Path(csv_path)
        return csv_path.with_suffix(".meta.json")

    def save(self, csv_path) -> Path:
        path = self.sidecar_path(csv_path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, csv_path) -> Optional["SessionMetadata"]:
        """Return the sidecar for a CSV, or None when it has none."""
        path = cls.sidecar_path(csv_path)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        extra = {k: v for k, v in data.items() if k not in known and k != "extra"}
        kwargs = {k: v for k, v in data.items() if k in known}
        kwargs.setdefault("session_id", Path(csv_path).stem)
        kwargs.setdefault("started_at", "")
        kwargs.setdefault("app", "unknown")
        meta = cls(**kwargs)
        meta.extra.update(data.get("extra", {}))
        meta.extra.update(extra)
        return meta


# Separator for the 'unmeasured' column. Not a comma: these files are CSV.
UNMEASURED_SEPARATOR = "|"


def row_from(
    t_sec: float,
    angles: Optional[Dict[str, float]] = None,
    *,
    timestamp: Optional[str] = None,
    body_state: str = "",
    upper_body: str = "",
    posture_label: str = "",
    extras: Optional[Dict[str, Any]] = None,
    comfort=None,
    quality=None,
    visibility: Optional[Dict[str, float]] = None,
    reference=None,
) -> Dict[str, Any]:
    """Build one canonical row. Unknown fields stay empty, never omitted.

    visibility: MediaPipe's per-landmark confidence for this frame, from
        PoseExtractor.visibility(results). Fills frame_confidence.
    reference: the driver_model.PostureReference the row was scored against.
        Used to fill measured_fraction / unmeasured / observation when no
        comfort or quality score was supplied; when one was, its own coverage
        fields are used, since they describe the score actually recorded here.
    """
    extras = extras or {}
    row: Dict[str, Any] = {column: "" for column in COLUMNS}
    row["schema_version"] = SCHEMA_VERSION
    row["timestamp"] = timestamp or datetime.now().isoformat()
    row["t_sec"] = round(float(t_sec), 2)
    row["body_state"] = body_state
    row["upper_body"] = upper_body
    row["posture_label"] = posture_label
    row["lower_visible"] = extras.get("lower_visible", "")
    row["lean_side"] = extras.get("lean_side") or ""

    for angle_key, column in ANGLE_COLUMNS.items():
        # An unmeasured angle stays an empty cell. Writing round(nan, 1) here
        # put the literal text "nan" in the CSV, which a reader cannot tell
        # apart from a column this writer simply does not produce.
        if angles and is_measured(angles.get(angle_key)):
            row[column] = round(float(angles[angle_key]), ANGLE_DECIMALS)

    if comfort is not None:
        row["comfort_score"] = _round_or_blank(comfort.overall_comfort, 1)
        row["comfort_category"] = comfort.comfort_category
        row["posture_quality"] = _round_or_blank(comfort.posture_quality, 1)
        row["ergonomic_risk"] = _round_or_blank(comfort.ergonomic_risk, 1)
        row["fatigue_indicator"] = _round_or_blank(comfort.fatigue_indicator, 1)

    if quality is not None:
        row["quality_score"] = _round_or_blank(quality.overall_score, 1)
        row["risk_level"] = quality.risk_level

    confidence = _frame_confidence(visibility)
    if confidence is not None:
        row["frame_confidence"] = round(confidence, 3)

    coverage = _coverage(angles, comfort, quality, reference)
    if coverage is not None:
        fraction, unmeasured, observation = coverage
        row["measured_fraction"] = round(fraction, 3)
        row["unmeasured"] = UNMEASURED_SEPARATOR.join(unmeasured)
        row["observation"] = observation

    return row


def _round_or_blank(value, decimals):
    """Round a score, or return '' when it was never computed."""
    return round(float(value), decimals) if is_measured(value) else ""


def _coverage(angles, comfort, quality, reference):
    """(measured_fraction, unmeasured angle names, observation) or None.

    Preference order matters: a score object's own coverage describes the number
    written into this row, so it wins over recomputing from the reference.
    """
    if comfort is not None and getattr(comfort, "measured_fraction", None) is not None:
        return (comfort.measured_fraction,
                list(comfort.unmeasured_angles or []),
                bool(comfort.is_observation))
    if quality is not None and getattr(quality, "measured_fraction", None) is not None:
        return (quality.measured_fraction,
                list(quality.unmeasured_categories or []),
                quality.risk_level != "Unknown")
    if reference is not None and angles is not None:
        return (reference.measured_fraction(angles),
                reference.unmeasured(angles),
                reference.is_observation(angles))
    return None


class SessionWriter:
    """Append-only writer for one canonical session CSV.

    Opens the file, writes the header, and writes the metadata sidecar on
    construction so a session is self-describing even if it crashes mid-run.
    """

    def __init__(self, path, metadata: Optional[SessionMetadata] = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(COLUMNS)
        if metadata is not None:
            metadata.save(self.path)

    def write(self, **kwargs) -> None:
        """Append one row; accepts the same keywords as row_from."""
        row = row_from(**kwargs)
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row[column] for column in COLUMNS])

    def write_row(self, row: Dict[str, Any]) -> None:
        """Append an already-built canonical row."""
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row.get(column, "") for column in COLUMNS])
