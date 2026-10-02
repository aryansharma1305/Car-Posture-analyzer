"""RULA / REBA categories and short feedback lines.

Two different kinds of number live here, and the difference matters:

  RULA_* / REBA_*    bounds that belong to the RULA and REBA instruments
                     themselves. They are properties of the assessment tool,
                     not of a seating context, so they are NOT part of a
                     PostureReference.
  feedback_lines     plain-language advice, which IS context-dependent and
                     therefore reads its thresholds from a PostureReference.
                     It used to carry its own copies (trunk 18/26, neck 25,
                     shoulder 35, hip 80..110, knee 80) that disagreed with
                     every other module - the label said "ForwardHead" at 20
                     degrees while the advice line stayed silent until 25.

RULA and REBA both score trunk flexion as a direction-invariant MAGNITUDE, so
these functions read the '*_from_vertical' keys. That is faithful to the
instruments but means a driver correctly reclined against the backrest scores
as trunk-flexed. Neither instrument was designed for a seat that supports the
torso; read the driver comfort score, not RULA, for that judgement.
"""
from typing import Dict, Tuple, List

from driver_model import DEFAULT_REFERENCE, MILD_BAND_DEG, PostureReference, torso_view
from geometry_utils import is_measured

# Colors as BGR tuples (for convenience if UI wants them)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)

# RULA: neck and trunk flexion band edges, and upper-arm elevation band edges.
RULA_NECK_BOUNDS = (10.0, 20.0)
RULA_TRUNK_BOUNDS = (10.0, 20.0)
RULA_SHOULDER_BOUNDS = (20.0, 45.0)

# REBA: trunk flexion band edges, and the knee/hip flexion band edges used for
# the leg score (near-straight -> 1, moderately bent -> 2, deeply bent -> 3).
REBA_TRUNK_BOUNDS = (10.0, 20.0)
REBA_LEG_STRAIGHT_DEG = 150.0
REBA_LEG_BENT_DEG = 70.0

# Category returned when a required angle was not measured this frame. Neither
# instrument has a "do not know" score, so one is added here rather than
# defaulting the missing angle to 0 - which is what `torso_view(...) or 0.0`
# did, scoring an unseen neck as perfectly neutral and the frame as Acceptable.
NO_DATA_CATEGORY = 0
NO_DATA_LABEL = "No data"


def _bucket(value: float, bounds: Tuple[float, float]) -> int:
    """Return 1/2/3 for value within simple thresholds (lo, hi)."""
    lo, hi = bounds
    if value <= lo:
        return 1
    if value <= hi:
        return 2
    return 3


def compute_rula(angles: Dict[str, float]) -> Tuple[int, str]:
    """Very simplified RULA category (1..3) and label.
    Uses neck, trunk and shoulder elevation.
    """
    neck_deg = torso_view(angles, "neck_from_vertical")
    trunk_deg = torso_view(angles, "trunk_from_vertical")
    shoulders = [angles[k] for k in ("left_shoulder_elev", "right_shoulder_elev")
                 if is_measured(angles.get(k))]
    if neck_deg is None or trunk_deg is None or not shoulders:
        return NO_DATA_CATEGORY, NO_DATA_LABEL

    neck = _bucket(neck_deg, RULA_NECK_BOUNDS)
    trunk = _bucket(trunk_deg, RULA_TRUNK_BOUNDS)
    sh = max(_bucket(v, RULA_SHOULDER_BOUNDS) for v in shoulders)
    score = neck + trunk + sh
    if score <= 3:
        return 1, "Acceptable"
    if score <= 5:
        return 2, "Investigate"
    return 3, "High Risk"


def compute_reba(angles: Dict[str, float]) -> Tuple[int, str]:
    """Very simplified REBA category (1..3) and label.
    Uses trunk tilt and leg posture (knee/hip flexion).
    """
    trunk_deg = torso_view(angles, "trunk_from_vertical")
    knees = [angles[k] for k in ("left_knee_angle", "right_knee_angle")
             if is_measured(angles.get(k))]
    hips = [angles[k] for k in ("left_hip_angle", "right_hip_angle")
            if is_measured(angles.get(k))]
    if trunk_deg is None or not knees or not hips:
        return NO_DATA_CATEGORY, NO_DATA_LABEL

    trunk_cat = _bucket(trunk_deg, REBA_TRUNK_BOUNDS)
    knee_min = min(knees)
    hip_min = min(hips)

    # Knee/hip flexion buckets: near 180 -> 1, 70..120 -> 2, <70 -> 3
    def flex_cat(a: float) -> int:
        if a >= REBA_LEG_STRAIGHT_DEG:
            return 1
        if a >= REBA_LEG_BENT_DEG:
            return 2
        return 3

    leg_cat = max(flex_cat(knee_min), flex_cat(hip_min))
    score = trunk_cat + leg_cat
    if score <= 2:
        return 1, "Low Risk"
    if score <= 4:
        return 2, "Medium Risk"
    return 3, "High Risk"


def feedback_lines(angles: Dict[str, float],
                   reference: PostureReference = DEFAULT_REFERENCE) -> List[str]:
    """Short, plain-language advice lines for the UI.

    Every threshold is the reference's, so the advice can never disagree with
    the label the same reference produced. A threshold whose angle view is
    missing from `angles` is skipped rather than guessed.
    """
    lines: List[str] = []
    ideal = reference.ideal

    trunk = torso_view(angles, reference.slouch_key)
    if trunk is not None:
        if trunk > reference.slouch_deg + MILD_BAND_DEG:
            lines.append("Trunk: Severe slouching detected.")
        elif trunk > reference.slouch_deg:
            lines.append("Trunk: Mild slouching detected.")

    neck = torso_view(angles, reference.forward_head_key)
    if neck is not None and neck > reference.forward_head_deg:
        lines.append("Neck: Forward head posture.")

    shoulders = [angles[k] for k in ("left_shoulder_elev", "right_shoulder_elev")
                 if is_measured(angles.get(k))]
    if shoulders and max(shoulders) > reference.shoulder_elev_deg:
        lines.append("Shoulders: Elevated. Relax and lower them.")

    hips = [angles[k] for k in ("left_hip_angle", "right_hip_angle")
            if is_measured(angles.get(k))]
    hip_range = ideal.get("left_hip_angle")
    if hips and hip_range is not None:
        hip_avg = sum(hips) / len(hips)
        if not hip_range.contains(hip_avg):
            lines.append("Hips: Poor lower back support (check seat).")

    knees = [angles[k] for k in ("left_knee_angle", "right_knee_angle")
             if is_measured(angles.get(k))]
    knee_range = ideal.get("left_knee_angle")
    if knees and knee_range is not None:
        knee_avg = sum(knees) / len(knees)
        if knee_avg < knee_range.min:
            lines.append("Knees: Too closed; lower chair or move feet forward.")

    if not lines:
        if trunk is None and neck is None and not hips and not knees:
            # Nothing was measurable, which is not the same as nothing wrong.
            lines.append("No reading: landmarks not visible.")
        else:
            lines.append("All Clear: Good posture.")

    return lines


def risk_buckets(rula_cat: int, reba_cat: int) -> str:
    """Return 'good' | 'neutral' | 'high' | 'unknown' from categories.

    'unknown' when either instrument could not be scored. It is reported
    separately rather than folded into 'neutral' so a session's good-time and
    high-risk-time totals only count frames that were actually assessed.
    """
    if NO_DATA_CATEGORY in (rula_cat, reba_cat):
        return "unknown"
    if rula_cat == 1 and reba_cat == 1:
        return "good"
    if rula_cat == 3 or reba_cat == 3:
        return "high"
    return "neutral"
