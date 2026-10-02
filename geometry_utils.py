"""Geometric primitives for posture angles.

Unmeasurable angles
-------------------
An angle this module cannot compute returns DEGENERATE_ANGLE, which is NaN.
It used to be 0.0, and 0.0 is a MEASUREMENT: fed to angle_with_vertical it
means "perfectly upright" and scores 100, and fed to angle_3pts it means
"fully folded joint" and fires a strain risk. Either way a lost landmark
entered the model as a confident reading.

NaN cannot be mistaken for a measurement, but it does not propagate safely
through comparisons either - `nan <= 25.0` is False, so a naive threshold test
silently reads an unknown angle as "over the limit". Every consumer must gate
on is_measured() first. The ones in this project do:

  driver_model.is_measured / torso_view / weighted_score / RiskRule.fires
  posture_fullbody_rules.detect_body_state
  posture_scoring.PostureQualityScorer.calculate_category_scores
  ergonomics_scores.compute_rula / compute_reba
  smoothing.EMA.update          (holds state, never poisons the filter)
  session_schema.row_from       (writes empty, never the string "nan")
"""
import math

import numpy as np

# Returned when an angle cannot be measured: a segment of (near) zero length,
# which happens when MediaPipe collapses two landmarks onto each other, or a
# landmark the caller marked as not trustworthy (see landmark_confidence).
DEGENERATE_ANGLE = float("nan")


def is_measured(value) -> bool:
    """True when `value` is a real measurement rather than absent or NaN.

    The single gate for "may I compare this number?". Guarding with
    `if value:` would also reject a true 0.0, and `if value is not None:`
    would let NaN through into a comparison that silently evaluates False.
    """
    if value is None:
        return False
    try:
        return not math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def format_angle(value, decimals: int = 1, placeholder: str = "--") -> str:
    """Render an angle for display, or `placeholder` if it was not measured.

    Exists so no UI prints the string "nan" at a driver, and so a frozen or
    absent reading is visibly absent rather than looking like a steady one.
    """
    if not is_measured(value):
        return placeholder
    return f"{float(value):.{decimals}f}"


def angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-6) -> float:
    """
    Angle at point B formed by A-B-C in degrees (0..180).

    Returns DEGENERATE_ANGLE when either segment has no length.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    ba = a - b
    bc = c - b
    na = float(np.linalg.norm(ba))
    nc = float(np.linalg.norm(bc))
    # Guard the division rather than padding the norms: adding eps to the
    # denominator shifts the exact cases (0 and 180 degrees) by ~0.01 deg.
    if na < eps or nc < eps:
        return DEGENERATE_ANGLE
    cosang = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def signed_angle_with_vertical(p1, p2, eps: float = 1e-6) -> float:
    """
    Signed angle in degrees between the segment p1->p2 and straight up.

    Range (-180, 180]. 0 means p2 sits directly above p1 (OpenCV pixel space,
    where y grows downward). Positive means p2 is displaced toward +x (image
    right), negative toward -x (image left).

    This is the primitive: `angle_with_vertical` is its direction-invariant
    magnitude view, so the two can never disagree.

    CAMERA DEPENDENCE - read this before using the sign anatomically. In 2D
    pixel space the sign encodes image-left vs image-right, NOT body-forward
    vs body-backward. The two coincide only when the subject is filmed from
    the side, and which side maps to "forward" depends on where the camera
    sits - declare it via posture_angles.compute_angles(
    forward_is_image_right=...). For a camera mounted in front of or behind
    the driver, leaning forward barely displaces x at all and this sign is
    mostly noise; recovering lean direction in that setup needs MediaPipe
    pose_world_landmarks (true 3D), not this function.

    Returns DEGENERATE_ANGLE when the segment has no length.
    """
    v = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
    if float(np.linalg.norm(v)) < eps:
        return DEGENERATE_ANGLE
    # atan2(dx, -dy): negating dy flips image coordinates so "up" is positive,
    # which puts 0 at straight up and grows positive toward image right.
    return float(np.degrees(np.arctan2(v[0], -v[1])))


def fold_to_vertical_magnitude(signed_deg: float) -> float:
    """
    Collapse a signed angle-from-vertical to its 0..90 magnitude.

    Direction-invariant in both axes: straight up (0) and straight down (180)
    both fold to 0, and +-16.7 both fold to 16.7. This is the measure that
    RULA/REBA and the threshold rules expect.
    """
    a = abs(float(signed_deg))
    return 90.0 - abs(a - 90.0)


def angle_with_vertical(p1, p2, eps: float = 1e-6) -> float:
    """
    Angle (0..90) between segment p1->p2 and the vertical axis.
    Up vs down are treated the same (direction-invariant).

    Derived from `signed_angle_with_vertical` so there is one computation of
    record. Use the signed function wherever lean DIRECTION matters: a
    reclined driver and a driver hunched over the wheel produce the same
    number here, which is why DriverComfortAnalyzer reads the signed values.

    Returns DEGENERATE_ANGLE when the segment has no length.
    """
    return fold_to_vertical_magnitude(signed_angle_with_vertical(p1, p2, eps))
