import numpy as np

from geometry_utils import (
    DEGENERATE_ANGLE,
    angle_3pts,
    angle_with_vertical,
    fold_to_vertical_magnitude,
    signed_angle_with_vertical,
)
from landmark_confidence import VISIBILITY_THRESHOLD, unmeasurable_angles

def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0

# Magnitude view -> the signed series it is derived from.
MAGNITUDE_FROM_SIGNED = {
    "trunk_from_vertical": "trunk_signed",
    "neck_from_vertical": "neck_signed",
}


def resync_magnitude_views(angles: dict) -> dict:
    """Recompute the *_from_vertical magnitudes from their signed sources.

    compute_angles emits the two views in agreement, but any per-key transform
    that treats them as independent series will pull them apart. AngleSmoother
    does exactly that: it EMAs every key in the dict, and for a driver
    oscillating either side of vertical the smoothed magnitude settles near the
    oscillation amplitude while the smoothed signed value settles near zero -
    a 5+ degree disagreement in testing.

    Smoothing the signed series and deriving the magnitude afterwards is also
    the better estimator: an EMA over folded magnitudes is biased away from
    zero, because the fold discards the cancelling sign.

    Call this immediately after smoothing. Returns a new dict.
    """
    out = dict(angles)
    for magnitude_key, signed_key in MAGNITUDE_FROM_SIGNED.items():
        if signed_key in out:
            out[magnitude_key] = fold_to_vertical_magnitude(out[signed_key])
    return out


def compute_angles(pts: dict, forward_is_image_right: bool = True,
                   visibility: dict = None,
                   visibility_threshold: float = VISIBILITY_THRESHOLD) -> dict:
    """
    Compute posture-relevant angles from pixel-space landmarks.

    Emits each torso angle twice, from one computation:
      *_from_vertical  0..90 magnitude, direction-invariant. What the
                       threshold rules and RULA/REBA consume.
      *_signed         -180..180, positive = forward lean, negative = recline.
                       What DriverComfortAnalyzer consumes, since recline is
                       the variable that distinguishes a supported driver from
                       a hunched one.

    forward_is_image_right: whether the driver's anatomical FORWARD direction
        points toward +x in the frame. True when the camera sits on the
        driver's left, False when it sits on their right. Only the sign of the
        *_signed values depends on it.

        This parameter is meaningless for a camera mounted in front of or
        behind the driver: leaning forward barely moves x in that view, so the
        sign carries little signal. See signed_angle_with_vertical.

    visibility: MediaPipe's per-landmark confidence, from
        PoseExtractor.visibility(results). Any angle whose least-visible
        landmark is at or below `visibility_threshold` is returned as NaN
        instead of a number.

        This matters because MediaPipe emits coordinates for occluded joints
        too - it extrapolates them - so without this the knee and hip angles of
        a driver whose legs are under the dash were scored as measurements.
        Passing None disables masking and keeps every computed value, which is
        what a caller replaying landmark-only data has to do.
    """
    ls, rs = pts["left_shoulder"], pts["right_shoulder"]
    lh, rh = pts["left_hip"], pts["right_hip"]
    le, re = pts["left_ear"], pts["right_ear"]
    nose = pts["nose"]
    lelb, relb = pts["left_elbow"], pts["right_elbow"]
    lk, rk = pts["left_knee"], pts["right_knee"]
    la, ra = pts["left_ankle"], pts["right_ankle"]

    shoulder_c = _mid(ls, rs)
    hip_c = _mid(lh, rh)
    head_ref = _mid(nose, _mid(le, re))  # average nose & ears

    # Sign first, magnitude derived - so the two views cannot drift apart.
    lean_sign = 1.0 if forward_is_image_right else -1.0
    neck_signed = lean_sign * signed_angle_with_vertical(shoulder_c, head_ref)
    trunk_signed = lean_sign * signed_angle_with_vertical(hip_c, shoulder_c)

    neck_from_vertical = angle_with_vertical(shoulder_c, head_ref)
    trunk_from_vertical = angle_with_vertical(hip_c, shoulder_c)

    left_hip_angle  = angle_3pts(ls, lh, lk)  # torso-hip-thigh
    right_hip_angle = angle_3pts(rs, rh, rk)

    left_knee_angle  = angle_3pts(lh, lk, la) # thigh-knee-shank
    right_knee_angle = angle_3pts(rh, rk, ra)

    left_shoulder_elev  = angle_with_vertical(ls, lelb)
    right_shoulder_elev = angle_with_vertical(rs, relb)

    angles = {
        "neck_from_vertical": neck_from_vertical,
        "trunk_from_vertical": trunk_from_vertical,
        "neck_signed": neck_signed,
        "trunk_signed": trunk_signed,
        "left_hip_angle": left_hip_angle,
        "right_hip_angle": right_hip_angle,
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "left_shoulder_elev": left_shoulder_elev,
        "right_shoulder_elev": right_shoulder_elev,
    }

    # Blank out angles built on landmarks MediaPipe did not actually see. Done
    # after the arithmetic rather than instead of it: the geometry is cheap, and
    # computing then masking keeps one code path for both cases.
    for angle in unmeasurable_angles(visibility, visibility_threshold):
        if angle in angles:
            angles[angle] = DEGENERATE_ANGLE

    return angles
