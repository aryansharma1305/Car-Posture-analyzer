import numpy as np
from geometry_utils import angle_3pts, angle_with_vertical

def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0

def compute_angles(pts: dict) -> dict:
    """
    Compute posture-relevant angles from pixel-space landmarks.
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

    neck_from_vertical = angle_with_vertical(shoulder_c, head_ref)
    trunk_from_vertical = angle_with_vertical(hip_c, shoulder_c)

    left_hip_angle  = angle_3pts(ls, lh, lk)  # torso-hip-thigh
    right_hip_angle = angle_3pts(rs, rh, rk)

    left_knee_angle  = angle_3pts(lh, lk, la) # thigh-knee-shank
    right_knee_angle = angle_3pts(rh, rk, ra)

    left_shoulder_elev  = angle_with_vertical(ls, lelb)
    right_shoulder_elev = angle_with_vertical(rs, relb)

    return {
        "neck_from_vertical": neck_from_vertical,
        "trunk_from_vertical": trunk_from_vertical,
        "left_hip_angle": left_hip_angle,
        "right_hip_angle": right_hip_angle,
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "left_shoulder_elev": left_shoulder_elev,
        "right_shoulder_elev": right_shoulder_elev,
    }
