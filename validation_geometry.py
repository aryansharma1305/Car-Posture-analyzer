"""Synthetic skeletons with an exactly known torso angle.

The validation harness has to separate two error sources that look identical in
the output:

  1. arithmetic error in this project's own pixel -> angle path
  2. estimation error in MediaPipe's landmark positions

This module removes the first from the question. It builds landmark
coordinates for a torso at an angle you choose, so compute_angles can be fed a
pose whose true angle is known to the last decimal. Whatever error remains in a
REAL trial is then attributable to the camera and the pose model, not to the
code reading them.

Coordinate convention is OpenCV pixel space: x grows rightward, y grows
DOWNWARD, so a smaller y is physically higher in the frame.

Sign convention matches posture_angles.compute_angles with
forward_is_image_right=True (camera on the driver's left):

    trunk_signed = degrees(atan2(dx, -dy))   for hip_centre -> shoulder_centre

so a recline of R degrees puts the shoulders BEHIND the hips, which is toward
image-left, and yields trunk_signed = -R.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

# Segment lengths in pixels, roughly proportional to a seated adult at ~1.5 m
# from a 720p camera. Only the ratios matter to the angles.
TORSO_PX = 120.0
NECK_PX = 45.0
THIGH_PX = 110.0
SHANK_PX = 105.0
UPPER_ARM_PX = 70.0
FOREARM_PX = 65.0


def _from_vertical(origin: np.ndarray, length: float, signed_deg: float) -> np.ndarray:
    """Point `length` away from `origin` at `signed_deg` from straight up.

    Positive `signed_deg` leans toward image-right, matching
    geometry_utils.signed_angle_with_vertical.
    """
    rad = math.radians(signed_deg)
    return origin + np.array([length * math.sin(rad), -length * math.cos(rad)])


def seated_skeleton(
    trunk_signed_deg: float,
    neck_signed_deg: float = 0.0,
    hip_angle_deg: float = 105.0,
    knee_angle_deg: float = 115.0,
    shoulder_elev_deg: float = 10.0,
    hip_centre=(360.0, 400.0),
) -> Dict[str, np.ndarray]:
    """Landmarks for a seated subject whose torso sits at `trunk_signed_deg`.

    Returns the same keys pose_core.PoseExtractor produces, so the result can go
    straight into posture_angles.compute_angles.

    hip_angle_deg and knee_angle_deg are the TARGET joint angles; the thigh and
    shank are placed to realise them, so a round-trip through compute_angles
    recovers what was asked for. That makes this usable as a check on the limb
    angles too, not only the torso.
    """
    hip_c = np.array(hip_centre, dtype=float)
    shoulder_c = _from_vertical(hip_c, TORSO_PX, trunk_signed_deg)
    head_ref = _from_vertical(shoulder_c, NECK_PX, neck_signed_deg)

    # Hip angle is shoulder-hip-knee. The torso direction from the hip is known,
    # so rotate it by the desired interior angle to place the knee. Rotating
    # toward image-right (increasing angle-from-vertical) puts the knee in front
    # of the subject, which is where a seated driver's knees are.
    thigh_dir_deg = trunk_signed_deg + hip_angle_deg
    knee = _from_vertical(hip_c, THIGH_PX, thigh_dir_deg)

    # Knee angle is hip-knee-ankle: rotate the thigh direction the other way so
    # the shank drops toward the floor rather than folding back up the thigh.
    shank_dir_deg = thigh_dir_deg - (180.0 - knee_angle_deg)
    ankle = _from_vertical(knee, SHANK_PX, shank_dir_deg)

    # Shoulder elevation is measured from vertical for shoulder -> elbow, where
    # the arm hangs DOWNWARD, so 180 - elev gives a downward direction.
    elbow = _from_vertical(shoulder_c, UPPER_ARM_PX, 180.0 - shoulder_elev_deg)
    wrist = _from_vertical(elbow, FOREARM_PX, 180.0 - shoulder_elev_deg)

    # Ears and nose must AVERAGE to head_ref, because compute_angles uses
    # mid(nose, mid(ears)) as its head reference.
    ear = head_ref + np.array([-6.0, -4.0])
    nose = 2.0 * head_ref - ear

    # A pure side view, so the left and right landmarks of each pair coincide.
    # That is the geometry the signed angles are only meaningful in anyway, and
    # it keeps the left/right angles identical so a discrepancy in a real trial
    # is unambiguously the pose model's.
    return {
        "nose": nose,
        "left_ear": ear, "right_ear": ear,
        "left_shoulder": shoulder_c, "right_shoulder": shoulder_c,
        "left_elbow": elbow, "right_elbow": elbow,
        "left_wrist": wrist, "right_wrist": wrist,
        "left_hip": hip_c, "right_hip": hip_c,
        "left_knee": knee, "right_knee": knee,
        "left_ankle": ankle, "right_ankle": ankle,
    }
