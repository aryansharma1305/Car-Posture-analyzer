"""Tests for compute_angles, including the signed/unsigned contract.

No camera or MediaPipe needed: compute_angles takes a plain dict of
pixel-space landmarks, so synthetic bodies are enough.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posture_angles import compute_angles  # noqa: E402

TORSO_KEYS = ("trunk", "neck")


def body(shoulder_dx: float = 0.0, head_dx: float = 14.0, elbow_dx: float = 22.0):
    """Side-view-ish synthetic landmarks.

    shoulder_dx shifts the whole torso top: positive puts the shoulders to the
    image RIGHT of the hips, i.e. a forward lean under the default
    forward_is_image_right=True.

    head_dx and elbow_dx add offsets relative to the shoulders so the neck and
    shoulder-elevation angles are non-zero too - otherwise those outputs stay
    at exactly 0 and the assertions below say nothing.
    """
    sx = shoulder_dx
    return {
        "nose":           np.array([100.0 + sx + head_dx, 180.0]),
        "left_ear":       np.array([95.0 + sx + head_dx, 185.0]),
        "right_ear":      np.array([105.0 + sx + head_dx, 185.0]),
        "left_shoulder":  np.array([95.0 + sx, 250.0]),
        "right_shoulder": np.array([105.0 + sx, 250.0]),
        "left_elbow":     np.array([95.0 + sx + elbow_dx, 330.0]),
        "right_elbow":    np.array([105.0 + sx + elbow_dx, 330.0]),
        "left_wrist":     np.array([95.0 + sx + elbow_dx, 400.0]),
        "right_wrist":    np.array([105.0 + sx + elbow_dx, 400.0]),
        "left_hip":       np.array([95.0, 400.0]),
        "right_hip":      np.array([105.0, 400.0]),
        "left_knee":      np.array([150.0, 470.0]),
        "right_knee":     np.array([160.0, 470.0]),
        "left_ankle":     np.array([150.0, 560.0]),
        "right_ankle":    np.array([160.0, 560.0]),
    }


def test_emits_both_signed_and_magnitude_torso_angles():
    angles = compute_angles(body())
    for key in TORSO_KEYS:
        assert f"{key}_signed" in angles
        assert f"{key}_from_vertical" in angles


@pytest.mark.parametrize("key", TORSO_KEYS)
def test_magnitude_equals_abs_signed_in_the_normal_regime(key):
    for dx in (-60, -25, 0, 25, 60):
        angles = compute_angles(body(dx))
        assert angles[f"{key}_from_vertical"] == pytest.approx(
            abs(angles[f"{key}_signed"]), abs=1e-9
        )


def test_forward_lean_and_recline_are_distinguishable():
    """The P0 fix: these two were previously identical in every output."""
    forward = compute_angles(body(40))
    reclined = compute_angles(body(-40))

    assert forward["trunk_signed"] > 0
    assert reclined["trunk_signed"] < 0
    # ... while the magnitude view still cannot tell them apart.
    assert forward["trunk_from_vertical"] == pytest.approx(
        reclined["trunk_from_vertical"], abs=1e-9
    )


def test_upright_torso_is_near_zero():
    angles = compute_angles(body(0))
    assert angles["trunk_signed"] == pytest.approx(0.0, abs=0.01)
    assert angles["trunk_from_vertical"] == pytest.approx(0.0, abs=0.01)


def test_camera_side_flips_only_the_sign():
    """forward_is_image_right=False mirrors the sign, not the magnitude."""
    right = compute_angles(body(40), forward_is_image_right=True)
    left = compute_angles(body(40), forward_is_image_right=False)

    for key in TORSO_KEYS:
        assert right[f"{key}_signed"] == pytest.approx(-left[f"{key}_signed"], abs=1e-9)
        assert right[f"{key}_from_vertical"] == pytest.approx(
            left[f"{key}_from_vertical"], abs=1e-9
        )


def test_unsigned_outputs_are_unchanged_by_the_signed_rewrite():
    """Golden values captured from the pre-signed-angle implementation.

    Verified against a verbatim reimplementation of the old primitives: every
    key agreed to within 2e-6 degrees. Guards every magnitude consumer
    (posture_rules, posture_fullbody_rules, ergonomics_scores,
    posture_scoring) against a silent shift.
    """
    angles = compute_angles(body(30))
    expected = {
        "neck_from_vertical": 11.7174,
        "trunk_from_vertical": 11.3099,
        "left_hip_angle": 130.5328,
        "right_hip_angle": 130.5328,
        "left_knee_angle": 141.8428,
        "right_knee_angle": 141.8428,
        "left_shoulder_elev": 15.3763,
        "right_shoulder_elev": 15.3763,
    }
    for key, value in expected.items():
        assert angles[key] == pytest.approx(value, abs=0.01), key


def test_all_magnitude_angles_stay_in_range():
    for dx in (-200, -40, 0, 40, 200):
        angles = compute_angles(body(dx))
        assert 0.0 <= angles["trunk_from_vertical"] <= 90.0
        assert 0.0 <= angles["neck_from_vertical"] <= 90.0
        for side in ("left", "right"):
            assert 0.0 <= angles[f"{side}_hip_angle"] <= 180.0
            assert 0.0 <= angles[f"{side}_knee_angle"] <= 180.0


# --------------------------------------------------------------------------
# resync_magnitude_views: keeps the two views consistent after smoothing
# --------------------------------------------------------------------------

def test_smoothing_alone_pulls_the_two_views_apart():
    """Documents why resync_magnitude_views exists.

    AngleSmoother EMAs every key in the dict independently, so for a driver
    oscillating either side of vertical the smoothed magnitude settles near
    the amplitude while the smoothed signed value settles near zero.
    """
    from smoothing import AngleSmoother

    sm = AngleSmoother(alpha=0.25)
    smoothed = None
    for i in range(40):
        smoothed = sm(compute_angles(body(16 if i % 2 else -16)))

    assert abs(abs(smoothed["trunk_signed"]) - smoothed["trunk_from_vertical"]) > 1.0


def test_resync_restores_the_invariant_after_smoothing():
    from smoothing import AngleSmoother
    from posture_angles import resync_magnitude_views

    sm = AngleSmoother(alpha=0.25)
    resynced = None
    for i in range(40):
        resynced = resync_magnitude_views(sm(compute_angles(body(16 if i % 2 else -16))))

    assert resynced["trunk_from_vertical"] == pytest.approx(
        abs(resynced["trunk_signed"]), abs=1e-9
    )
    assert resynced["neck_from_vertical"] == pytest.approx(
        abs(resynced["neck_signed"]), abs=1e-9
    )


def test_resync_returns_a_new_dict():
    from posture_angles import resync_magnitude_views

    angles = compute_angles(body(20))
    angles["trunk_signed"] = -45.0
    before = dict(angles)
    resync_magnitude_views(angles)
    assert angles == before


def test_resync_leaves_non_torso_angles_alone():
    from posture_angles import resync_magnitude_views

    angles = compute_angles(body(20))
    resynced = resync_magnitude_views(angles)
    for key in ("left_hip_angle", "right_knee_angle", "left_shoulder_elev"):
        assert resynced[key] == angles[key]


def test_resync_tolerates_missing_signed_keys():
    """Replayed historical data carries magnitudes only."""
    from posture_angles import resync_magnitude_views

    legacy = {"trunk_from_vertical": 12.0, "left_hip_angle": 95.0}
    assert resync_magnitude_views(legacy) == legacy
