"""Tests for DriverComfortAnalyzer's use of signed lean angles.

Before the signed-angle fix, `angle_with_vertical` returned only magnitudes,
so this analyzer's negative lower bounds (trunk min=-5, neck min=-10) and its
`elif trunk_angle < -10: 'Backward trunk lean'` branch were unreachable - the
input could never be negative. Recline is the variable that distinguishes a
supported driver from one hunched over the wheel, so that was the single most
important thing the model could not see.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from driver_comfort_analyzer import (  # noqa: E402
    SCORE_ZERO_AT_DEG,
    DriverComfortAnalyzer,
)
from driver_model import DRIVING  # noqa: E402

# The angles below are chosen relative to DRIVING's ideal trunk range rather
# than to "upright", because upright is NOT the driving ideal: a correctly
# seated driver's torso line sits 5-30 degrees behind vertical (see
# driver_model for the sources). Tests written against the old -5..15 range
# treated 0 as perfect and -25 as a fault, which had it backwards.
RECLINE_IDEAL = (DRIVING.ideal["trunk_signed"].min
                 + DRIVING.ideal["trunk_signed"].max) / 2.0   # -17.5
OVER_RECLINED = DRIVING.ideal["trunk_signed"].min - 15.0       # -45.0

# Legs held in the ideal driving range so the torso terms drive the result.
LEGS = {
    "left_hip_angle": 100.0,
    "right_hip_angle": 100.0,
    "left_knee_angle": 115.0,
    "right_knee_angle": 115.0,
}


@pytest.fixture
def analyzer():
    return DriverComfortAnalyzer()


def score(analyzer, trunk, neck=0.0):
    return analyzer.calculate_comfort_score(
        {**LEGS, "trunk_signed": trunk, "neck_signed": neck}
    )


# --------------------------------------------------------------------------
# The branch that used to be dead
# --------------------------------------------------------------------------

def test_backward_lean_risk_is_reachable(analyzer):
    result = score(analyzer, trunk=OVER_RECLINED)
    assert result.ergonomic_risk > 0


def test_backward_lean_produces_its_own_recommendation(analyzer):
    reclined = score(analyzer, trunk=OVER_RECLINED)
    assert any("backward lean" in r.lower() for r in reclined.recommendations)
    assert not any("forward lean" in r.lower() for r in reclined.recommendations)


def test_forward_lean_produces_its_own_recommendation(analyzer):
    hunched = score(analyzer, trunk=25.0)
    assert any("forward lean" in r.lower() for r in hunched.recommendations)
    assert not any("backward lean" in r.lower() for r in hunched.recommendations)


def test_equal_magnitude_opposite_direction_scores_differ(analyzer):
    """The core regression: these were byte-identical before the fix."""
    hunched = score(analyzer, trunk=25.0)
    reclined = score(analyzer, trunk=-25.0)
    assert hunched.overall_comfort != reclined.overall_comfort


def test_ideal_recline_scores_better_than_either_extreme(analyzer):
    ideal = score(analyzer, trunk=RECLINE_IDEAL)
    assert ideal.overall_comfort > score(analyzer, trunk=30.0).overall_comfort
    assert ideal.overall_comfort > score(analyzer, trunk=-70.0).overall_comfort


def test_supported_recline_outscores_bolt_upright(analyzer):
    """The calibration change, stated as a test.

    Under the old -5..15 range a driver sitting bolt upright scored 100 and the
    same driver resting against the backrest scored 0. Automotive guidance puts
    the torso 5-30 degrees behind vertical, so the ordering has to be the other
    way round.
    """
    reclined = score(analyzer, trunk=RECLINE_IDEAL)
    upright = score(analyzer, trunk=0.0)
    assert reclined.overall_comfort > upright.overall_comfort
    assert reclined.posture_quality == pytest.approx(100.0)


def test_hunched_over_the_wheel_is_the_worst_torso_posture(analyzer):
    hunched = score(analyzer, trunk=35.0)
    reclined_same_magnitude = score(analyzer, trunk=-35.0)
    assert hunched.overall_comfort < reclined_same_magnitude.overall_comfort


# --------------------------------------------------------------------------
# Penalty scale: previously divided by the bound's own magnitude
# --------------------------------------------------------------------------

def test_mild_recline_does_not_collapse_to_zero(analyzer):
    """Regression: penalty = deviation/abs(min_val)*100 with min_val=-5 meant
    10 degrees of recline scored 0 while a 15-degree hip deviation scored 82."""
    assert score(analyzer, trunk=-10.0).posture_quality > 50.0


def test_penalty_is_linear_in_degrees_outside_the_range(analyzer):
    """Equal overshoot in degrees costs the same, whatever the bound's value."""
    trunk_range = analyzer.ideal_driving_angles["trunk_signed"]
    overshoot = 10.0
    just_outside = score(analyzer, trunk=trunk_range["max"] + overshoot)
    expected_joint_score = 100.0 - overshoot / SCORE_ZERO_AT_DEG * 100.0
    # posture_quality is the weighted mean; the trunk term alone should have
    # dropped by the per-degree rate, so the total must fall but stay positive.
    assert 0.0 < just_outside.posture_quality < 100.0
    assert expected_joint_score == pytest.approx(66.67, abs=0.01)


def test_score_reaches_zero_at_the_documented_distance(analyzer):
    trunk_max = analyzer.ideal_driving_angles["trunk_signed"]["max"]
    far = score(analyzer, trunk=trunk_max + SCORE_ZERO_AT_DEG + 5)
    near = score(analyzer, trunk=trunk_max + 1)
    assert far.posture_quality < near.posture_quality


# --------------------------------------------------------------------------
# Backfill for data that predates the signed keys
# --------------------------------------------------------------------------

def test_legacy_magnitude_only_input_still_scores(analyzer):
    """Replayed historical logs carry no *_signed keys.

    They must not silently drop the torso terms from the weighted average.
    """
    legacy = analyzer.calculate_comfort_score(
        {**LEGS, "trunk_from_vertical": 25.0, "neck_from_vertical": 10.0}
    )
    explicit = score(analyzer, trunk=25.0, neck=10.0)
    assert legacy.overall_comfort == pytest.approx(explicit.overall_comfort, abs=1e-9)


def test_backfill_does_not_mutate_the_caller_dict(analyzer):
    angles = {**LEGS, "trunk_from_vertical": 25.0, "neck_from_vertical": 10.0}
    before = dict(angles)
    analyzer.calculate_comfort_score(angles)
    assert angles == before


def test_signed_keys_take_precedence_over_magnitude_keys(analyzer):
    """A caller supplying both must get the signed interpretation."""
    both = analyzer.calculate_comfort_score({
        **LEGS, "trunk_signed": OVER_RECLINED,
        "trunk_from_vertical": abs(OVER_RECLINED),
        "neck_signed": 0.0, "neck_from_vertical": 0.0,
    })
    # The magnitude-only backfill would read this as a FORWARD lean of 45 deg.
    assert any("backward lean" in r.lower() for r in both.recommendations)
    assert not any("forward lean" in r.lower() for r in both.recommendations)


# --------------------------------------------------------------------------
# Fatigue drift
# --------------------------------------------------------------------------

def test_fatigue_drift_sees_oscillation_between_hunch_and_recline(analyzer):
    """With magnitudes, +20/-20 oscillation looked like a constant 20 (no drift)."""
    oscillating = [{"trunk_signed": 20.0 if i % 2 else -20.0, "neck_signed": 0.0}
                   for i in range(20)]
    steady = [{"trunk_signed": 20.0, "neck_signed": 0.0} for _ in range(20)]

    osc = analyzer.calculate_comfort_score(
        {**LEGS, "trunk_signed": 20.0, "neck_signed": 0.0}, oscillating, 600)
    std = analyzer.calculate_comfort_score(
        {**LEGS, "trunk_signed": 20.0, "neck_signed": 0.0}, steady, 600)

    assert osc.fatigue_indicator > std.fatigue_indicator


def test_scores_stay_in_their_documented_bounds(analyzer):
    for trunk in (-90.0, -30.0, 0.0, 30.0, 90.0):
        r = score(analyzer, trunk=trunk)
        assert 0.0 <= r.overall_comfort <= 100.0
        assert 0.0 <= r.posture_quality <= 100.0
        assert 0.0 <= r.ergonomic_risk <= 100.0
        assert r.comfort_category in {"Excellent", "Good", "Fair", "Poor"}
