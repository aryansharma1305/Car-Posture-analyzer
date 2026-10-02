"""Tests for the single posture reference model.

These pin the two things Phase 4 was for:

  1. There is ONE definition of each ergonomic number, and every module that
     used to carry its own copy now reads it. The tests below compare modules
     against driver_model rather than against literals, so a future edit to a
     threshold cannot leave one consumer behind.
  2. DRIVING and DESK describe DIFFERENT postures, and the difference is the
     sign of the torso angle. A driver reclined against the backrest is Neutral
     under DRIVING and Slouch under DESK, and that is correct in both cases.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config_manager  # noqa: E402
import driver_model  # noqa: E402
from driver_model import (  # noqa: E402
    DESK,
    DRIVING,
    MILD_BAND_DEG,
    Range,
    RiskRule,
    get_reference,
    torso_view,
)
from ergonomics_scores import feedback_lines  # noqa: E402
from posture_fullbody_rules import classify_upper_body  # noqa: E402
from posture_rules import PostureConfig, classify_posture  # noqa: E402
from posture_scoring import PostureQualityScorer  # noqa: E402
from session_schema import SessionMetadata  # noqa: E402


# Legs inside both references' ideal ranges, so the torso drives every result.
LEGS = {
    "left_hip_angle": 100.0, "right_hip_angle": 100.0,
    "left_knee_angle": 100.0, "right_knee_angle": 100.0,
}
RELAXED_SHOULDERS = {"left_shoulder_elev": 10.0, "right_shoulder_elev": 10.0}


def torso(signed: float) -> dict:
    """Both views of one torso angle, consistent with each other."""
    from geometry_utils import fold_to_vertical_magnitude
    return {
        "trunk_signed": signed,
        "trunk_from_vertical": fold_to_vertical_magnitude(signed),
        "neck_signed": 0.0,
        "neck_from_vertical": 0.0,
    }


# ---------------------------------------------------------------------------
# Range
# ---------------------------------------------------------------------------

def test_range_scores_100_inside_and_0_at_the_documented_distance():
    r = Range(10.0, 20.0)
    assert r.score(15.0, zero_at=30.0) == 100.0
    assert r.score(20.0 + 30.0, zero_at=30.0) == 0.0
    assert r.score(10.0 - 30.0, zero_at=30.0) == 0.0


def test_range_penalty_is_linear_and_symmetric_in_degrees():
    r = Range(10.0, 20.0)
    assert r.score(25.0, 40.0) == pytest.approx(r.score(5.0, 40.0))
    # 10 deg past max=20, zero at 40 deg out -> 100 - 10/40*100
    assert r.score(30.0, 40.0) == pytest.approx(75.0)


def test_range_never_scores_below_zero():
    assert Range(0.0, 10.0).score(1000.0, 30.0) == 0.0


def test_desk_slope_reproduces_the_old_two_and_a_half_points_per_degree():
    """posture_scoring used `min(100, deviation * 2.5)`, which is zero_at=40."""
    r = DESK.ideal["trunk_from_vertical"]
    deviation = 8.0
    assert r.score(r.max + deviation, DESK.score_zero_at_deg) == pytest.approx(
        100.0 - deviation * 2.5)


# ---------------------------------------------------------------------------
# weighted_score
# ---------------------------------------------------------------------------

def test_weighted_score_ignores_angles_that_were_not_supplied():
    """An upper-body-only monitor must not be punished for having no legs."""
    torso_only = {**torso(-17.5), "neck_signed": 0.0}
    full = {**torso_only, **LEGS}
    assert DRIVING.weighted_score(torso_only)[0] == pytest.approx(
        DRIVING.weighted_score(full)[0])


def test_weighted_score_is_zero_when_nothing_is_supplied():
    assert DRIVING.weighted_score({})[0] == 0.0


def test_left_and_right_carry_equal_weight():
    """The old driving weights gave the left knee 0.15 and the right 0.10, so
    the score depended on which leg the driver moved."""
    for reference in (DRIVING, DESK):
        for left, right in (("left_hip_angle", "right_hip_angle"),
                            ("left_knee_angle", "right_knee_angle")):
            assert reference.weights[left] == reference.weights[right], reference.name


def test_driving_weights_sum_to_one():
    assert sum(DRIVING.weights.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Risk rules
# ---------------------------------------------------------------------------

def test_no_risk_fires_on_a_posture_the_model_calls_ideal():
    """The regression this consolidation was for.

    The risk bounds used to be written separately from the ideal ranges and had
    drifted: the forward-lean flag sat at +20 while the ideal range ended at
    +15, so a trunk at +18 scored imperfect with no risk and no explanation.
    """
    for reference in (DRIVING, DESK):
        ideal_posture = {
            angle: (rng.min + rng.max) / 2.0
            for angle, rng in reference.ideal.items()
        }
        assert reference.risks(ideal_posture) == [], reference.name


def test_every_risk_rule_lies_outside_its_own_ideal_range():
    for reference in (DRIVING, DESK):
        for rule in reference.risk_rules:
            rng = reference.ideal.get(rule.angle)
            if rng is None:
                continue
            if rule.below is not None:
                assert rule.below <= rng.min, (reference.name, rule)
            if rule.above is not None:
                assert rule.above >= rng.max, (reference.name, rule)


def test_risk_rule_skips_an_angle_it_was_not_given():
    rule = RiskRule("trunk_signed", "x", 10, above=5.0)
    assert rule.fires({}) is False
    assert rule.fires({"trunk_signed": 6.0}) is True


def test_neck_risk_is_symmetric_because_it_uses_the_magnitude():
    assert DRIVING.risk_score({"neck_signed": 40.0}) == DRIVING.risk_score(
        {"neck_signed": -40.0})


def test_risk_score_is_capped_at_100():
    everything_wrong = {
        "trunk_signed": 90.0, "neck_signed": 90.0,
        "left_hip_angle": 10.0, "right_hip_angle": 10.0,
        "left_knee_angle": 10.0, "right_knee_angle": 10.0,
    }
    assert DRIVING.risk_score(everything_wrong) == 100.0


# ---------------------------------------------------------------------------
# The calibration: what "good" means in a car
# ---------------------------------------------------------------------------

def test_driving_ideal_trunk_is_entirely_reclined():
    """Automotive guidance puts the torso 5-30 deg BEHIND vertical (Grandjean
    20-25, SAE J1100 A40 ~22-25, Rebiffe). The previous range was -5..15, which
    made bolt upright perfect and the designed-for posture a fault."""
    trunk = DRIVING.ideal["trunk_signed"]
    assert trunk.max < 0.0
    assert trunk.min >= -35.0
    assert not trunk.contains(0.0)


def test_desk_ideal_trunk_is_upright():
    assert DESK.ideal["trunk_from_vertical"].contains(0.0)


def test_driving_hip_and_knee_match_the_cited_ranges():
    """Rebiffe (1969): trunk-thigh 95-120, knee 95-135."""
    assert (DRIVING.ideal["left_hip_angle"].min,
            DRIVING.ideal["left_hip_angle"].max) == (95.0, 120.0)
    assert (DRIVING.ideal["left_knee_angle"].min,
            DRIVING.ideal["left_knee_angle"].max) == (95.0, 135.0)


def test_driving_judges_slouch_on_the_signed_angle_and_desk_on_the_magnitude():
    assert DRIVING.slouch_key == "trunk_signed"
    assert DESK.slouch_key == "trunk_from_vertical"


# ---------------------------------------------------------------------------
# One definition, many consumers
# ---------------------------------------------------------------------------

RECLINED_DRIVER = {**torso(-25.0), **LEGS, **RELAXED_SHOULDERS}
HUNCHED_DRIVER = {**torso(25.0), **LEGS, **RELAXED_SHOULDERS}


def test_reclined_driver_is_neutral_under_driving_and_slouch_under_desk():
    assert classify_upper_body(RECLINED_DRIVER, DRIVING) == "Neutral"
    assert classify_upper_body(RECLINED_DRIVER, DESK) == "Slouch"


def test_hunched_driver_is_slouch_under_both():
    assert classify_upper_body(HUNCHED_DRIVER, DRIVING) == "Slouch"
    assert classify_upper_body(HUNCHED_DRIVER, DESK) == "Slouch"


def test_classify_upper_body_threshold_is_the_references_own():
    for reference in (DRIVING, DESK):
        key = reference.slouch_key
        at = {**LEGS, **RELAXED_SHOULDERS, key: reference.slouch_deg}
        over = {**LEGS, **RELAXED_SHOULDERS, key: reference.slouch_deg + 0.1}
        assert classify_upper_body(at, reference) != "Slouch", reference.name
        assert classify_upper_body(over, reference) == "Slouch", reference.name


def test_label_and_advice_fire_at_the_same_threshold():
    """They used to disagree: the label said ForwardHead at 20 degrees while
    feedback_lines stayed silent until 25."""
    for reference in (DRIVING, DESK):
        key = reference.forward_head_key
        angles = {**LEGS, **RELAXED_SHOULDERS, key: reference.forward_head_deg + 1.0}
        assert classify_upper_body(angles, reference) == "ForwardHead"
        assert any("forward head" in line.lower()
                   for line in feedback_lines(angles, reference)), reference.name


def test_feedback_escalates_to_severe_one_mild_band_past_the_threshold():
    reference = DESK
    key = reference.slouch_key
    mild = {**LEGS, **RELAXED_SHOULDERS, key: reference.slouch_deg + 1.0}
    severe = {**LEGS, **RELAXED_SHOULDERS,
              key: reference.slouch_deg + MILD_BAND_DEG + 1.0}
    assert any("mild slouching" in l.lower() for l in feedback_lines(mild, reference))
    assert any("severe slouching" in l.lower() for l in feedback_lines(severe, reference))


def test_reclined_driver_gets_no_complaints_under_the_driving_model():
    assert feedback_lines(RECLINED_DRIVER, DRIVING) == ["All Clear: Good posture."]


def test_posture_rules_defaults_come_from_the_default_reference():
    cfg = PostureConfig()
    reference = driver_model.DEFAULT_REFERENCE
    assert cfg.trunk_slouch_deg == reference.slouch_deg
    assert cfg.neck_forward_deg == reference.forward_head_deg
    assert cfg.shoulder_elev_deg == reference.shoulder_elev_deg
    assert cfg.slouch_key == reference.slouch_key


def test_posture_config_from_reference_tracks_the_driving_model():
    cfg = PostureConfig.from_reference(DRIVING)
    assert cfg.slouch_key == DRIVING.slouch_key
    assert cfg.trunk_slouch_deg == DRIVING.slouch_deg
    # Seated detection derived from the reference's own seated ranges.
    assert cfg.hip_angle_target - cfg.hip_angle_tol == pytest.approx(
        DRIVING.seated_hip.min)
    assert cfg.hip_angle_target + cfg.hip_angle_tol == pytest.approx(
        DRIVING.seated_hip.max)


def test_classify_posture_follows_the_configs_reference():
    desk_cfg = PostureConfig.from_reference(DESK)
    driving_cfg = PostureConfig.from_reference(DRIVING)
    assert classify_posture(RECLINED_DRIVER, driving_cfg)[0] == "Neutral"
    assert classify_posture(RECLINED_DRIVER, desk_cfg)[0] == "Slouch"


def test_quality_scorer_ranges_are_the_references():
    scorer = PostureQualityScorer(DESK)
    for angle, (lo, hi) in scorer.ideal_ranges.items():
        assert (lo, hi) == (DESK.ideal[angle].min, DESK.ideal[angle].max)


def test_quality_scorer_rejects_a_reference_it_cannot_score():
    """It groups magnitude angles into desk categories; DRIVING defines neither
    '*_from_vertical' nor shoulder elevation, so it must say so rather than
    KeyError deep inside a category calculation."""
    with pytest.raises(ValueError, match="DriverComfortAnalyzer"):
        PostureQualityScorer(DRIVING)


# ---------------------------------------------------------------------------
# torso_view
# ---------------------------------------------------------------------------

def test_magnitude_is_derived_from_a_signed_value():
    assert torso_view({"trunk_signed": -25.0}, "trunk_from_vertical") == pytest.approx(25.0)


def test_signed_is_never_invented_from_a_magnitude():
    """The direction a legacy log leaned is genuinely unrecoverable."""
    assert torso_view({"trunk_from_vertical": 25.0}, "trunk_signed") is None


def test_an_explicit_value_wins_over_derivation():
    angles = {"trunk_signed": -25.0, "trunk_from_vertical": 3.0}
    assert torso_view(angles, "trunk_from_vertical") == 3.0


def test_a_rule_whose_view_is_missing_is_skipped_not_guessed():
    """A replayed legacy log has no signed keys, so the DRIVING slouch rule
    cannot be evaluated - and must not fall through to a wrong label."""
    legacy = {**LEGS, **RELAXED_SHOULDERS, "trunk_from_vertical": 40.0,
              "neck_from_vertical": 2.0}
    assert classify_upper_body(legacy, DRIVING) == "Neutral"
    assert classify_upper_body(legacy, DESK) == "Slouch"


# ---------------------------------------------------------------------------
# Lookup, config profiles, provenance
# ---------------------------------------------------------------------------

def test_get_reference_by_name_and_none():
    assert get_reference("driving") is DRIVING
    assert get_reference("desk") is DESK
    assert get_reference(None) is driver_model.DEFAULT_REFERENCE


def test_get_reference_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="Unknown posture reference"):
        get_reference("sofa")


def test_driving_preset_names_the_reference_and_overrides_nothing():
    preset = config_manager.ConfigManager().get_ergonomic_presets()["driving"]
    assert preset.posture_reference == DRIVING.name
    assert preset.to_posture_config().slouch_key == DRIVING.slouch_key
    # The automotive numbers live in driver_model, not in the preset.
    assert preset.trunk_slouch_threshold is None
    assert preset.neck_forward_threshold is None


def test_a_profile_without_a_reference_keeps_the_monitors_own():
    """So `--config user` cannot silently switch the in-car monitor to DESK."""
    profile = config_manager.PostureConfig()
    assert profile.posture_reference is None
    assert profile.reference(DRIVING) is DRIVING
    assert profile.to_posture_config(DRIVING).slouch_key == DRIVING.slouch_key


def test_a_profile_override_beats_the_reference():
    profile = config_manager.PostureConfig(trunk_slouch_threshold=3.0)
    cfg = profile.to_posture_config(DRIVING)
    assert cfg.trunk_slouch_deg == 3.0
    # and still reads the driving angle view
    assert cfg.slouch_key == DRIVING.slouch_key


def test_unknown_reference_in_a_profile_is_a_validation_error():
    manager = config_manager.ConfigManager()
    bad = config_manager.PostureConfig(posture_reference="sofa")
    assert any("posture_reference" in issue for issue in manager.validate_config(bad))
    assert manager.validate_config(config_manager.PostureConfig()) == []


def test_sidecar_records_which_reference_produced_the_labels(tmp_path):
    csv_path = tmp_path / "driver_comfort_20260101_000000.csv"
    csv_path.write_text("x\n", encoding="utf-8")
    SessionMetadata(session_id="s", started_at="t", app="a",
                    posture_reference=DRIVING.name).save(csv_path)
    assert SessionMetadata.load(csv_path).posture_reference == DRIVING.name
