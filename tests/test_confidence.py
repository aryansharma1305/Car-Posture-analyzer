"""Tests for measurement confidence and frame exclusion.

MediaPipe returns a landmark for every joint on every frame whether or not it
saw one. Before this the pipeline had no way to say "unknown": an angle it could
not compute came back as 0.0, which means "perfectly upright" to
angle_with_vertical and "fully folded joint" to angle_3pts. Both are
measurements, and both scored.

These tests pin the replacement end to end: unknown is NaN, NaN is never
compared against a threshold, NaN never reaches a CSV as text, and a frame that
could not be measured is excluded from statistics rather than averaged into
them.
"""
import csv
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import landmark_confidence  # noqa: E402
from driver_comfort_analyzer import DriverComfortAnalyzer  # noqa: E402
from driver_model import DESK, DRIVING, MIN_MEASURED_FRACTION, Range  # noqa: E402
from ergonomics_scores import (  # noqa: E402
    NO_DATA_CATEGORY,
    compute_reba,
    compute_rula,
    feedback_lines,
    risk_buckets,
)
from geometry_utils import (  # noqa: E402
    DEGENERATE_ANGLE,
    angle_3pts,
    angle_with_vertical,
    format_angle,
    is_measured,
)
from landmark_confidence import (  # noqa: E402
    ANGLE_LANDMARKS,
    VISIBILITY_THRESHOLD,
    angle_confidence,
    frame_confidence,
    unmeasurable_angles,
)
from posture_angles import compute_angles  # noqa: E402
from posture_fullbody_rules import (  # noqa: E402
    classify_upper_body,
    detect_body_state,
)
from posture_scoring import PostureQualityScorer  # noqa: E402
from session_schema import COLUMNS, SCHEMA_VERSION, SessionMetadata, SessionWriter  # noqa: E402
from smoothing import AngleSmoother  # noqa: E402

NAN = float("nan")

# A seated subject in pixel space (y grows downward), reclined ~18 degrees.
PTS = {k: np.array(v, dtype=float) for k, v in {
    "nose": (290, 165), "left_ear": (295, 160), "right_ear": (295, 160),
    "left_shoulder": (300, 200), "right_shoulder": (300, 200),
    "left_elbow": (310, 270), "right_elbow": (310, 270),
    "left_wrist": (320, 300), "right_wrist": (320, 300),
    "left_hip": (340, 320), "right_hip": (340, 320),
    "left_knee": (250, 360), "right_knee": (250, 360),
    "left_ankle": (200, 420), "right_ankle": (200, 420),
}.items()}

ALL_VISIBLE = {k: 0.95 for k in PTS}
LEGS_HIDDEN = {**ALL_VISIBLE, "left_knee": 0.1, "right_knee": 0.1,
               "left_ankle": 0.05, "right_ankle": 0.05}

FULL_ANGLES = {
    "trunk_signed": -20.0, "trunk_from_vertical": 20.0,
    "neck_signed": 0.0, "neck_from_vertical": 0.0,
    "left_hip_angle": 105.0, "right_hip_angle": 105.0,
    "left_knee_angle": 115.0, "right_knee_angle": 115.0,
    "left_shoulder_elev": 10.0, "right_shoulder_elev": 10.0,
}
NO_LEGS = {**FULL_ANGLES, "left_hip_angle": NAN, "right_hip_angle": NAN,
           "left_knee_angle": NAN, "right_knee_angle": NAN}
NOTHING = {k: NAN for k in FULL_ANGLES}

# FULL_ANGLES is a good DRIVING posture, which is a mediocre DESK one by
# construction (20 degrees of recline, knees at 115). For desk assertions that
# need every joint inside its range, use this instead.
DESK_IDEAL = {
    "trunk_signed": -5.0, "trunk_from_vertical": 5.0,
    "neck_signed": 5.0, "neck_from_vertical": 5.0,
    "left_hip_angle": 95.0, "right_hip_angle": 95.0,
    "left_knee_angle": 95.0, "right_knee_angle": 95.0,
    "left_shoulder_elev": 10.0, "right_shoulder_elev": 10.0,
}
DESK_IDEAL_NO_LEGS = {**DESK_IDEAL, "left_hip_angle": NAN, "right_hip_angle": NAN,
                      "left_knee_angle": NAN, "right_knee_angle": NAN}


# ---------------------------------------------------------------------------
# The sentinel
# ---------------------------------------------------------------------------

def test_degenerate_angle_is_nan():
    assert math.isnan(DEGENERATE_ANGLE)


def test_a_lost_landmark_no_longer_reads_as_perfectly_upright():
    """The headline regression. Two coincident landmarks used to give 0.0, and
    0.0 from angle_with_vertical is the ideal desk posture."""
    collapsed = angle_with_vertical((5, 5), (5, 5))
    assert not is_measured(collapsed)
    assert DESK.angle_score("trunk_from_vertical", collapsed) != 100.0
    assert math.isnan(DESK.angle_score("trunk_from_vertical", collapsed))


def test_a_lost_landmark_no_longer_reads_as_a_folded_joint():
    """0.0 from angle_3pts is a fully closed knee, which fires a strain risk."""
    collapsed = angle_3pts(PTS["left_hip"], PTS["left_hip"], PTS["left_hip"])
    assert not is_measured(collapsed)
    assert DRIVING.risks({"left_knee_angle": collapsed}) == []
    assert DRIVING.risks({"left_knee_angle": 0.0}) != []


@pytest.mark.parametrize("value, expected", [
    (0.0, True), (-20.0, True), (NAN, False), (None, False), ("", False),
])
def test_is_measured(value, expected):
    assert is_measured(value) is expected


def test_format_angle_never_shows_nan_to_a_user():
    assert format_angle(NAN) == "--"
    assert format_angle(None) == "--"
    assert format_angle(-20.04) == "-20.0"


# ---------------------------------------------------------------------------
# landmark_confidence
# ---------------------------------------------------------------------------

def test_angle_confidence_is_the_minimum_over_its_landmarks():
    """Not the mean: two confident landmarks must not carry an invented third."""
    vis = {**ALL_VISIBLE, "left_knee": 0.1}
    scores = angle_confidence(vis)
    assert scores["left_knee_angle"] == pytest.approx(0.1)
    assert scores["right_knee_angle"] == pytest.approx(0.95)


def test_a_landmark_absent_from_visibility_scores_zero():
    vis = {k: v for k, v in ALL_VISIBLE.items() if k != "left_hip"}
    assert angle_confidence(vis)["left_hip_angle"] == 0.0


def test_derived_magnitude_inherits_its_signed_confidence():
    scores = angle_confidence(ALL_VISIBLE)
    assert scores["trunk_from_vertical"] == scores["trunk_signed"]
    assert scores["neck_from_vertical"] == scores["neck_signed"]


def test_every_angle_the_pipeline_emits_declares_its_landmarks():
    """A new angle without an entry here would silently never be masked."""
    emitted = set(compute_angles(PTS))
    declared = set(ANGLE_LANDMARKS) | set(landmark_confidence.DERIVED_ANGLES)
    assert emitted == declared


def test_declared_landmarks_all_exist_in_the_pose_vocabulary():
    """The only test that needs the pose stack installed.

    Skipped rather than failed when MediaPipe is absent, so a contributor on an
    unsupported interpreter can still run the other 276. CI asserts the imports
    in a separate step, so this skip cannot happen there and hide a broken
    install.
    """
    pose_core = pytest.importorskip(
        "pose_core", reason="needs mediapipe and opencv")
    needed = set().union(*ANGLE_LANDMARKS.values())
    assert needed <= set(pose_core.LANDMARK_INDEX)


def test_no_visibility_information_masks_nothing():
    """A caller replaying landmark-only data must get the unmasked behaviour,
    not a blank frame."""
    assert angle_confidence(None) == {}
    assert unmeasurable_angles(None) == set()
    assert frame_confidence(None) is None


def test_frame_confidence_is_the_worst_angle():
    assert frame_confidence(LEGS_HIDDEN) == pytest.approx(0.05)
    assert frame_confidence(LEGS_HIDDEN, ["trunk_signed"]) == pytest.approx(0.95)


def test_threshold_is_inclusive_at_the_boundary():
    at = {k: VISIBILITY_THRESHOLD for k in ALL_VISIBLE}
    just_over = {k: VISIBILITY_THRESHOLD + 0.01 for k in ALL_VISIBLE}
    assert unmeasurable_angles(at)
    assert unmeasurable_angles(just_over) == set()


# ---------------------------------------------------------------------------
# compute_angles masking
# ---------------------------------------------------------------------------

def test_occluded_legs_produce_nan_not_extrapolated_angles():
    masked = compute_angles(PTS, True, LEGS_HIDDEN)
    unmasked = compute_angles(PTS, True)
    for angle in ("left_knee_angle", "right_knee_angle"):
        assert is_measured(unmasked[angle]), "fixture should be measurable"
        assert not is_measured(masked[angle])


def test_a_hidden_knee_also_invalidates_the_hip_angle():
    """The hip angle is shoulder-hip-knee, so an invented knee invents it too."""
    masked = compute_angles(PTS, True, LEGS_HIDDEN)
    assert not is_measured(masked["left_hip_angle"])


def test_masking_leaves_the_torso_alone():
    masked = compute_angles(PTS, True, LEGS_HIDDEN)
    unmasked = compute_angles(PTS, True)
    for angle in ("trunk_signed", "trunk_from_vertical", "neck_signed"):
        assert masked[angle] == pytest.approx(unmasked[angle])


def test_masking_is_off_when_no_visibility_is_given():
    angles = compute_angles(PTS, True)
    assert all(is_measured(v) for v in angles.values())


# ---------------------------------------------------------------------------
# Smoothing must not be poisoned
# ---------------------------------------------------------------------------

def test_one_nan_does_not_poison_the_filter_forever():
    """`alpha * nan + (1 - alpha) * state` is NaN, so the old EMA never
    recovered once a landmark was lost."""
    sm = AngleSmoother(alpha=0.5)
    for value in (10.0, 12.0, NAN, NAN, NAN):
        sm({"trunk_signed": value})
    after = sm({"trunk_signed": 14.0})["trunk_signed"]
    assert is_measured(after)
    assert after == pytest.approx(0.5 * 14.0 + 0.5 * 11.0)


def test_an_unmeasured_sample_is_passed_through_as_unmeasured():
    """Substituting the last good value would make a frozen angle look steady."""
    sm = AngleSmoother(alpha=0.5)
    sm({"trunk_signed": 10.0})
    assert not is_measured(sm({"trunk_signed": NAN})["trunk_signed"])
    assert sm.last_good("trunk_signed") == pytest.approx(10.0)


def test_gap_counts_consecutive_unmeasured_samples():
    sm = AngleSmoother()
    sm({"x": 1.0})
    assert sm.gap("x") == 0
    sm({"x": NAN})
    sm({"x": NAN})
    assert sm.gap("x") == 2
    sm({"x": 2.0})
    assert sm.gap("x") == 0


def test_a_key_that_was_never_measured_has_no_last_good_value():
    """The old implementation did float(None) here and raised TypeError."""
    sm = AngleSmoother()
    assert not is_measured(sm({"x": NAN})["x"])
    assert sm.last_good("x") is None


def test_resync_keeps_an_unmeasured_torso_unmeasured():
    from posture_angles import resync_magnitude_views
    out = resync_magnitude_views({"trunk_signed": NAN, "trunk_from_vertical": 5.0})
    assert not is_measured(out["trunk_from_vertical"])


# ---------------------------------------------------------------------------
# driver_model gating
# ---------------------------------------------------------------------------

def test_range_does_not_claim_an_unmeasured_value_is_inside_or_outside():
    r = Range(0.0, 10.0)
    assert r.contains(5.0) is True
    assert r.contains(NAN) is False
    assert math.isnan(r.score(NAN, 30.0))


def test_a_risk_rule_cannot_fire_on_an_unmeasured_angle():
    """NaN fails every comparison, which looks identical to "within limits"."""
    assert DRIVING.risks(NOTHING) == []


def test_weighted_score_drops_unmeasured_angles_from_the_denominator():
    partial, scores = DRIVING.weighted_score(NO_LEGS)
    assert set(scores) == {"trunk_signed", "neck_signed"}
    assert partial == pytest.approx(DRIVING.weighted_score(FULL_ANGLES)[0])


def test_measured_fraction_reports_how_much_of_the_model_was_seen():
    assert DRIVING.measured_fraction(FULL_ANGLES) == pytest.approx(1.0)
    assert DRIVING.measured_fraction(NOTHING) == 0.0
    assert 0.0 < DRIVING.measured_fraction(NO_LEGS) < 1.0


def test_unmeasured_lists_the_angles_the_frame_could_not_supply():
    assert DRIVING.unmeasured(NO_LEGS) == [
        "left_hip_angle", "left_knee_angle", "right_hip_angle", "right_knee_angle"]
    assert DRIVING.unmeasured(FULL_ANGLES) == []


def test_driving_requires_the_torso_line_whatever_else_was_measured():
    """A frame that saw only legs is not a weaker reading of the seat model, it
    is a reading of something else - so weight alone cannot qualify it."""
    legs_only = {**FULL_ANGLES, "trunk_signed": NAN, "neck_signed": NAN}
    assert DRIVING.measured_fraction(legs_only) >= MIN_MEASURED_FRACTION
    assert DRIVING.missing_required(legs_only) == ["trunk_signed"]
    assert DRIVING.is_observation(legs_only) is False


def test_a_frame_with_enough_weight_and_its_required_angles_is_an_observation():
    assert DRIVING.is_observation(FULL_ANGLES) is True


def test_torso_view_reports_an_unmeasured_angle_as_absent():
    from driver_model import torso_view
    assert torso_view({"trunk_signed": NAN}, "trunk_signed") is None
    assert torso_view({"trunk_signed": NAN}, "trunk_from_vertical") is None


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def test_body_state_is_unknown_without_a_trunk_angle():
    state, extras = detect_body_state(PTS, NOTHING, ALL_VISIBLE, (480, 640))
    assert state == "Unknown"
    assert extras["upright_deg"] is None


def test_masked_leg_angles_mark_the_lower_body_as_not_visible():
    _, extras = detect_body_state(PTS, NO_LEGS, ALL_VISIBLE, (480, 640))
    assert extras["lower_visible"] is False


def test_upper_body_label_is_unknown_when_nothing_was_measurable():
    """"Neutral" would be a claim rather than an observation."""
    assert classify_upper_body(NOTHING, DRIVING) == "Unknown"
    assert classify_upper_body(NOTHING, DESK) == "Unknown"


@pytest.mark.parametrize("scorer", [compute_rula, compute_reba])
def test_rula_and_reba_report_no_data_instead_of_scoring_zeros(scorer):
    """`torso_view(...) or 0.0` bucketed an unseen neck as perfectly neutral,
    and the frame came back Acceptable."""
    category, label = scorer(NOTHING)
    assert category == NO_DATA_CATEGORY
    assert label == "No data"


def test_reba_needs_legs():
    assert compute_reba(NO_LEGS)[0] == NO_DATA_CATEGORY
    assert compute_rula(NO_LEGS)[0] != NO_DATA_CATEGORY


def test_risk_buckets_reports_unknown_separately_from_neutral():
    assert risk_buckets(NO_DATA_CATEGORY, 1) == "unknown"
    assert risk_buckets(1, NO_DATA_CATEGORY) == "unknown"
    assert risk_buckets(1, 1) == "good"


def test_advice_distinguishes_nothing_wrong_from_nothing_measured():
    assert feedback_lines(FULL_ANGLES, DRIVING) == ["All Clear: Good posture."]
    assert feedback_lines(NOTHING, DRIVING) == ["No reading: landmarks not visible."]


# ---------------------------------------------------------------------------
# Desk quality scoring
# ---------------------------------------------------------------------------

def test_an_unmeasured_category_scores_none_not_zero_and_not_nan():
    score = PostureQualityScorer(DESK).score_posture(NO_LEGS)
    assert score.category_scores["lower_body"] is None
    assert score.unmeasured_categories == ["lower_body"]
    assert is_measured(score.overall_score)


def test_overall_score_is_renormalised_over_measured_categories():
    scorer = PostureQualityScorer(DESK)
    full = scorer.score_posture(DESK_IDEAL)
    partial = scorer.score_posture(DESK_IDEAL_NO_LEGS)
    # Every measured joint is inside its range in both, so dropping the legs
    # must not move the score - only the coverage.
    assert full.overall_score == pytest.approx(100.0)
    assert partial.overall_score == pytest.approx(full.overall_score)
    assert partial.measured_fraction < full.measured_fraction


def test_a_missing_category_is_neither_treated_as_zero_nor_as_perfect():
    """Both would be a claim. 0 drags a good posture down, 100 flatters a bad
    one; renormalising does neither."""
    scorer = PostureQualityScorer(DESK)
    bad_upper = {**DESK_IDEAL, "trunk_from_vertical": 40.0, "trunk_signed": 40.0,
                 "neck_from_vertical": 40.0, "neck_signed": 40.0}
    with_legs = scorer.score_posture(bad_upper)
    without_legs = scorer.score_posture({
        **bad_upper, "left_hip_angle": NAN, "right_hip_angle": NAN,
        "left_knee_angle": NAN, "right_knee_angle": NAN})
    # Dropping ideal legs from a bad-upper-body frame must make the score worse,
    # because the legs were the part propping it up.
    assert without_legs.overall_score < with_legs.overall_score


def test_risk_level_is_unknown_without_the_required_angles():
    score = PostureQualityScorer(DESK).score_posture(NOTHING)
    assert score.risk_level == "Unknown"
    assert score.measured_fraction == 0.0


def test_unmeasured_categories_are_named_in_the_recommendations():
    score = PostureQualityScorer(DESK).score_posture(NO_LEGS)
    assert "lower body" in score.recommendations[0]


# ---------------------------------------------------------------------------
# Driver comfort: frame exclusion
# ---------------------------------------------------------------------------

@pytest.fixture
def analyzer():
    return DriverComfortAnalyzer(DRIVING)


def test_a_frame_below_the_threshold_is_not_given_a_comfort_category(analyzer):
    score = analyzer.calculate_comfort_score(NO_LEGS)
    assert score.is_observation is False
    assert score.comfort_category == "Insufficient data"
    assert score.measured_fraction < MIN_MEASURED_FRACTION


def test_a_blank_frame_scores_nan_not_a_plausible_number(analyzer):
    """With nothing measured the posture term was 0 and the risk term was a
    clean 100, which combined to a confident-looking 37.5."""
    score = analyzer.calculate_comfort_score(NOTHING)
    assert not is_measured(score.overall_comfort)
    assert score.comfort_category == "Insufficient data"


def test_a_full_frame_is_still_scored_normally(analyzer):
    score = analyzer.calculate_comfort_score(FULL_ANGLES)
    assert score.is_observation is True
    assert score.comfort_category in {"Excellent", "Good", "Fair", "Poor"}
    assert score.measured_fraction == 1.0


def test_unmeasured_angles_are_named_in_the_recommendations(analyzer):
    score = analyzer.calculate_comfort_score(NO_LEGS)
    assert "left_knee_angle" in score.recommendations[0]
    assert len(score.recommendations) <= 5


def test_fatigue_drift_ignores_unmeasured_samples(analyzer):
    """np.std over a window containing one NaN returns NaN, which propagated
    into overall_comfort."""
    history = [{"trunk_signed": (NAN if i % 3 == 0 else 20.0), "neck_signed": 0.0}
               for i in range(20)]
    score = analyzer.calculate_comfort_score(FULL_ANGLES, history, 600)
    assert is_measured(score.overall_comfort)
    assert is_measured(score.fatigue_indicator)


def test_session_report_excludes_unobservable_frames(analyzer):
    good = [{"angles": dict(FULL_ANGLES)} for _ in range(20)]
    bad = [{"angles": dict(NO_LEGS)} for _ in range(5)]
    report = analyzer.generate_comfort_report(good + bad, {"duration": 25})
    assert report["data_points"] == 20
    assert report["excluded_data_points"] == 5
    assert report["measurement_coverage"]["usable_fraction"] == pytest.approx(0.8)
    assert report["measurement_coverage"]["angles_most_often_missing"][
        "left_knee_angle"] == 5


def test_a_session_with_no_usable_frame_reports_an_error_not_a_score(analyzer):
    report = analyzer.generate_comfort_report(
        [{"angles": dict(NOTHING)} for _ in range(20)], {"duration": 20})
    assert "error" in report
    assert report["excluded_data_points"] == 20
    assert "summary" not in report


def test_seat_comparison_excludes_unobservable_frames(analyzer):
    result = analyzer.analyze_seat_comparison({
        "SEAT-A": [{"angles": dict(FULL_ANGLES)} for _ in range(10)]
                  + [{"angles": dict(NO_LEGS)} for _ in range(4)],
    })
    assert result["SEAT-A"]["sample_size"] == 10
    assert result["SEAT-A"]["excluded_samples"] == 4


def test_a_seat_with_only_unusable_frames_is_not_given_an_average(analyzer):
    result = analyzer.analyze_seat_comparison({
        "SEAT-B": [{"angles": dict(NOTHING)} for _ in range(5)],
    })
    assert result["SEAT-B"]["sample_size"] == 0
    assert "average_comfort" not in result["SEAT-B"]


# ---------------------------------------------------------------------------
# The log
# ---------------------------------------------------------------------------

def read_one(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[0]


def write_one(tmp_path, angles, visibility=None, reference=DRIVING, comfort=None):
    path = tmp_path / "driver_comfort_20260101_000000.csv"
    writer = SessionWriter(path, metadata=SessionMetadata(
        session_id=path.stem, started_at="t", app="test",
        posture_reference=reference.name))
    writer.write(t_sec=1.0, angles=angles, visibility=visibility,
                 reference=reference, comfort=comfort)
    return read_one(path)


def test_schema_version_is_three_and_the_columns_are_appended():
    assert SCHEMA_VERSION == 3
    assert COLUMNS[-4:] == ["frame_confidence", "measured_fraction",
                            "unmeasured", "observation"]


def test_an_unmeasured_angle_is_written_as_an_empty_cell(tmp_path):
    """round(nan, 1) put the literal text "nan" in the CSV."""
    row = write_one(tmp_path, NO_LEGS, LEGS_HIDDEN)
    assert row["knee_L"] == ""
    assert row["hip_L"] == ""
    assert "nan" not in ",".join(row.values()).lower()


def test_measured_angles_are_still_written(tmp_path):
    row = write_one(tmp_path, NO_LEGS, LEGS_HIDDEN)
    assert float(row["trunk_signed"]) == pytest.approx(-20.0)


def test_the_row_records_why_it_is_incomplete(tmp_path):
    row = write_one(tmp_path, NO_LEGS, LEGS_HIDDEN)
    assert row["observation"] == "False"
    assert float(row["measured_fraction"]) < 1.0
    assert "left_knee_angle" in row["unmeasured"].split("|")
    assert float(row["frame_confidence"]) == pytest.approx(0.05)


def test_a_complete_row_is_marked_as_an_observation(tmp_path):
    row = write_one(tmp_path, FULL_ANGLES, ALL_VISIBLE)
    assert row["observation"] == "True"
    assert float(row["measured_fraction"]) == 1.0
    assert row["unmeasured"] == ""


def test_confidence_is_blank_when_the_writer_had_no_visibility(tmp_path):
    """Not 0, which would claim every landmark was invisible."""
    row = write_one(tmp_path, FULL_ANGLES, visibility=None)
    assert row["frame_confidence"] == ""


def test_a_comfort_scores_own_coverage_is_what_gets_logged(tmp_path):
    comfort = DriverComfortAnalyzer(DRIVING).calculate_comfort_score(NO_LEGS)
    row = write_one(tmp_path, NO_LEGS, ALL_VISIBLE, comfort=comfort)
    assert row["observation"] == "False"
    assert float(row["measured_fraction"]) == pytest.approx(comfort.measured_fraction)
    assert row["comfort_category"] == "Insufficient data"


def test_a_nan_comfort_score_is_written_as_empty(tmp_path):
    comfort = DriverComfortAnalyzer(DRIVING).calculate_comfort_score(NOTHING)
    row = write_one(tmp_path, NOTHING, ALL_VISIBLE, comfort=comfort)
    assert row["comfort_score"] == ""
    assert row["trunk_signed"] == ""
