"""Tests for the angle-validation harness.

The harness is the thing that decides whether any other number in this project
means anything, so its own correctness has to be pinned first. Two properties
matter more than the rest:

  1. The sign conversion between "degrees of recline, positive" (how a seat is
     specified) and trunk_signed (negative = reclined). Getting it backwards
     would invert the sign of every conclusion while still producing a
     plausible-looking table.
  2. That aggregation uses ONE value per trial. Frames of a person sitting still
     are heavily correlated, so pooling them would shrink the apparent error bar
     by sqrt(frames) and make a sensor look far better than it is.
"""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import validate_angles as va  # noqa: E402
from driver_model import DRIVING  # noqa: E402
from posture_angles import compute_angles  # noqa: E402
from validation_geometry import seated_skeleton  # noqa: E402

NAN = float("nan")


def trial(recline, frames, subject="S1", source="sternum", trial_id="t"):
    return va.Trial(
        trial_id=trial_id, recorded_at="2026-01-01T00:00:00",
        recline_deg=float(recline), reference_source=source, subject_id=subject,
        frames_measured=len([f for f in frames if not math.isnan(f)]),
        frames_seen=len(frames),
        samples={"trunk_signed": list(frames)},
    )


# ---------------------------------------------------------------------------
# The sign convention
# ---------------------------------------------------------------------------

def test_recline_is_declared_positive_and_stored_negative():
    """A seat spec says "25 degrees of recline"; trunk_signed says -25."""
    assert va.expected_trunk_signed(25) == -25.0
    assert va.expected_trunk_signed(0) == 0.0


def test_the_expected_value_lands_inside_the_cited_driving_range():
    """A 20-25 degree backrest is the posture DRIVING calls ideal. If this ever
    fails, the harness and the model disagree about which way recline runs."""
    for recline in (10, 20, 25, 30):
        assert DRIVING.ideal["trunk_signed"].contains(
            va.expected_trunk_signed(recline)), recline


def test_an_upright_subject_is_outside_the_driving_ideal_range():
    assert not DRIVING.ideal["trunk_signed"].contains(va.expected_trunk_signed(0))


def test_error_is_positive_when_the_system_reads_more_upright():
    # Truth is 25 deg of recline (-25). The system says -20, i.e. less reclined.
    assert trial(25, [-20.0] * 5).error() == pytest.approx(5.0)
    # And negative when it over-reports recline.
    assert trial(25, [-30.0] * 5).error() == pytest.approx(-5.0)


# ---------------------------------------------------------------------------
# The synthetic control
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("recline", [0, 5, 10, 17.5, 25, 30, 45, 60])
def test_synthetic_geometry_round_trips_the_torso_angle_exactly(recline):
    angles = compute_angles(
        seated_skeleton(va.expected_trunk_signed(recline)), True)
    assert angles["trunk_signed"] == pytest.approx(
        va.expected_trunk_signed(recline), abs=1e-9)


@pytest.mark.parametrize("neck,hip,knee", [(0, 95, 95), (8, 105, 115), (-6, 120, 135)])
def test_synthetic_geometry_round_trips_the_limbs_too(neck, hip, knee):
    angles = compute_angles(seated_skeleton(-22.0, neck, hip, knee), True)
    assert angles["neck_signed"] == pytest.approx(neck, abs=1e-9)
    assert angles["left_hip_angle"] == pytest.approx(hip, abs=1e-9)
    assert angles["left_knee_angle"] == pytest.approx(knee, abs=1e-9)


def test_a_right_side_camera_mirrors_the_sign():
    skeleton = seated_skeleton(-25.0)
    assert compute_angles(skeleton, False)["trunk_signed"] == pytest.approx(
        -compute_angles(skeleton, True)["trunk_signed"])


def test_the_synthetic_skeleton_is_a_pure_side_view():
    """Left and right coincide, so any left/right discrepancy in a real trial is
    unambiguously the pose model's rather than the fixture's."""
    angles = compute_angles(seated_skeleton(-20.0), True)
    assert angles["left_hip_angle"] == pytest.approx(angles["right_hip_angle"])
    assert angles["left_knee_angle"] == pytest.approx(angles["right_knee_angle"])


def test_synth_command_passes(capsys):
    assert va.main(["synth"]) == 0
    assert "PASS" in capsys.readouterr().out


def test_synth_passes_from_either_camera_side(capsys):
    assert va.main(["synth", "--camera-position", "right-side"]) == 0
    assert "PASS" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Trial bookkeeping
# ---------------------------------------------------------------------------

def test_unmeasured_frames_are_excluded_from_the_trial_mean():
    """A NaN frame must not drag the mean, and must not make it NaN."""
    t = trial(25, [-24.0, NAN, -26.0, NAN])
    assert t.mean("trunk_signed") == pytest.approx(-25.0)
    assert t.error() == pytest.approx(0.0)


def test_a_trial_with_no_measured_frame_has_no_error():
    assert trial(25, [NAN, NAN]).error() is None


def test_error_is_only_defined_for_the_angle_the_reference_measures():
    assert trial(25, [-25.0]).error("left_knee_angle") is None


def test_trials_round_trip_through_the_jsonl_file(tmp_path):
    path = tmp_path / "trials.jsonl"
    va.save_trial(trial(25, [-23.0, -24.0], trial_id="a"), path)
    va.save_trial(trial(10, [-8.0], trial_id="b"), path)
    loaded = va.load_trials(path)
    assert [t.trial_id for t in loaded] == ["a", "b"]
    assert loaded[0].error() == pytest.approx(1.5)


def test_loading_tolerates_unknown_fields(tmp_path):
    """So a file written by a later version of the harness still reads."""
    path = tmp_path / "trials.jsonl"
    va.save_trial(trial(25, [-25.0]), path)
    data = json.loads(path.read_text().strip())
    data["some_future_field"] = 1
    path.write_text(json.dumps(data) + "\n")
    assert len(va.load_trials(path)) == 1


def test_no_trials_file_is_not_an_error(tmp_path):
    assert va.load_trials(tmp_path / "absent.jsonl") == []


# ---------------------------------------------------------------------------
# Aggregation: one value per trial
# ---------------------------------------------------------------------------

def write(path, trials):
    for t in trials:
        va.save_trial(t, path)
    return path


def test_aggregation_weights_every_trial_equally_regardless_of_frame_count(
        tmp_path, capsys):
    """The statistical crux. A 1000-frame trial and a 10-frame trial describe one
    sitting each; pooling frames would shrink the error bar by sqrt(n) and make
    the sensor look far better than it is."""
    path = write(tmp_path / "t.jsonl", [
        trial(25, [-20.0] * 1000, trial_id="many"),   # error +5
        trial(25, [-30.0] * 10, trial_id="few"),      # error -5
    ])
    va.main(["report", "--trials", str(path)])
    out = capsys.readouterr().out
    # Equal weight means the biases cancel to zero.
    assert "bias (mean signed error)   +0.00 deg" in out


def test_report_recovers_a_known_slope(tmp_path, capsys):
    """A sensor that under-reports recline by 20% must show slope 0.8, because a
    bias of zero would otherwise hide it."""
    trials = [trial(r, [0.8 * va.expected_trunk_signed(r)], trial_id=f"t{r}")
              for r in (0, 10, 20, 30, 40)]
    va.main(["report", "--trials", str(write(tmp_path / "t.jsonl", trials))])
    out = capsys.readouterr().out
    assert "measured = 0.800 x truth" in out
    assert "off unity" in out


def test_a_single_recline_angle_cannot_report_linearity(tmp_path, capsys):
    trials = [trial(25, [-24.0], trial_id="a"), trial(25, [-26.0], trial_id="b")]
    va.main(["report", "--trials", str(write(tmp_path / "t.jsonl", trials))])
    assert "linearity                  n/a" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def tight_trials(spread):
    """Five trials whose errors have a controlled spread, bias zero."""
    offsets = [-2 * spread, -spread, 0.0, spread, 2 * spread]
    return [trial(25, [-25.0 + o], trial_id=f"t{i}")
            for i, o in enumerate(offsets)]


def test_a_precise_sensor_is_called_usable(tmp_path, capsys):
    va.main(["report", "--trials",
             str(write(tmp_path / "t.jsonl", tight_trials(0.5)))])
    assert "VERDICT  Usable" in capsys.readouterr().out


def test_a_sensor_as_wide_as_the_model_cannot_resolve_it(tmp_path, capsys):
    """Agreement spanning the whole 25-degree ideal range means a frame scored
    Excellent could truly be outside it."""
    va.main(["report", "--trials",
             str(write(tmp_path / "t.jsonl", tight_trials(6.0)))])
    out = capsys.readouterr().out
    assert "cannot resolve the model" in out
    assert "Do not compare seats yet" in out


def test_one_trial_refuses_to_state_a_verdict(tmp_path, capsys):
    va.main(["report", "--trials",
             str(write(tmp_path / "t.jsonl", [trial(25, [-24.0])]))])
    out = capsys.readouterr().out
    assert "Only one trial" in out
    assert "limits of agreement    n/a" in out


def test_a_large_bias_is_called_out_as_correctable(tmp_path, capsys):
    trials = [trial(25, [-25.0 + 9.0], trial_id=f"t{i}") for i in range(4)]
    va.main(["report", "--trials", str(write(tmp_path / "t.jsonl", trials))])
    assert "systematic and correctable" in capsys.readouterr().out


def test_minimum_detectable_difference_shrinks_with_more_trials(tmp_path, capsys):
    va.main(["report", "--trials",
             str(write(tmp_path / "t.jsonl", tight_trials(2.0)))])
    out = capsys.readouterr().out
    values = [float(line.split("difference")[1].split("deg")[0])
              for line in out.splitlines() if "smallest resolvable difference" in line]
    assert len(values) == 4
    assert values == sorted(values, reverse=True)


# ---------------------------------------------------------------------------
# Filters and guards
# ---------------------------------------------------------------------------

def test_report_can_filter_by_reference_source(tmp_path, capsys):
    path = write(tmp_path / "t.jsonl", [
        trial(25, [-25.0], source="sternum", trial_id="s"),
        trial(25, [-15.0], source="backrest", trial_id="b"),
    ])
    va.main(["report", "--trials", str(path), "--reference-source", "sternum"])
    out = capsys.readouterr().out
    assert "1 trial(s)" in out
    assert "bias (mean signed error)   +0.00 deg" in out


def test_report_exits_nonzero_when_there_is_nothing_to_report(tmp_path, capsys):
    path = write(tmp_path / "t.jsonl", [trial(25, [NAN])])
    assert va.main(["report", "--trials", str(path)]) == 1
    assert "No usable trials" in capsys.readouterr().out


@pytest.mark.parametrize("position", ["front", "rear"])
def test_capture_refuses_a_front_or_rear_camera(position, capsys):
    """Leaning forward barely moves x in those views, so the sign the harness
    exists to validate carries almost no signal."""
    assert va.main(["capture", "--recline", "25", "--reference-source", "backrest",
                    "--subject-id", "S1", "--camera-position", position]) == 1
    assert "Refusing to capture" in capsys.readouterr().out


def test_protocol_names_the_sign_convention(capsys):
    assert va.main(["protocol"]) == 0
    out = capsys.readouterr().out
    assert "DEGREES OF RECLINE FROM VERTICAL, POSITIVE" in out
    assert "sternum" in out


def test_bare_invocation_prints_help_without_failing(capsys):
    assert va.main([]) == 0
    assert "validate" in capsys.readouterr().out.lower()
