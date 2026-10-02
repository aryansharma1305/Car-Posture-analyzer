#!/usr/bin/env python3
"""Measure the measurement: how well does this system recover a known angle?

Every number this project produces is an estimate of a joint angle from a 2-D
webcam, and until that estimate is compared against an independent reference the
accuracy of every downstream score is unknown. A seat comparison built on an
unvalidated sensor is not a finding - if the trunk estimate carries +-8 degrees
of error, the DRIVING ideal range (25 degrees wide) is barely wider than the
noise and a few points of difference between two seats means nothing.

This harness answers three questions, in order:

  1. Is the ARITHMETIC right?        `synth`   - no camera needed
  2. Is the SENSOR accurate?         `capture` then `report`
  3. Is it accurate ENOUGH?          `report`  - compares the limits of
                                     agreement against the width of the model's
                                     ideal range, and states the smallest seat
                                     difference the sensor can resolve.

Separating 1 from 2 matters. `synth` feeds compute_angles a skeleton whose true
angle is known exactly, so it isolates this project's own pixel-to-angle path.
It should show essentially zero error. Whatever `report` then shows for real
trials is attributable to the camera and MediaPipe, which is the thing you
actually want to characterise.

Reference angle convention
--------------------------
`--recline` is DEGREES OF RECLINE FROM VERTICAL, POSITIVE, which is how seat
specifications and inclinometer apps state it: an upright backrest is 0 and a
typical car backrest is 20-25.

Internally the project uses trunk_signed, where NEGATIVE means reclined, so a
trial declared `--recline 25` expects `trunk_signed == -25`. The conversion
happens in one place (expected_trunk_signed below) because getting it backwards
would invert the sign of every conclusion.

Reference sources, best to worst
-------------------------------
sternum   An inclinometer taped to the breastbone. Measures the TORSO, which is
          what the system claims to measure. Use this if you can.
backrest  The backrest angle, from a phone inclinometer app laid on the seat
          back. Easy, repeatable, and the usual way a seat is specified - but a
          person does not perfectly conform to a backrest, so it carries an
          extra subject-conformity term that inflates the apparent error.
photo     A protractor on a still side-view photo. Fine for a handful of points,
          not for a sweep.

The source is recorded per trial and `report` breaks results down by it, so the
conformity term shows up as a difference between sources rather than hiding
inside one pooled number.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from driver_model import DRIVING
from geometry_utils import is_measured
from landmark_confidence import VISIBILITY_THRESHOLD, frame_confidence
from posture_angles import compute_angles

TRIALS_PATH = Path("validation") / "trials.jsonl"

# Angles this harness validates. The torso pair is the point of the exercise;
# the limbs are included because they feed the comfort score too.
VALIDATED_ANGLES = ("trunk_signed", "neck_signed",
                    "left_hip_angle", "left_knee_angle")

REFERENCE_SOURCES = ("sternum", "backrest", "photo", "synthetic")

# Two-sided 95% coverage factor, used for limits of agreement.
Z95 = 1.96

# Approximate factor for the smallest difference two independent groups of n
# samples can resolve at 80% power, alpha 0.05: 2.8 * SD / sqrt(n) per group.
MDD_FACTOR = 2.8


def expected_trunk_signed(recline_deg: float) -> float:
    """The trunk_signed value a backrest reclined `recline_deg` should produce.

    One function so the sign convention is converted in exactly one place.
    """
    return -float(recline_deg)


@dataclass
class Trial:
    """One capture at one declared reference angle."""
    trial_id: str
    recorded_at: str
    recline_deg: float                  # declared truth, positive = reclined
    reference_source: str
    subject_id: str
    seat_id: Optional[str] = None
    camera_position: str = "left-side"
    camera_distance_cm: Optional[float] = None
    duration_s: float = 0.0
    frames_seen: int = 0
    frames_measured: int = 0
    mean_frame_confidence: Optional[float] = None
    # Per-angle raw per-frame samples. Raw, NOT smoothed: an EMA would hide the
    # scatter this harness exists to measure.
    samples: Dict[str, List[float]] = field(default_factory=dict)
    notes: Optional[str] = None

    def measured(self, angle: str) -> List[float]:
        return [v for v in self.samples.get(angle, []) if is_measured(v)]

    def mean(self, angle: str) -> Optional[float]:
        values = self.measured(angle)
        return statistics.fmean(values) if values else None

    def error(self, angle: str = "trunk_signed") -> Optional[float]:
        """Mean measured value minus the expected one, in degrees.

        Positive means the system reads MORE UPRIGHT (less reclined) than truth.
        Defined only for trunk_signed, the angle the reference measures.
        """
        if angle != "trunk_signed":
            return None
        mean = self.mean(angle)
        if mean is None:
            return None
        return mean - expected_trunk_signed(self.recline_deg)


def load_trials(path: Path = TRIALS_PATH) -> List[Trial]:
    if not path.exists():
        return []
    trials = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        known = set(Trial.__dataclass_fields__)
        trials.append(Trial(**{k: v for k, v in data.items() if k in known}))
    return trials


def save_trial(trial: Trial, path: Path = TRIALS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(trial)) + "\n")
    return path


# ---------------------------------------------------------------------------
# synth: is the arithmetic right?
# ---------------------------------------------------------------------------

def run_synth(args) -> int:
    """Feed compute_angles skeletons with an exactly known angle.

    This is the control. It needs no camera and must show essentially zero
    error; anything else is a bug in this project rather than in the sensor.
    """
    from validation_geometry import seated_skeleton

    reclines = [float(r) for r in args.reclines]
    forward_is_image_right = args.camera_position != "right-side"

    print(f"Synthetic control - {len(reclines)} angle(s), exact geometry\n")
    header = f"{'recline':>8} {'expected':>9} {'measured':>9} {'error':>8}"
    print(header)
    print("-" * len(header))

    # The skeleton is built in a left-side view. Read with
    # forward_is_image_right=False it stands in for a right-side camera, which
    # mirrors every SIGNED angle - trunk and neck both. The hip and knee angles
    # are unsigned interior angles and do not mirror. Getting this wrong is what
    # the harness's own control caught the first time it ran.
    mirror = 1.0 if forward_is_image_right else -1.0

    errors = []
    limb_errors = []
    for recline in reclines:
        skeleton = seated_skeleton(
            expected_trunk_signed(recline),
            neck_signed_deg=args.neck,
            hip_angle_deg=args.hip,
            knee_angle_deg=args.knee,
        )
        angles = compute_angles(skeleton, forward_is_image_right)
        expected = mirror * expected_trunk_signed(recline)
        measured = angles["trunk_signed"]
        error = measured - expected
        errors.append(error)
        limb_errors.append(abs(angles["left_hip_angle"] - args.hip))
        limb_errors.append(abs(angles["left_knee_angle"] - args.knee))
        limb_errors.append(abs(angles["neck_signed"] - mirror * args.neck))
        print(f"{recline:8.1f} {expected:9.2f} {measured:9.2f} {error:8.3f}")

    worst = max(abs(e) for e in errors)
    worst_limb = max(limb_errors)
    print(f"\nworst torso error {worst:.4f} deg, worst limb/neck error {worst_limb:.4f} deg")
    tolerance = 0.01
    if worst <= tolerance and worst_limb <= tolerance:
        print(f"PASS - the pixel-to-angle path is exact to within {tolerance} deg.")
        print("Any error in a real trial is therefore the camera and pose model,")
        print("not this project's arithmetic.")
        return 0
    print(f"FAIL - error exceeds {tolerance} deg with no sensor involved.")
    print("Fix this before interpreting any captured trial.")
    return 1


# ---------------------------------------------------------------------------
# capture: record one trial against a declared reference
# ---------------------------------------------------------------------------

def run_capture(args) -> int:
    try:
        import cv2
    except ImportError:
        print("capture needs opencv-python: pip install -r requirements.txt")
        return 1
    from pose_core import PoseExtractor

    if args.reference_source not in REFERENCE_SOURCES:
        print(f"--reference-source must be one of {REFERENCE_SOURCES}")
        return 1

    forward_is_image_right = args.camera_position != "right-side"
    if args.camera_position in ("front", "rear"):
        print("Refusing to capture from a front or rear camera: leaning forward")
        print("barely moves x in that view, so the lean SIGN this harness is")
        print("validating carries almost no signal. Mount the camera to one side.")
        return 1

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Camera {args.camera} not available.")
        return 1

    extractor = PoseExtractor(min_detection_confidence=0.6,
                              min_tracking_confidence=0.6)
    trial = Trial(
        trial_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
        recorded_at=datetime.now().isoformat(),
        recline_deg=float(args.recline),
        reference_source=args.reference_source,
        subject_id=args.subject_id,
        seat_id=args.seat_id,
        camera_position=args.camera_position,
        camera_distance_cm=args.camera_distance_cm,
        notes=args.notes,
    )
    trial.samples = {angle: [] for angle in VALIDATED_ANGLES}

    expected = expected_trunk_signed(trial.recline_deg)
    print(f"Trial {trial.trial_id}: subject {trial.subject_id}, "
          f"recline {trial.recline_deg:.1f} deg from {trial.reference_source}")
    print(f"Expecting trunk_signed near {expected:+.1f}. "
          f"Hold still for {args.duration:.0f}s. 'q' aborts.")

    confidences = []
    start = time.monotonic()
    aborted = False
    try:
        while time.monotonic() - start < args.duration:
            ok, frame = cap.read()
            if not ok:
                continue
            trial.frames_seen += 1
            results, pts, _ = extractor.process_bgr(frame)
            if results is None:
                if not args.no_preview:
                    cv2.putText(frame, "no person detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow("validation capture", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        aborted = True
                        break
                continue

            visibility = extractor.visibility(results)
            angles = compute_angles(pts, forward_is_image_right, visibility)
            for angle in VALIDATED_ANGLES:
                trial.samples[angle].append(
                    float(angles[angle]) if is_measured(angles[angle]) else float("nan"))
            if is_measured(angles["trunk_signed"]):
                trial.frames_measured += 1
            confidence = frame_confidence(visibility, ["trunk_signed"])
            if confidence is not None:
                confidences.append(confidence)

            if not args.no_preview:
                extractor.draw(frame, results)
                remaining = args.duration - (time.monotonic() - start)
                live = angles["trunk_signed"]
                cv2.putText(frame,
                            f"trunk {live:+.1f} (expect {expected:+.1f})"
                            if is_measured(live) else "trunk -- (not measured)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)
                cv2.putText(frame, f"{remaining:.0f}s  frames {trial.frames_measured}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("validation capture", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    aborted = True
                    break
    finally:
        cap.release()
        if not args.no_preview:
            cv2.destroyAllWindows()

    trial.duration_s = round(time.monotonic() - start, 2)
    trial.mean_frame_confidence = (round(statistics.fmean(confidences), 3)
                                   if confidences else None)

    if aborted:
        print("Aborted; trial not saved.")
        return 1
    if trial.frames_measured < args.min_frames:
        print(f"Only {trial.frames_measured} measured frame(s), below "
              f"--min-frames {args.min_frames}. Trial NOT saved.")
        print("The torso landmarks were not visible enough. Fix the framing "
              "and lighting, then retry - a trial this thin would add noise, "
              "not evidence.")
        return 1

    save_trial(trial)
    mean = trial.mean("trunk_signed")
    error = trial.error()
    scatter = (statistics.stdev(trial.measured("trunk_signed"))
               if len(trial.measured("trunk_signed")) > 1 else 0.0)
    print(f"\nmeasured {mean:+.2f} +- {scatter:.2f} deg over "
          f"{trial.frames_measured}/{trial.frames_seen} frames")
    print(f"error vs reference {error:+.2f} deg "
          f"({'reads more upright' if error > 0 else 'reads more reclined'})")
    print(f"saved to {TRIALS_PATH}")
    print("\nWithin-trial scatter is NOT the error bar for a seat comparison - "
          "frames of a person sitting still are highly correlated. Run several "
          "separate trials and let `report` use between-trial spread.")
    return 0


# ---------------------------------------------------------------------------
# report: bias, scatter, linearity, and whether it is good enough
# ---------------------------------------------------------------------------

def _linear_fit(truths: List[float], measured: List[float]):
    """Least-squares slope and intercept of measured against truth.

    Slope matters as much as bias: a slope of 0.8 means the system
    systematically under-reports recline by 20%, which distorts comparisons
    between seats set to DIFFERENT recline angles even if the mean bias is zero.
    """
    if len(set(truths)) < 2:
        return None, None
    slope, intercept = np.polyfit(np.array(truths), np.array(measured), 1)
    return float(slope), float(intercept)


def run_report(args) -> int:
    trials = load_trials(Path(args.trials) if args.trials else TRIALS_PATH)
    if args.reference_source:
        trials = [t for t in trials if t.reference_source == args.reference_source]
    if args.subject_id:
        trials = [t for t in trials if t.subject_id == args.subject_id]

    usable = [t for t in trials if t.error() is not None]
    if not usable:
        print("No usable trials. Record some with:")
        print("  python3 validate_angles.py capture --recline 25 "
              "--reference-source backrest --subject-id S1")
        print("Or check the arithmetic first, with no camera:")
        print("  python3 validate_angles.py synth")
        return 1

    print(f"{len(usable)} trial(s)"
          + (f", filtered to {args.reference_source}" if args.reference_source else "")
          + "\n")

    # --- per trial
    header = (f"{'trial':16} {'subj':6} {'src':9} {'recline':>8} {'expect':>7} "
              f"{'measured':>9} {'error':>7} {'sd':>6} {'frames':>7}")
    print(header)
    print("-" * len(header))
    for t in sorted(usable, key=lambda t: (t.recline_deg, t.trial_id)):
        values = t.measured("trunk_signed")
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"{t.trial_id:16} {t.subject_id:6} {t.reference_source:9} "
              f"{t.recline_deg:8.1f} {expected_trunk_signed(t.recline_deg):7.1f} "
              f"{t.mean('trunk_signed'):9.2f} {t.error():7.2f} {sd:6.2f} "
              f"{t.frames_measured:7}")

    # --- aggregate, one value per trial so correlated frames cannot inflate n
    errors = [t.error() for t in usable]
    bias = statistics.fmean(errors)
    spread = statistics.stdev(errors) if len(errors) > 1 else 0.0
    mae = statistics.fmean(abs(e) for e in errors)
    loa_low, loa_high = bias - Z95 * spread, bias + Z95 * spread

    print(f"\n{'=' * 60}\nAGREEMENT WITH THE REFERENCE (one value per trial)\n{'=' * 60}")
    print(f"  bias (mean signed error)   {bias:+.2f} deg")
    print(f"  between-trial SD           {spread:.2f} deg")
    print(f"  mean absolute error        {mae:.2f} deg")
    if len(errors) > 1:
        print(f"  95% limits of agreement    {loa_low:+.2f} to {loa_high:+.2f} deg")
    else:
        print("  95% limits of agreement    n/a - needs at least 2 trials")

    slope, intercept = _linear_fit([expected_trunk_signed(t.recline_deg) for t in usable],
                                  [t.mean("trunk_signed") for t in usable])
    if slope is None:
        print("  linearity                  n/a - needs >=2 distinct recline angles")
    else:
        print(f"  measured = {slope:.3f} x truth + {intercept:+.2f}")
        if abs(slope - 1.0) > 0.1:
            direction = "under" if slope < 1.0 else "over"
            print(f"    NOTE slope is {abs(1 - slope) * 100:.0f}% off unity: the system "
                  f"{direction}-reports recline as it grows. Comparing seats set to "
                  f"different recline angles will be distorted even where the bias is 0.")

    # --- by group
    for label, key, footnote in (
        ("reference source", lambda t: t.reference_source,
         "A gap between 'backrest' and 'sternum' is subject conformity, "
         "not sensor error."),
        ("subject", lambda t: t.subject_id,
         "A subject whose bias stands apart is usually sitting rotated toward "
         "or away from the camera, which compresses the measured angle."),
        ("recline angle", lambda t: f"{t.recline_deg:.0f} deg", None),
    ):
        groups: Dict[str, List[float]] = {}
        for t in usable:
            groups.setdefault(key(t), []).append(t.error())
        if len(groups) > 1:
            print(f"\n  by {label}:")
            for name, errs in sorted(groups.items()):
                sd = statistics.stdev(errs) if len(errs) > 1 else 0.0
                print(f"    {name:12} n={len(errs):3} bias {statistics.fmean(errs):+6.2f} "
                      f"sd {sd:5.2f}")
            if footnote:
                print(f"    {footnote}")

    # --- is it good enough?
    trunk_range = DRIVING.ideal["trunk_signed"]
    range_width = trunk_range.max - trunk_range.min
    print(f"\n{'=' * 60}\nIS IT ACCURATE ENOUGH?\n{'=' * 60}")
    print(f"  DRIVING ideal trunk range  {trunk_range.min:.0f} to {trunk_range.max:.0f} "
          f"deg ({range_width:.0f} deg wide)")
    if len(errors) > 1:
        loa_width = loa_high - loa_low
        print(f"  95% agreement width        {loa_width:.1f} deg "
              f"({loa_width / range_width * 100:.0f}% of the range)")
        if loa_width >= range_width:
            print("  VERDICT  The sensor cannot resolve the model. Agreement is as wide")
            print("           as the whole ideal range, so a frame scored 'Excellent'")
            print("           could truly be outside it. Do not compare seats yet.")
        elif loa_width >= range_width / 2:
            print("  VERDICT  Marginal. Agreement spans over half the ideal range, so")
            print("           only large seat differences will be real. Reduce scatter")
            print("           (fix camera placement, lighting, subject rotation) before")
            print("           drawing conclusions.")
        else:
            print("  VERDICT  Usable. Agreement is comfortably inside the ideal range.")
        for n in (3, 5, 10, 20):
            mdd = MDD_FACTOR * spread / math.sqrt(n)
            print(f"  with {n:2} trials per seat, smallest resolvable difference "
                  f"{mdd:.1f} deg")
        print("\n  Those assume INDEPENDENT trials - separate sittings, not frames or")
        print("  repeats without getting out of the seat. They are a planning guide,")
        print("  not a substitute for a test on the real comparison.")
    else:
        print("  VERDICT  Only one trial. Record at least 5 across 3+ recline angles")
        print("           before reading anything into the bias.")

    if abs(bias) > 2.0:
        print(f"\n  The {bias:+.1f} deg bias is systematic and correctable: subtract it")
        print("  from measurements, or re-site the camera. It matters less than the")
        print("  scatter for COMPARING seats, since it cancels - but it matters a lot")
        print("  for judging one seat against the cited ideal range.")
    return 0


def run_list(args) -> int:
    trials = load_trials()
    if not trials:
        print(f"No trials recorded in {TRIALS_PATH}.")
        return 0
    print(f"{len(trials)} trial(s) in {TRIALS_PATH}:")
    for t in trials:
        error = t.error()
        tail = (f"error={error:+6.2f}" if error is not None
                else "(no measured frames)")
        print(f"  {t.trial_id}  subj={t.subject_id:6} {t.reference_source:9} "
              f"recline={t.recline_deg:5.1f}  frames={t.frames_measured:5}  {tail}")
    return 0


def run_protocol(args) -> int:
    print(PROTOCOL)
    return 0


PROTOCOL = """\
VALIDATION PROTOCOL
===================

Goal: a bias-and-scatter table for the trunk angle, so you know whether the
seat differences you measure later are larger than the sensor's own error.

Before touching a camera
------------------------
  python3 validate_angles.py synth
Must PASS. It proves the pixel-to-angle arithmetic is exact, so any error you
measure afterwards belongs to the camera and pose model.

Setting up
----------
1. Camera to ONE SIDE of the subject, lens roughly at chest height, optical
   axis perpendicular to the direction the subject faces. Not front, not rear:
   leaning forward barely moves x in those views.
2. Whole torso and both hips in frame. Include knees if you want the limb
   angles validated too.
3. Fix the camera. Tape the tripod. Moving it between trials turns camera
   placement into an uncontrolled variable.
4. Record the distance in cm and pass it as --camera-distance-cm. Perspective
   error grows as the subject gets closer.

Getting a reference angle
-------------------------
Best:  inclinometer taped to the breastbone -> --reference-source sternum
       This measures the torso, which is what the system claims to measure.
Easy:  phone inclinometer app laid flat on the backrest -> backrest
       Expect a few degrees of extra apparent error from the subject not
       conforming perfectly to the seat. That term is real, and `report`
       separates it by showing bias per source.

Record the angle as DEGREES OF RECLINE FROM VERTICAL, POSITIVE.
Upright = 0. A typical car backrest = 20 to 25.

The sweep
---------
At least 5 recline settings spanning the range you care about, e.g.
0, 10, 20, 25, 35. For each:

  python3 validate_angles.py capture \\
      --recline 25 --reference-source sternum \\
      --subject-id S1 --camera-distance-cm 150

Then repeat the WHOLE sweep at least 3 times, standing up and sitting back down
between repeats. Getting out of the seat is the point: it captures the
variation a real seat test will face. Repeats without moving measure only
camera noise and will flatter the result.

Then: more than one subject. Three builds minimum (small, median, large). Seat
ergonomics varies with body size, and a sweep from one body characterises one
body.

Reading the result
------------------
  python3 validate_angles.py report

  bias     a systematic offset. Correctable, and it cancels when comparing two
           seats - but it shifts a single seat against the cited ideal range.
  SD       scatter between trials. This is what limits a seat comparison.
  slope    should be near 1.0. Away from it means recline is compressed or
           stretched as it grows, which distorts comparisons across different
           recline settings even when bias is zero.
  verdict  compares the 95% agreement width against the 25-degree DRIVING ideal
           range, and states the smallest seat difference you can resolve.

If the verdict is not "Usable", fixing the setup is cheaper than collecting
more data against a sensor that cannot see the effect.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate this project's angle measurement against a known reference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Start here:
  python3 validate_angles.py protocol          # how to run a valid study
  python3 validate_angles.py synth             # check the arithmetic, no camera
  python3 validate_angles.py capture --recline 25 --reference-source backrest \\
      --subject-id S1
  python3 validate_angles.py report            # bias, scatter, and the verdict
""")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("synth", help="Check the pixel-to-angle path against exact geometry")
    p.add_argument("--reclines", nargs="+", type=float,
                   default=[0, 5, 10, 15, 20, 25, 30, 35, 45],
                   help="Recline angles to test, degrees from vertical")
    p.add_argument("--neck", type=float, default=8.0)
    p.add_argument("--hip", type=float, default=105.0)
    p.add_argument("--knee", type=float, default=115.0)
    p.add_argument("--camera-position", default="left-side",
                   choices=["left-side", "right-side"])
    p.set_defaults(func=run_synth)

    p = sub.add_parser("capture", help="Record one trial against a declared reference angle")
    p.add_argument("--recline", type=float, required=True,
                   help="Reference angle: DEGREES OF RECLINE FROM VERTICAL, positive")
    p.add_argument("--reference-source", required=True, choices=REFERENCE_SOURCES[:3],
                   help="How the reference angle was obtained")
    p.add_argument("--subject-id", required=True, help="Who is sitting")
    p.add_argument("--seat-id", help="Seat under test, if this doubles as a seat trial")
    p.add_argument("--camera-position", default="left-side",
                   choices=["left-side", "right-side", "front", "rear"])
    p.add_argument("--camera-distance-cm", type=float,
                   help="Lens to subject, cm. Perspective error grows as this shrinks.")
    p.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    p.add_argument("--duration", type=float, default=15.0, help="Seconds to record")
    p.add_argument("--min-frames", type=int, default=30,
                   help="Reject the trial below this many measured frames")
    p.add_argument("--no-preview", action="store_true", help="No window; for scripting")
    p.add_argument("--notes", help="Anything unusual about this trial")
    p.set_defaults(func=run_capture)

    p = sub.add_parser("report", help="Bias, scatter, linearity, and the resolution verdict")
    p.add_argument("--trials", help=f"Trials file (default {TRIALS_PATH})")
    p.add_argument("--reference-source", choices=REFERENCE_SOURCES)
    p.add_argument("--subject-id")
    p.set_defaults(func=run_report)

    p = sub.add_parser("list", help="List recorded trials")
    p.set_defaults(func=run_list)

    p = sub.add_parser("protocol", help="Print the measurement protocol")
    p.set_defaults(func=run_protocol)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
