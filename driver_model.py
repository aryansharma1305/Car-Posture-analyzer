"""The one posture reference model.

Before this module the project held the same ergonomic numbers in six places,
and they disagreed:

  posture_rules.PostureConfig            trunk slouch 18, neck 20, shoulder 35
  posture_fullbody_rules.classify_upper_body
                                         the same three numbers, hardcoded,
                                         ignoring the config that was meant to
                                         own them
  ergonomics_scores.feedback_lines       trunk 18/26, neck 25, shoulder 35,
                                         hip 80..110, knee 80
  ergonomics_scores.compute_rula/reba    buckets at 10/20
  posture_scoring.PostureQualityScorer   neck 0..15, trunk 0..10, hip 85..105,
                                         knee 85..105, shoulder 0..25
  driver_comfort_analyzer                trunk_signed -5..15, neck_signed
                                         -10..20, hip 85..110, knee 100..130
  config_manager.PostureConfig           a seventh copy that nothing read

Two of those are desk-ergonomics numbers and one is an automotive set, which
is the real problem: a driver sitting correctly against a 25-degree backrest
scores as a sloucher under desk thresholds. The numbers were not merely
duplicated, they encoded two different postures.

So this module defines a PostureReference - one named set of ideal ranges,
risk rules and label thresholds - and ships exactly two of them:

  DRIVING   automotive seating. Recline is normal and supported.
  DESK      upright office seating. Recline is not the goal.

Every consumer takes a reference instead of carrying its own constants.

Sign convention
---------------
Torso ranges are expressed on the SIGNED angles, where NEGATIVE MEANS
RECLINED (see geometry_utils.signed_angle_with_vertical). That is the whole
reason DRIVING can say what DESK cannot: a driver reclined 25 degrees and a
driver hunched 25 degrees over the wheel are opposite postures that the
magnitude-only '*_from_vertical' keys report identically.

Sources for the DRIVING ranges
------------------------------
Rebiffe, R. (1969) "Le siege du conducteur: son adaptation aux exigences
    fonctionnelles et anthropometriques", Ergonomics 12(2). Recommended
    driving joint ranges: trunk-thigh (hip) 95-120 deg, knee 95-135 deg.
Grandjean, E. (1980) "Sitting posture of car drivers from the point of view of
    ergonomics". Backrest inclination 20-25 deg from vertical.
SAE J1100 / SAE J826. Dimension A40, "back angle" of the H-point manikin
    relative to vertical; passenger-car design values cluster around 22-25
    deg, and 25 deg is the common seating reference.

Those give a torso line 5-30 degrees behind vertical for a correctly seated
driver, which is what trunk_signed = -30..-5 encodes. Treating 0 degrees as
the driving ideal - which the previous range -5..15 did - penalised the
posture the seat is designed to produce.

Sources for the DESK ranges
---------------------------
These are the pre-existing values, unchanged, and correspond to the usual
display-screen-equipment guidance (neck and trunk near vertical, hips and
knees near 90, shoulders relaxed). They are kept as a second named reference
rather than deleted because posture_scoring, posture_analytics, demo.py and
`main.py score` are all office-posture tools.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from geometry_utils import fold_to_vertical_magnitude, is_measured

# Degrees outside an ideal range at which a joint's score reaches 0, linear in
# between. Per-reference because the two models were calibrated at different
# steepnesses and both are now stated rather than implied:
#   DRIVING 30.0 - was `deviation / SCORE_ZERO_AT_DEG * 100`
#   DESK    40.0 - was `min(100, deviation * 2.5)`, which is the same line
# Calibrating either against measured discomfort is still open work.
DRIVING_SCORE_ZERO_AT_DEG = 30.0
DESK_SCORE_ZERO_AT_DEG = 40.0

# Minimum share of a reference's weight that must actually be measured for a
# frame's score to count as an observation. Below this the frame is reported as
# insufficient and excluded from session summaries, rather than contributing a
# number built from two visible joints out of six.
# 0.5 is a judgement, not a citation: it is the point at which more than half
# the model is guessing. Calibrating it against how often the camera loses the
# legs in a real cabin is open work.
MIN_MEASURED_FRACTION = 0.5

# Width of the "mild" band above a label threshold, used both for the 'mild'
# tag in posture_rules.classify_posture and for mild-vs-severe feedback lines.
MILD_BAND_DEG = 8.0


@dataclass(frozen=True)
class Range:
    """An inclusive ideal range for one angle, in degrees."""
    min: float
    max: float

    def contains(self, value: float) -> bool:
        """True only for a measured value inside the range.

        An unmeasured value is not "outside" the range either - callers that
        need to tell those apart must check geometry_utils.is_measured first.
        """
        return is_measured(value) and self.min <= value <= self.max

    def deviation(self, value: float) -> float:
        """Degrees outside the range; 0.0 when inside."""
        if value < self.min:
            return self.min - value
        if value > self.max:
            return value - self.max
        return 0.0

    def score(self, value: float, zero_at: float) -> float:
        """0..100 for one angle. 100 inside the range, 0 at `zero_at` degrees
        outside it, linear in between.

        The penalty is per DEGREE outside the range and is never scaled by a
        bound's own magnitude: dividing by abs(bound) made sensitivity depend
        on how near zero the bound happened to sit, so a trunk bound of -5 was
        penalised ~20x steeper than a hip bound of 85.
        """
        if not is_measured(value):
            # NaN in, NaN out. Returning 100 would score a joint the camera
            # never saw as ideal, and 0 would invent a fault.
            return float("nan")
        deviation = self.deviation(value)
        if deviation == 0.0:
            return 100.0
        return max(0.0, 100.0 - deviation / zero_at * 100.0)


@dataclass(frozen=True)
class RiskRule:
    """One ergonomic risk flag.

    angle   key in the angles dict, as produced by posture_angles.compute_angles
    below   fires when the value is under this, if set
    above   fires when the value is over this, if set
    use_abs compare the magnitude, for "strain in either direction"
    points  contribution to the 0..100 risk total
    """
    angle: str
    label: str
    points: int
    below: Optional[float] = None
    above: Optional[float] = None
    use_abs: bool = False

    def fires(self, angles: Dict[str, float]) -> bool:
        if self.angle not in angles:
            return False
        value = angles[self.angle]
        # An unmeasured angle cannot raise a risk flag. Without this gate the
        # comparisons below are False for NaN, which looks the same as "within
        # limits" - a lost landmark would silently clear the flag it should
        # have made unavailable.
        if not is_measured(value):
            return False
        value = float(value)
        if self.use_abs:
            value = abs(value)
        if self.below is not None and value < self.below:
            return True
        if self.above is not None and value > self.above:
            return True
        return False


@dataclass(frozen=True)
class PostureReference:
    """One named posture model: what good looks like, and how it fails."""

    name: str
    ideal: Dict[str, Range]
    weights: Dict[str, float]
    score_zero_at_deg: float
    risk_rules: Tuple[RiskRule, ...]

    # Label thresholds. The *_key fields say WHICH view of the torso angle the
    # threshold applies to: DRIVING judges slouch on the signed forward angle,
    # because recline is not slouch. DESK judges it on the magnitude, where
    # any departure from vertical is a departure.
    slouch_key: str
    slouch_deg: float
    forward_head_key: str
    forward_head_deg: float
    shoulder_elev_deg: float

    # Hip/knee ranges used by the seated-detection heuristic, expressed as
    # target +- tolerance because posture_rules.is_seated reads it that way.
    seated_hip: Range = field(default=Range(70.0, 120.0))
    seated_knee: Range = field(default=Range(70.0, 120.0))

    # Angles without which a frame is not an observation of this model at all,
    # regardless of how much other weight was measured. For DRIVING that is the
    # torso line: the seat model exists to describe how the backrest supports
    # the trunk, so a frame that saw only legs is not a weaker reading of it,
    # it is a reading of something else.
    required_angles: Tuple[str, ...] = ()

    def angle_score(self, angle: str, value: float) -> float:
        """0..100 for one angle under this reference. Unknown angles score 100."""
        if angle not in self.ideal:
            return 100.0
        return self.ideal[angle].score(value, self.score_zero_at_deg)

    def weighted_score(self, angles: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Weighted mean of the per-angle scores over the angles present.

        Returns (0..100, per-angle scores). Angles that are absent OR
        unmeasured (NaN) are dropped from both the numerator and the
        denominator, so a monitor that cannot see legs is not punished for it -
        and neither is a frame where the legs went behind the wheel. The caller
        should read `measured_fraction` to decide whether the result is worth
        acting on.
        """
        scores: Dict[str, float] = {}
        total_weight = 0.0
        for angle in self.ideal:
            if angle in angles and is_measured(angles[angle]):
                scores[angle] = self.angle_score(angle, angles[angle])
                total_weight += self.weights.get(angle, 0.0)
        if total_weight <= 0.0:
            return 0.0, scores
        weighted = sum(
            score * self.weights.get(angle, 0.0) for angle, score in scores.items()
        )
        return weighted / total_weight, scores

    def unmeasured(self, angles: Dict[str, float]) -> List[str]:
        """Angles this reference scores that the frame could not supply."""
        return sorted(
            angle for angle in self.ideal
            if not is_measured(angles.get(angle))
        )

    def missing_required(self, angles: Dict[str, float]) -> List[str]:
        """Required angles this frame did not measure."""
        return [a for a in self.required_angles
                if not is_measured(angles.get(a))]

    def is_observation(self, angles: Dict[str, float],
                       min_fraction: float = None) -> bool:
        """Whether this frame can be scored against the model at all.

        Two conditions, not one: enough total weight measured, AND every
        required angle present.
        """
        if min_fraction is None:
            min_fraction = MIN_MEASURED_FRACTION
        return (not self.missing_required(angles)
                and self.measured_fraction(angles) >= min_fraction)

    def measured_fraction(self, angles: Dict[str, float]) -> float:
        """Share of this reference's TOTAL weight that was actually measured.

        A comfort score built from two of six joints is not comparable with one
        built from all six, and nothing in the output said so before: both came
        back as a number between 0 and 100.
        """
        total = sum(self.weights.get(a, 0.0) for a in self.ideal)
        if total <= 0.0:
            return 0.0
        measured = sum(
            self.weights.get(a, 0.0) for a in self.ideal
            if is_measured(angles.get(a))
        )
        return measured / total

    def risks(self, angles: Dict[str, float]) -> List[Tuple[str, int]]:
        """Every risk rule that fires, as (label, points)."""
        return [(r.label, r.points) for r in self.risk_rules if r.fires(angles)]

    def risk_score(self, angles: Dict[str, float]) -> float:
        """0..100, lower is better."""
        return float(min(100, sum(points for _, points in self.risks(angles))))


# ---------------------------------------------------------------------------
# Shared weights
# ---------------------------------------------------------------------------
# Left and right carry the SAME weight. The previous driving weights gave the
# left knee 0.15 and the right 0.10 with no stated reason, which made the
# score depend on which leg a driver happened to move.
_DRIVING_WEIGHTS = {
    "trunk_signed": 0.25,
    "neck_signed": 0.20,
    "left_hip_angle": 0.15,
    "right_hip_angle": 0.15,
    "left_knee_angle": 0.125,
    "right_knee_angle": 0.125,
}

# Degrees below an ideal lower bound at which a limb counts as strained. One
# margin for every limb, so the risk flags move with the ranges instead of
# being a second, independently-drifting set of numbers.
LIMB_RISK_MARGIN_DEG = 10.0

_DRIVING_IDEAL = {
    # Torso line 5-30 deg behind vertical: Grandjean 20-25, SAE A40 ~22-25.
    "trunk_signed": Range(-30.0, -5.0),
    # Head close to vertical; a little forward is normal road scanning, and a
    # little back is the headrest.
    "neck_signed": Range(-10.0, 15.0),
    # Trunk-thigh and knee: Rebiffe 1969.
    "left_hip_angle": Range(95.0, 120.0),
    "right_hip_angle": Range(95.0, 120.0),
    "left_knee_angle": Range(95.0, 135.0),
    "right_knee_angle": Range(95.0, 135.0),
}

DRIVING = PostureReference(
    name="driving",
    ideal=_DRIVING_IDEAL,
    weights=_DRIVING_WEIGHTS,
    score_zero_at_deg=DRIVING_SCORE_ZERO_AT_DEG,
    risk_rules=(
        # Forward of upright means the driver has come off the backrest, which
        # is the posture the seat exists to prevent. Flagged well before the
        # 30-degree forward angle a desk model would call severe slouch.
        RiskRule("trunk_signed", "Forward trunk lean", 30, above=10.0),
        # Past the ideal recline the driver loses reach to the wheel and pedals
        # and the belt no longer sits on the chest.
        RiskRule("trunk_signed", "Excessive recline", 25, below=-35.0),
        RiskRule("neck_signed", "Neck strain", 20, above=25.0, use_abs=True),
        RiskRule("left_hip_angle", "Left hip strain", 15,
                 below=_DRIVING_IDEAL["left_hip_angle"].min - LIMB_RISK_MARGIN_DEG),
        RiskRule("right_hip_angle", "Right hip strain", 15,
                 below=_DRIVING_IDEAL["right_hip_angle"].min - LIMB_RISK_MARGIN_DEG),
        RiskRule("left_knee_angle", "Left knee strain", 10,
                 below=_DRIVING_IDEAL["left_knee_angle"].min - LIMB_RISK_MARGIN_DEG),
        RiskRule("right_knee_angle", "Right knee strain", 10,
                 below=_DRIVING_IDEAL["right_knee_angle"].min - LIMB_RISK_MARGIN_DEG),
    ),
    # Judged on the SIGNED angle: a reclined driver is not slouching.
    slouch_key="trunk_signed",
    slouch_deg=10.0,
    forward_head_key="neck_signed",
    forward_head_deg=20.0,
    shoulder_elev_deg=35.0,
    seated_hip=Range(85.0, 130.0),
    seated_knee=Range(85.0, 145.0),
    required_angles=("trunk_signed",),
)

_DESK_IDEAL = {
    "neck_from_vertical": Range(0.0, 15.0),
    "trunk_from_vertical": Range(0.0, 10.0),
    "left_hip_angle": Range(85.0, 105.0),
    "right_hip_angle": Range(85.0, 105.0),
    "left_knee_angle": Range(85.0, 105.0),
    "right_knee_angle": Range(85.0, 105.0),
    "left_shoulder_elev": Range(0.0, 25.0),
    "right_shoulder_elev": Range(0.0, 25.0),
}

DESK = PostureReference(
    name="desk",
    ideal=_DESK_IDEAL,
    # posture_scoring groups these into categories and weights the categories,
    # so the per-angle weights here are uniform and only matter when a caller
    # uses weighted_score directly.
    weights={angle: 1.0 for angle in _DESK_IDEAL},
    score_zero_at_deg=DESK_SCORE_ZERO_AT_DEG,
    risk_rules=(
        RiskRule("trunk_from_vertical", "Trunk slouch", 30, above=18.0),
        RiskRule("neck_from_vertical", "Forward head", 20, above=20.0),
        RiskRule("left_shoulder_elev", "Left shoulder elevation", 10, above=35.0),
        RiskRule("right_shoulder_elev", "Right shoulder elevation", 10, above=35.0),
        RiskRule("left_hip_angle", "Left hip strain", 15,
                 below=_DESK_IDEAL["left_hip_angle"].min - LIMB_RISK_MARGIN_DEG),
        RiskRule("right_hip_angle", "Right hip strain", 15,
                 below=_DESK_IDEAL["right_hip_angle"].min - LIMB_RISK_MARGIN_DEG),
    ),
    # Judged on the magnitude: at a desk, any departure from vertical counts.
    slouch_key="trunk_from_vertical",
    slouch_deg=18.0,
    forward_head_key="neck_from_vertical",
    forward_head_deg=20.0,
    shoulder_elev_deg=35.0,
    seated_hip=Range(70.0, 120.0),
    seated_knee=Range(70.0, 120.0),
    required_angles=("trunk_from_vertical", "neck_from_vertical"),
)

REFERENCES = {DRIVING.name: DRIVING, DESK.name: DESK}

# The reference used when a caller does not name one. DESK, because the desk
# tools (posture_scoring, posture_analytics, demo, `main.py score`) are the
# ones that historically called into these rules without any context, and
# changing their behaviour silently would be worse than leaving it stated.
DEFAULT_REFERENCE = DESK


def get_reference(name: Optional[str]) -> PostureReference:
    """Look up a reference by name; None gives DEFAULT_REFERENCE."""
    if name is None:
        return DEFAULT_REFERENCE
    try:
        return REFERENCES[name]
    except KeyError:
        raise ValueError(
            f"Unknown posture reference {name!r}; known: {sorted(REFERENCES)}"
        ) from None


def torso_view(angles: Dict[str, float], key: str) -> Optional[float]:
    """Read a torso angle under the view `key` names, deriving it if possible.

    A monitor on the canonical pipeline emits both views (posture_angles.
    compute_angles), but replayed historical logs carry only the magnitudes.
    Folding a signed value to its magnitude is lossless, so that derivation is
    done here; the reverse is not, so asking for a signed value that was never
    recorded returns None rather than a guess, and the caller skips the rule
    instead of inventing a lean direction.

    An angle present but NaN - the landmark was not trustworthy this frame - is
    also reported as None, for the same reason: the caller must skip the rule,
    not compare against a value whose every comparison is False.
    """
    if key in angles:
        # NaN means "measured nothing", so report it as absent rather than
        # handing a caller a number that fails every comparison silently.
        return float(angles[key]) if is_measured(angles[key]) else None
    SIGNED_SOURCE = {
        "trunk_from_vertical": "trunk_signed",
        "neck_from_vertical": "neck_signed",
    }
    source = SIGNED_SOURCE.get(key)
    if source is not None and is_measured(angles.get(source)):
        return fold_to_vertical_magnitude(angles[source])
    return None
