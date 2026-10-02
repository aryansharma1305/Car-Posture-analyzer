"""Desk-posture quality scoring.

The ideal ranges and the per-degree penalty slope come from a
driver_model.PostureReference (DESK by default). This module used to own its own
copy of them, which is why the same angle could score 100 here and fail a
threshold rule elsewhere.

The CATEGORY weights below are this module's own: they describe how it groups
angles for reporting, not what good posture is.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from driver_model import DESK, PostureReference
from geometry_utils import is_measured

@dataclass
class PostureScore:
    """Posture quality score with breakdown and recommendations."""
    overall_score: float  # 0-100, higher is better
    category_scores: Dict[str, float]  # individual category scores
    risk_level: str  # "Low", "Medium", "High", or "Unknown"
    recommendations: List[str]  # specific improvement suggestions
    duration_warnings: List[str]  # warnings about time spent in poor posture
    # Categories with no measurable angle this frame. Their entry in
    # category_scores is None, not 0 and not 100.
    unmeasured_categories: List[str] = field(default_factory=list)
    # Share of the category weight that was actually measured, 0..1. A score
    # built from the torso alone is not comparable with a whole-body one, and
    # nothing in this object said so before.
    measured_fraction: float = 1.0

class PostureQualityScorer:
    """Evaluates posture quality against a driver_model.PostureReference."""

    # How the per-angle scores are grouped and weighted into one number.
    CATEGORY_WEIGHTS = {
        "upper_body": 0.35,      # Most important
        "lower_body": 0.30,      # Important for seated work
        "shoulders": 0.20,       # Moderate importance
        "symmetry": 0.15         # Less critical but good indicator
    }

    # Points of symmetry score lost per degree of left/right mismatch, summed
    # over hips, knees and shoulders.
    SYMMETRY_PENALTY_PER_DEG = 2.0

    # Degrees past the ideal maximum at which advice escalates from "sit up" to
    # "change your setup". 10 reproduces the old second-tier thresholds
    # (neck 15 -> 25, trunk 10 -> 20).
    SEVERE_MARGIN_DEG = 10.0

    # Symmetry score below which left/right imbalance is called out.
    SYMMETRY_WARNING_SCORE = 70.0

    # The angles this scorer groups into categories. A reference that does not
    # define all of them cannot be scored here.
    REQUIRED_ANGLES = (
        "neck_from_vertical", "trunk_from_vertical",
        "left_hip_angle", "right_hip_angle",
        "left_knee_angle", "right_knee_angle",
        "left_shoulder_elev", "right_shoulder_elev",
    )

    def __init__(self, reference: PostureReference = DESK):
        missing = [a for a in self.REQUIRED_ANGLES if a not in reference.ideal]
        if missing:
            raise ValueError(
                f"PostureQualityScorer needs a reference defining {missing}; "
                f"{reference.name!r} does not. It scores the magnitude-based "
                "desk model - for the automotive model use "
                "driver_comfort_analyzer.DriverComfortAnalyzer, which reads the "
                "signed torso angles."
            )
        self.reference = reference

        # Risk thresholds
        self.risk_thresholds = {
            "low": 80,      # Score >= 80: Low risk
            "medium": 60,   # Score >= 60: Medium risk
            "high": 0       # Score < 60: High risk
        }

        # Duration warnings (minutes)
        self.duration_warnings = {
            "neck_forward": 15,      # Forward head posture
            "trunk_slouch": 20,      # Slouched trunk
            "raised_shoulders": 30,  # Elevated shoulders
            "crossed_legs": 45       # Crossed legs
        }

    @property
    def ideal_ranges(self) -> Dict[str, tuple]:
        """The reference's ranges as (min, max) pairs.

        Read-only compatibility view for callers written against the
        pre-driver_model attribute. New code should read self.reference.ideal.
        """
        return {angle: (rng.min, rng.max)
                for angle, rng in self.reference.ideal.items()}

    def calculate_angle_score(self, angle_name: str, angle_value: float) -> float:
        """Calculate score for a single angle (0-100).

        100 inside the reference's ideal range, falling linearly to 0 at
        reference.score_zero_at_deg degrees outside it. For DESK that slope is
        40 degrees, which is the same line as the old `deviation * 2.5`.
        """
        return self.reference.angle_score(angle_name, angle_value)

    # Which angles make up each reported category.
    CATEGORY_ANGLES = {
        "upper_body": ("neck_from_vertical", "trunk_from_vertical"),
        "lower_body": ("left_hip_angle", "right_hip_angle",
                       "left_knee_angle", "right_knee_angle"),
        "shoulders": ("left_shoulder_elev", "right_shoulder_elev"),
    }

    # Left/right pairs whose mismatch feeds the symmetry score.
    SYMMETRY_PAIRS = (
        ("left_hip_angle", "right_hip_angle"),
        ("left_knee_angle", "right_knee_angle"),
        ("left_shoulder_elev", "right_shoulder_elev"),
    )

    def calculate_category_scores(self, angles: Dict[str, float]) -> Dict[str, Optional[float]]:
        """Score each posture category, or None where nothing was measurable.

        Every angle used to be read with angles["..."], so an unmeasured joint
        made the category mean NaN and the overall score NaN - a number that
        prints, compares as False against every threshold, and lands in the CSV.
        Unmeasured angles are now dropped from their category's mean, and a
        category with nothing left scores None.
        """
        scores: Dict[str, Optional[float]] = {}

        for category, members in self.CATEGORY_ANGLES.items():
            measured = [self.calculate_angle_score(angle, angles[angle])
                        for angle in members if is_measured(angles.get(angle))]
            scores[category] = sum(measured) / len(measured) if measured else None

        # Symmetry needs BOTH sides of a pair to mean anything.
        diffs = [abs(angles[left] - angles[right])
                 for left, right in self.SYMMETRY_PAIRS
                 if is_measured(angles.get(left)) and is_measured(angles.get(right))]
        if diffs:
            # Scaled by the number of pairs available, so a frame with one
            # visible pair is not flattered relative to one with three.
            penalty = min(100.0, sum(diffs) * self.SYMMETRY_PENALTY_PER_DEG
                          * len(self.SYMMETRY_PAIRS) / len(diffs))
            scores["symmetry"] = max(0.0, 100.0 - penalty)
        else:
            scores["symmetry"] = None

        return scores

    def generate_recommendations(self, angles: Dict[str, float], category_scores: Dict[str, float]) -> List[str]:
        """Generate specific improvement recommendations.

        Every trigger is "outside the reference's ideal range", with the second,
        stronger line added once the angle is SEVERE_MARGIN_DEG past it. Before
        this the bounds were written out again here (15/25, 10/20, 85/105,
        85/105, 25) and had to be kept in step with self.ideal_ranges by hand.
        """
        recommendations = []
        ideal = self.reference.ideal

        def over(angle: str) -> Optional[float]:
            """Degrees above the ideal maximum, or None if not over it.

            Also None when the angle was not measured, so advice is never given
            about a joint the camera could not see.
            """
            if angle not in ideal or not is_measured(angles.get(angle)):
                return None
            excess = angles[angle] - ideal[angle].max
            return excess if excess > 0 else None

        # Neck recommendations
        neck_over = over("neck_from_vertical")
        if neck_over is not None:
            recommendations.append("Bring your head back to align with your spine")
            if neck_over > self.SEVERE_MARGIN_DEG:
                recommendations.append("Consider adjusting your monitor height to reduce forward head posture")

        # Trunk recommendations
        trunk_over = over("trunk_from_vertical")
        if trunk_over is not None:
            recommendations.append("Sit up straight and engage your core muscles")
            if trunk_over > self.SEVERE_MARGIN_DEG:
                recommendations.append("Check your chair backrest and lumbar support")

        # Hip recommendations
        hip_range = ideal.get("left_hip_angle")
        hips = [angles[k] for k in ("left_hip_angle", "right_hip_angle")
                if is_measured(angles.get(k))]
        if hip_range is not None and hips:
            hip_avg = sum(hips) / len(hips)
            if hip_avg < hip_range.min:
                recommendations.append("Move your chair closer to the desk to open hip angle")
            elif hip_avg > hip_range.max:
                recommendations.append("Move your chair back to reduce hip angle")

        # Knee recommendations
        knee_range = ideal.get("left_knee_angle")
        knees = [angles[k] for k in ("left_knee_angle", "right_knee_angle")
                 if is_measured(angles.get(k))]
        if knee_range is not None and knees:
            knee_avg = sum(knees) / len(knees)
            if knee_avg < knee_range.min:
                recommendations.append("Lower your chair to open knee angle")
            elif knee_avg > knee_range.max:
                recommendations.append("Raise your chair to reduce knee angle")

        # Shoulder recommendations
        if over("left_shoulder_elev") is not None or over("right_shoulder_elev") is not None:
            recommendations.append("Relax your shoulders and keep them down")
            recommendations.append("Check your desk height and keyboard position")

        # Symmetry recommendations
        symmetry = category_scores.get("symmetry")
        if symmetry is not None and symmetry < self.SYMMETRY_WARNING_SCORE:
            recommendations.append("Check for leg crossing or uneven weight distribution")
            recommendations.append("Ensure both feet are flat on the floor")

        return recommendations

    def assess_risk_level(self, overall_score: float) -> str:
        """Determine risk level based on overall score."""
        if overall_score >= self.risk_thresholds["low"]:
            return "Low"
        elif overall_score >= self.risk_thresholds["medium"]:
            return "Medium"
        else:
            return "High"
    
    def score_posture(self, angles: Dict[str, float]) -> PostureScore:
        """Calculate comprehensive posture quality score."""
        category_scores = self.calculate_category_scores(angles)

        unmeasured = sorted(c for c, v in category_scores.items() if v is None)
        measured = {c: v for c, v in category_scores.items() if v is not None}

        # Renormalise over the categories that were measured, so a missing
        # category neither scores 0 nor is quietly treated as perfect.
        total_weight = sum(self.CATEGORY_WEIGHTS[c] for c in measured)
        all_weight = sum(self.CATEGORY_WEIGHTS.values())
        measured_fraction = total_weight / all_weight if all_weight else 0.0

        if total_weight > 0:
            overall_score = sum(
                value * self.CATEGORY_WEIGHTS[category]
                for category, value in measured.items()
            ) / total_weight
            # A risk level is a claim about the subject, so it is only stated
            # when the reference's required angles were actually measured.
            risk_level = (self.assess_risk_level(overall_score)
                          if self.reference.is_observation(angles) else "Unknown")
        else:
            overall_score = 0.0
            risk_level = "Unknown"

        recommendations = self.generate_recommendations(angles, category_scores)
        if unmeasured:
            recommendations.insert(0, (
                "Not scored: " + ", ".join(c.replace('_', ' ') for c in unmeasured)
                + " not visible to the camera."))

        # Duration warnings (will be populated by external tracking)
        duration_warnings = []

        return PostureScore(
            overall_score=round(overall_score, 1),
            category_scores={k: (round(v, 1) if v is not None else None)
                             for k, v in category_scores.items()},
            risk_level=risk_level,
            recommendations=recommendations,
            duration_warnings=duration_warnings,
            unmeasured_categories=unmeasured,
            measured_fraction=round(measured_fraction, 3),
        )

# Convenience function for quick scoring
def score_posture(angles: Dict[str, float]) -> PostureScore:
    """Quick function to score posture without creating scorer instance."""
    scorer = PostureQualityScorer()
    return scorer.score_posture(angles)
