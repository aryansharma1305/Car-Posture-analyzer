#!/usr/bin/env python3
"""
Driver Comfort Analyzer for Mahindra & Mahindra
Analyzes sitting posture and driver comfort when driving

The ergonomic numbers this analyzer scores against live in driver_model.py, not
here. Pass a different PostureReference to score the same angles under a
different seating model; see driver_model.DRIVING for the sources.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json

from driver_model import DRIVING, PostureReference
from geometry_utils import is_measured

# Re-exported for callers and tests that referenced this module's constant
# before the model moved. driver_model owns the value.
SCORE_ZERO_AT_DEG = DRIVING.score_zero_at_deg

@dataclass
class DriverComfortScore:
    """Comprehensive driver comfort score with breakdown."""
    overall_comfort: float  # 0-100, higher is better
    posture_quality: float  # 0-100
    ergonomic_risk: float   # 0-100, lower is better
    fatigue_indicator: float # 0-100, lower is better
    comfort_category: str   # "Excellent", "Good", "Fair", "Poor", "Insufficient data"
    recommendations: List[str]
    seat_comparison: Dict[str, float]  # For comparing different seat types
    # Share of the reference's weight actually measured, 0..1. A score built
    # from the torso alone is not comparable with a whole-body one.
    measured_fraction: float = 1.0
    unmeasured_angles: List[str] = None
    # Angles filled in by assumption rather than measurement - currently only a
    # torso lean DIRECTION backfilled from a magnitude-only legacy log.
    inferred_angles: List[str] = None
    # False when measured_fraction fell below driver_model.MIN_MEASURED_FRACTION;
    # such frames are excluded from session summaries.
    is_observation: bool = True

    def __post_init__(self):
        if self.unmeasured_angles is None:
            self.unmeasured_angles = []
        if self.inferred_angles is None:
            self.inferred_angles = []

class DriverComfortAnalyzer:
    """
    Analyzes driver comfort based on research-backed ergonomic principles.
    Focuses on driving-specific posture metrics and comfort indicators.
    """
    
    def __init__(self, reference: PostureReference = DRIVING):
        # The ideal ranges, risk rules and per-angle weights all come from the
        # reference. Nothing ergonomic is defined in this file.
        self.reference = reference

        # Research-based comfort thresholds (from ergonomic studies)
        self.comfort_thresholds = {
            'excellent': {'min': 85, 'color': (0, 255, 0)},
            'good': {'min': 70, 'color': (0, 255, 255)},
            'fair': {'min': 50, 'color': (0, 165, 255)},
            'poor': {'min': 0, 'color': (0, 0, 255)}
        }

        # Fatigue indicators (from driver behavior studies)
        self.fatigue_indicators = {
            'posture_drift': {'threshold': 0.3, 'weight': 0.4},
            'fidgeting': {'threshold': 0.25, 'weight': 0.3},
            'slouching_tendency': {'threshold': 0.2, 'weight': 0.3}
        }

    @property
    def ideal_driving_angles(self) -> Dict[str, Dict[str, float]]:
        """The reference's ranges in this class's old dict shape.

        Read-only compatibility view for callers written against the
        pre-driver_model attribute. New code should read self.reference.ideal.
        """
        return {
            angle: {
                'min': rng.min,
                'max': rng.max,
                'weight': self.reference.weights.get(angle, 0.0),
            }
            for angle, rng in self.reference.ideal.items()
        }

    @staticmethod
    def _with_signed_lean(angles: Dict[str, float]):
        """Backfill 'trunk_signed'/'neck_signed' for pre-signed-angle inputs.

        Replayed historical logs and older callers only carry the
        magnitude-only '*_from_vertical' keys. Their lean direction is
        genuinely unrecoverable, so it is assumed FORWARD (the positive
        sense) - which reproduces the pre-fix scoring for that data rather
        than silently dropping the torso terms from the weighted average.

        Returns (angles, inferred_keys). The second value is what stops an
        assumption from passing as a measurement: a backfilled trunk_signed must
        not satisfy DRIVING's required-angle gate, or every magnitude-only
        legacy session would come back marked as a clean observation of a lean
        direction nobody ever recorded.
        """
        inferred = []
        for signed_key, magnitude_key in (('trunk_signed', 'trunk_from_vertical'),
                                          ('neck_signed', 'neck_from_vertical')):
            if not is_measured(angles.get(signed_key)) and is_measured(angles.get(magnitude_key)):
                angles = dict(angles)
                angles[signed_key] = abs(angles[magnitude_key])
                inferred.append(signed_key)
        return angles, inferred

    def calculate_comfort_score(self, angles: Dict[str, float], 
                              posture_history: List[Dict] = None,
                              session_duration: float = 0) -> DriverComfortScore:
        """
        Calculate comprehensive driver comfort score.
        
        Args:
            angles: Current posture angles
            posture_history: Historical posture data for fatigue analysis
            session_duration: Duration of driving session in seconds
        
        Returns:
            DriverComfortScore object with detailed breakdown
        """
        
        angles, inferred = self._with_signed_lean(angles)

        # How much of the model this frame can actually support. Everything
        # below is computed over the measured angles only; without this the
        # output of a frame that saw a torso and guessed four limbs was
        # indistinguishable from a full-body reading.
        measured_fraction = self.reference.measured_fraction(angles)
        unmeasured = self.reference.unmeasured(angles)
        # An inferred required angle does not count as measured.
        required_inferred = sorted(set(inferred) & set(self.reference.required_angles))
        is_observation = (self.reference.is_observation(angles)
                          and not required_inferred)

        # 1. Posture Quality Score (0-100) - weighted mean of the
        #    reference's per-angle scores over whatever angles are present.
        weighted_posture_score, posture_scores = self.reference.weighted_score(angles)

        # 2. Ergonomic Risk Score (0-100, lower is better) - the reference's
        #    risk rules. Previously this block hardcoded its own bounds, which
        #    drifted out of step with the ideal ranges above it: the forward
        #    lean flag sat at +20 while the ideal range ended at +15.
        risk_factors = self.reference.risks(angles)
        ergonomic_risk = self.reference.risk_score(angles)

        # 3. Fatigue Indicator (0-100, lower is better)
        fatigue_score = 0
        # Drift needs MEASURED samples. np.std over a window containing one NaN
        # returns NaN, which then propagated into overall_comfort as NaN.
        recent = (posture_history or [])[-10:]
        trunk_samples = [a['trunk_signed'] for a in recent
                         if is_measured(a.get('trunk_signed'))]
        neck_samples = [a['neck_signed'] for a in recent
                        if is_measured(a.get('neck_signed'))]
        fatigue_known = (bool(posture_history) and len(posture_history) > 10
                         and len(trunk_samples) >= 2 and len(neck_samples) >= 2)
        if fatigue_known:
            # Signed values, so a driver oscillating between hunched and
            # reclined registers as drift instead of cancelling out.
            trunk_drift = np.std(trunk_samples)
            neck_drift = np.std(neck_samples)

            # Calculate fatigue based on posture instability
            fatigue_score = min(100, (trunk_drift + neck_drift) * 10)
        
        # 4. Overall Comfort Score
        # Weighted combination of posture quality, risk, and fatigue.
        # When there is no history the fatigue term is UNKNOWN, not zero, so it
        # is dropped and the remaining weights renormalised. Carrying it at
        # fatigue_score = 0 handed every single-frame score a free 20 points,
        # which is why a visibly hunched driver could still read "Good".
        # The posture and risk terms are only meaningful over angles that were
        # measured, so with nothing measured both are dropped and the result is
        # NaN - not 37.5, which is what a zero posture score combined with a
        # zero risk score used to produce for a blank frame.
        terms = []
        if measured_fraction > 0.0:
            terms.append((weighted_posture_score, 0.5))
            terms.append((100 - ergonomic_risk, 0.3))
        if fatigue_known:
            terms.append((100 - fatigue_score, 0.2))
        term_weight = sum(weight for _, weight in terms)
        overall_comfort = (
            sum(value * weight for value, weight in terms) / term_weight
            if term_weight > 0 else float("nan")
        )
        
        # 5. Comfort Category
        comfort_category = "Poor"
        for category, threshold in self.comfort_thresholds.items():
            if is_measured(overall_comfort) and overall_comfort >= threshold['min']:
                comfort_category = category.title()
                break
        if not is_observation:
            # The number is still returned so a UI can show it greyed out, but
            # the category must not claim "Excellent" on the strength of one
            # visible joint.
            comfort_category = "Insufficient data"
        
        # 6. Generate Recommendations
        recommendations = self._generate_recommendations(
            angles, posture_scores, risk_factors, fatigue_score, session_duration
        )
        if required_inferred:
            recommendations.insert(0, (
                "Lean direction was assumed, not measured ("
                + ", ".join(required_inferred)
                + "). Record with --camera-position left-side or right-side."))
        if unmeasured:
            recommendations.insert(0, (
                "Not measured this frame: " + ", ".join(unmeasured)
                + ". Reposition the camera to include them."))
        del recommendations[5:]
        
        # 7. Seat Comparison Data (placeholder for different seat types)
        seat_comparison = {
            'current_seat': overall_comfort,
            'standard_seat': 65,  # Baseline comparison
            'premium_seat': 85,   # Target improvement
        }
        
        return DriverComfortScore(
            overall_comfort=round(overall_comfort, 1),
            posture_quality=round(weighted_posture_score, 1),
            ergonomic_risk=round(ergonomic_risk, 1),
            fatigue_indicator=round(fatigue_score, 1),
            comfort_category=comfort_category,
            recommendations=recommendations,
            seat_comparison=seat_comparison,
            measured_fraction=round(measured_fraction, 3),
            unmeasured_angles=unmeasured,
            inferred_angles=inferred,
            is_observation=is_observation,
        )
    
    def _generate_recommendations(self, angles: Dict[str, float], 
                                posture_scores: Dict[str, float],
                                risk_factors: List[Tuple[str, int]],
                                fatigue_score: float,
                                session_duration: float) -> List[str]:
        """Generate specific recommendations for driver comfort improvement.

        The trigger for every posture recommendation is "outside the
        reference's ideal range". This block used to carry its own fourth set
        of bounds (trunk 15/-5, hip 85, knee 100) that did not match either the
        ideal ranges or the risk rules, so a driver could be scored imperfect
        and told nothing, or told to fix an angle the model called ideal.
        """
        
        recommendations = []
        ideal = self.reference.ideal

        # Torso. Direction matters: too far forward means the driver has come
        # off the backrest, too far back means they have lost reach and belt fit.
        if 'trunk_signed' in ideal and 'trunk_signed' in angles:
            trunk_angle = angles['trunk_signed']
            if trunk_angle > ideal['trunk_signed'].max:
                recommendations.append("Adjust seat backrest to reduce forward lean")
            elif trunk_angle < ideal['trunk_signed'].min:
                recommendations.append("Adjust seat backrest to reduce backward lean")

        if 'neck_signed' in ideal and 'neck_signed' in angles:
            if not ideal['neck_signed'].contains(angles['neck_signed']):
                recommendations.append("Adjust headrest height and seat position for better neck alignment")

        # Hip and knee recommendations
        for side in ['left', 'right']:
            hip_key = f'{side}_hip_angle'
            knee_key = f'{side}_knee_angle'

            if hip_key in ideal and hip_key in angles and angles[hip_key] < ideal[hip_key].min:
                recommendations.append(f"Adjust {side} side seat height and angle for better hip support")

            if knee_key in ideal and knee_key in angles and angles[knee_key] < ideal[knee_key].min:
                recommendations.append(f"Adjust {side} side seat position for better knee angle")
        
        # Fatigue-based recommendations
        if fatigue_score > 50:
            recommendations.append("Consider taking a break - posture instability detected")
        
        # Duration-based recommendations
        if session_duration > 3600:  # 1 hour
            recommendations.append("Long driving session - consider seat adjustments or breaks")
        
        # Reassurance, only when there is nothing to correct. Appending it
        # whenever fewer than three items had accumulated produced lists that
        # said "reduce backward lean" and "good driving position detected" in
        # the same breath.
        if not recommendations:
            recommendations.append("Maintain current posture - good driving position detected")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def analyze_seat_comparison(self, seat_data: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """
        Compare comfort levels between different seat types.
        
        Args:
            seat_data: Dictionary with seat types as keys and posture data as values
        
        Returns:
            Comparison analysis for each seat type
        """
        comparison_results = {}
        
        for seat_type, posture_data in seat_data.items():
            if not posture_data:
                continue
                
            # Calculate average comfort scores for this seat type
            # Only frames the model could actually observe. Averaging in a
            # frame scored from two visible joints would make a seat look
            # better or worse than it is for reasons of camera framing.
            comfort_scores = []
            excluded = 0
            for data_point in posture_data:
                if 'angles' in data_point:
                    score = self.calculate_comfort_score(data_point['angles'])
                    if score.is_observation:
                        comfort_scores.append(score.overall_comfort)
                    else:
                        excluded += 1

            if comfort_scores:
                comparison_results[seat_type] = {
                    'average_comfort': np.mean(comfort_scores),
                    'comfort_std': np.std(comfort_scores),
                    'min_comfort': np.min(comfort_scores),
                    'max_comfort': np.max(comfort_scores),
                    'sample_size': len(comfort_scores),
                    'excluded_samples': excluded,
                }
            elif excluded:
                comparison_results[seat_type] = {
                    'error': 'no frame met the measurement threshold',
                    'sample_size': 0,
                    'excluded_samples': excluded,
                }
        
        return comparison_results
    
    def generate_comfort_report(self, session_data: List[Dict], 
                              session_info: Dict) -> Dict:
        """
        Generate comprehensive comfort report for a driving session.
        
        Args:
            session_data: List of posture data points
            session_info: Session metadata (duration, driver, seat_type, etc.)
        
        Returns:
            Complete comfort analysis report
        """
        
        if not session_data:
            return {"error": "No session data provided"}
        
        # Calculate comfort scores for each data point. Frames the model could
        # not observe are counted and set aside, not averaged in: a session
        # summary built partly from guessed limbs reads as a measurement.
        comfort_scores = []
        excluded_scores = []
        angles_history = []

        for data_point in session_data:
            if 'angles' in data_point:
                angles_history.append(data_point['angles'])
                score = self.calculate_comfort_score(
                    data_point['angles'],
                    angles_history,
                    session_info.get('duration', 0)
                )
                if score.is_observation:
                    comfort_scores.append(score)
                else:
                    excluded_scores.append(score)

        if not comfort_scores:
            return {
                "error": "No frame met the measurement threshold",
                "session_info": session_info,
                "data_points": 0,
                "excluded_data_points": len(excluded_scores),
                "measurement_coverage": {
                    'usable_fraction': 0.0,
                    'note': (
                        "Every frame fell below "
                        "driver_model.MIN_MEASURED_FRACTION or was missing a "
                        "required angle. Check camera framing."
                    ),
                },
            }
        
        # Calculate session statistics
        overall_scores = [score.overall_comfort for score in comfort_scores]
        posture_scores = [score.posture_quality for score in comfort_scores]
        risk_scores = [score.ergonomic_risk for score in comfort_scores]
        fatigue_scores = [score.fatigue_indicator for score in comfort_scores]
        
        # Time-based analysis
        session_duration = session_info.get('duration', 0)
        comfort_trend = self._analyze_comfort_trend(comfort_scores, session_duration)
        
        # Generate recommendations
        final_score = comfort_scores[-1]  # Use last score for final recommendations
        
        report = {
            'session_info': session_info,
            'summary': {
                'total_duration_minutes': round(session_duration / 60, 1),
                'average_comfort': round(np.mean(overall_scores), 1),
                'comfort_std': round(np.std(overall_scores), 1),
                'min_comfort': round(np.min(overall_scores), 1),
                'max_comfort': round(np.max(overall_scores), 1),
                'final_comfort_category': final_score.comfort_category
            },
            'detailed_scores': {
                'posture_quality': {
                    'average': round(np.mean(posture_scores), 1),
                    'trend': 'stable' if np.std(posture_scores) < 10 else 'variable'
                },
                'ergonomic_risk': {
                    'average': round(np.mean(risk_scores), 1),
                    'trend': 'decreasing' if risk_scores[-1] < risk_scores[0] else 'stable'
                },
                'fatigue_indicator': {
                    'average': round(np.mean(fatigue_scores), 1),
                    'trend': 'increasing' if fatigue_scores[-1] > fatigue_scores[0] else 'stable'
                }
            },
            'comfort_trend': comfort_trend,
            'recommendations': final_score.recommendations,
            'seat_analysis': final_score.seat_comparison,
            'data_points': len(comfort_scores),
            'excluded_data_points': len(excluded_scores),
            'measurement_coverage': {
                'usable_fraction': round(
                    len(comfort_scores)
                    / max(1, len(comfort_scores) + len(excluded_scores)), 3),
                'mean_measured_fraction': round(
                    float(np.mean([s.measured_fraction for s in comfort_scores])), 3),
                'angles_most_often_missing': _most_common_missing(
                    comfort_scores + excluded_scores),
            },
        }
        
        return report
    
    def _analyze_comfort_trend(self, comfort_scores: List[DriverComfortScore], 
                              duration: float) -> Dict:
        """Analyze how comfort changes over the driving session."""
        
        if len(comfort_scores) < 5:
            return {"trend": "insufficient_data", "description": "Need more data points"}
        
        # Split session into quarters for trend analysis
        quarter_size = len(comfort_scores) // 4
        quarters = []
        
        for i in range(4):
            start_idx = i * quarter_size
            end_idx = start_idx + quarter_size if i < 3 else len(comfort_scores)
            quarter_scores = [score.overall_comfort for score in comfort_scores[start_idx:end_idx]]
            quarters.append(np.mean(quarter_scores))
        
        # Determine trend
        if quarters[3] > quarters[0] + 5:
            trend = "improving"
            description = "Comfort improved throughout the session"
        elif quarters[3] < quarters[0] - 5:
            trend = "declining"
            description = "Comfort decreased during the session"
        else:
            trend = "stable"
            description = "Comfort remained relatively stable"
        
        return {
            "trend": trend,
            "description": description,
            "quarterly_scores": [round(q, 1) for q in quarters]
        }

def _most_common_missing(scores: List[DriverComfortScore]) -> Dict[str, int]:
    """How many frames each angle was missing from, worst first.

    This is the number that tells a researcher whether a session is worth
    keeping or whether the camera needs moving before the next run.
    """
    counts: Dict[str, int] = {}
    for score in scores:
        for angle in score.unmeasured_angles:
            counts[angle] = counts.get(angle, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# Convenience function for quick comfort scoring
def score_driver_comfort(angles: Dict[str, float], 
                        posture_history: List[Dict] = None,
                        session_duration: float = 0) -> DriverComfortScore:
    """Quick function to get driver comfort score."""
    analyzer = DriverComfortAnalyzer()
    return analyzer.calculate_comfort_score(angles, posture_history, session_duration)
