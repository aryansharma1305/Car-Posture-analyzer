import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List

@dataclass
class PostureScore:
    """Posture quality score with breakdown and recommendations."""
    overall_score: float  # 0-100, higher is better
    category_scores: Dict[str, float]  # individual category scores
    risk_level: str  # "Low", "Medium", "High"
    recommendations: List[str]  # specific improvement suggestions
    duration_warnings: List[str]  # warnings about time spent in poor posture

class PostureQualityScorer:
    """Evaluates posture quality based on ergonomic standards."""
    
    def __init__(self):
        # Ideal angle ranges for good posture (in degrees)
        self.ideal_ranges = {
            "neck_from_vertical": (0, 15),      # Neck should be nearly vertical
            "trunk_from_vertical": (0, 10),     # Trunk should be nearly vertical
            "left_hip_angle": (85, 105),        # Hips should be ~90° when seated
            "right_hip_angle": (85, 105),
            "left_knee_angle": (85, 105),       # Knees should be ~90° when seated
            "right_knee_angle": (85, 105),
            "left_shoulder_elev": (0, 25),      # Shoulders should be relaxed
            "right_shoulder_elev": (0, 25)
        }
        
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
    
    def calculate_angle_score(self, angle_name: str, angle_value: float) -> float:
        """Calculate score for a single angle (0-100)."""
        if angle_name not in self.ideal_ranges:
            return 100.0  # Unknown angle gets perfect score
        
        ideal_min, ideal_max = self.ideal_ranges[angle_name]
        
        # Check if angle is within ideal range
        if ideal_min <= angle_value <= ideal_max:
            return 100.0
        
        # Calculate deviation penalty
        if angle_value < ideal_min:
            deviation = ideal_min - angle_value
        else:
            deviation = angle_value - ideal_max
        
        # Penalty increases exponentially with deviation
        penalty = min(100, deviation * 2.5)  # 2.5 points per degree deviation
        return max(0, 100 - penalty)
    
    def calculate_category_scores(self, angles: Dict[str, float]) -> Dict[str, float]:
        """Calculate scores for different posture categories."""
        scores = {}
        
        # Upper body (neck and trunk)
        neck_score = self.calculate_angle_score("neck_from_vertical", angles["neck_from_vertical"])
        trunk_score = self.calculate_angle_score("trunk_from_vertical", angles["trunk_from_vertical"])
        scores["upper_body"] = (neck_score + trunk_score) / 2
        
        # Lower body (hips and knees)
        hip_l_score = self.calculate_angle_score("left_hip_angle", angles["left_hip_angle"])
        hip_r_score = self.calculate_angle_score("right_hip_angle", angles["right_hip_angle"])
        knee_l_score = self.calculate_angle_score("left_knee_angle", angles["left_knee_angle"])
        knee_r_score = self.calculate_angle_score("right_knee_angle", angles["right_knee_angle"])
        scores["lower_body"] = (hip_l_score + hip_r_score + knee_l_score + knee_r_score) / 4
        
        # Shoulders
        shoulder_l_score = self.calculate_angle_score("left_shoulder_elev", angles["left_shoulder_elev"])
        shoulder_r_score = self.calculate_angle_score("right_shoulder_elev", angles["right_shoulder_elev"])
        scores["shoulders"] = (shoulder_l_score + shoulder_r_score) / 2
        
        # Symmetry (check for imbalances)
        hip_diff = abs(angles["left_hip_angle"] - angles["right_hip_angle"])
        knee_diff = abs(angles["left_knee_angle"] - angles["right_knee_angle"])
        shoulder_diff = abs(angles["left_shoulder_elev"] - angles["right_shoulder_elev"])
        
        symmetry_penalty = min(100, (hip_diff + knee_diff + shoulder_diff) * 2)
        scores["symmetry"] = max(0, 100 - symmetry_penalty)
        
        return scores
    
    def generate_recommendations(self, angles: Dict[str, float], category_scores: Dict[str, float]) -> List[str]:
        """Generate specific improvement recommendations."""
        recommendations = []
        
        # Neck recommendations
        if angles["neck_from_vertical"] > 15:
            recommendations.append("Bring your head back to align with your spine")
            if angles["neck_from_vertical"] > 25:
                recommendations.append("Consider adjusting your monitor height to reduce forward head posture")
        
        # Trunk recommendations
        if angles["trunk_from_vertical"] > 10:
            recommendations.append("Sit up straight and engage your core muscles")
            if angles["trunk_from_vertical"] > 20:
                recommendations.append("Check your chair backrest and lumbar support")
        
        # Hip recommendations
        hip_avg = (angles["left_hip_angle"] + angles["right_hip_angle"]) / 2
        if hip_avg < 85:
            recommendations.append("Move your chair closer to the desk to open hip angle")
        elif hip_avg > 105:
            recommendations.append("Move your chair back to reduce hip angle")
        
        # Knee recommendations
        knee_avg = (angles["left_knee_angle"] + angles["right_knee_angle"]) / 2
        if knee_avg < 85:
            recommendations.append("Lower your chair to open knee angle")
        elif knee_avg > 105:
            recommendations.append("Raise your chair to reduce knee angle")
        
        # Shoulder recommendations
        if angles["left_shoulder_elev"] > 25 or angles["right_shoulder_elev"] > 25:
            recommendations.append("Relax your shoulders and keep them down")
            recommendations.append("Check your desk height and keyboard position")
        
        # Symmetry recommendations
        if category_scores["symmetry"] < 70:
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
        # Calculate individual category scores
        category_scores = self.calculate_category_scores(angles)
        
        # Calculate overall score (weighted average)
        weights = {
            "upper_body": 0.35,      # Most important
            "lower_body": 0.30,      # Important for seated work
            "shoulders": 0.20,       # Moderate importance
            "symmetry": 0.15         # Less critical but good indicator
        }
        
        overall_score = sum(
            category_scores[cat] * weights[cat] 
            for cat in weights.keys()
        )
        
        # Determine risk level
        risk_level = self.assess_risk_level(overall_score)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(angles, category_scores)
        
        # Duration warnings (will be populated by external tracking)
        duration_warnings = []
        
        return PostureScore(
            overall_score=round(overall_score, 1),
            category_scores={k: round(v, 1) for k, v in category_scores.items()},
            risk_level=risk_level,
            recommendations=recommendations,
            duration_warnings=duration_warnings
        )

# Convenience function for quick scoring
def score_posture(angles: Dict[str, float]) -> PostureScore:
    """Quick function to score posture without creating scorer instance."""
    scorer = PostureQualityScorer()
    return scorer.score_posture(angles)
