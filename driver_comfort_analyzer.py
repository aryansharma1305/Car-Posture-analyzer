#!/usr/bin/env python3
"""
Driver Comfort Analyzer for Mahindra & Mahindra
Analyzes sitting posture and driver comfort when driving
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json

@dataclass
class DriverComfortScore:
    """Comprehensive driver comfort score with breakdown."""
    overall_comfort: float  # 0-100, higher is better
    posture_quality: float  # 0-100
    ergonomic_risk: float   # 0-100, lower is better
    fatigue_indicator: float # 0-100, lower is better
    comfort_category: str   # "Excellent", "Good", "Fair", "Poor"
    recommendations: List[str]
    seat_comparison: Dict[str, float]  # For comparing different seat types

class DriverComfortAnalyzer:
    """
    Analyzes driver comfort based on research-backed ergonomic principles.
    Focuses on driving-specific posture metrics and comfort indicators.
    """
    
    def __init__(self):
        # Research-based comfort thresholds (from ergonomic studies)
        self.comfort_thresholds = {
            'excellent': {'min': 85, 'color': (0, 255, 0)},
            'good': {'min': 70, 'color': (0, 255, 255)},
            'fair': {'min': 50, 'color': (0, 165, 255)},
            'poor': {'min': 0, 'color': (0, 0, 255)}
        }
        
        # Driving-specific angle ranges (from automotive ergonomics research)
        self.ideal_driving_angles = {
            'trunk_from_vertical': {'min': -5, 'max': 15, 'weight': 0.25},
            'neck_from_vertical': {'min': -10, 'max': 20, 'weight': 0.20},
            'left_hip_angle': {'min': 85, 'max': 110, 'weight': 0.15},
            'right_hip_angle': {'min': 85, 'max': 110, 'weight': 0.15},
            'left_knee_angle': {'min': 100, 'max': 130, 'weight': 0.15},
            'right_knee_angle': {'min': 100, 'max': 130, 'weight': 0.10}
        }
        
        # Fatigue indicators (from driver behavior studies)
        self.fatigue_indicators = {
            'posture_drift': {'threshold': 0.3, 'weight': 0.4},
            'fidgeting': {'threshold': 0.25, 'weight': 0.3},
            'slouching_tendency': {'threshold': 0.2, 'weight': 0.3}
        }
    
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
        
        # 1. Posture Quality Score (0-100)
        posture_scores = {}
        total_weight = 0
        
        for angle_name, ideal_range in self.ideal_driving_angles.items():
            if angle_name in angles:
                current_angle = angles[angle_name]
                min_val = ideal_range['min']
                max_val = ideal_range['max']
                weight = ideal_range['weight']
                
                # Calculate score based on deviation from ideal range
                if min_val <= current_angle <= max_val:
                    score = 100  # Perfect
                else:
                    # Calculate penalty based on distance from ideal range
                    if current_angle < min_val:
                        penalty = (min_val - current_angle) / abs(min_val) * 100
                    else:
                        penalty = (current_angle - max_val) / abs(max_val) * 100
                    score = max(0, 100 - penalty)
                
                posture_scores[angle_name] = score
                total_weight += weight
        
        # Weighted average posture score
        weighted_posture_score = sum(
            score * self.ideal_driving_angles[angle]['weight'] 
            for angle, score in posture_scores.items()
        ) / total_weight if total_weight > 0 else 0
        
        # 2. Ergonomic Risk Score (0-100, lower is better)
        risk_factors = []
        
        # Trunk angle risk (most critical for driving)
        if 'trunk_from_vertical' in angles:
            trunk_angle = angles['trunk_from_vertical']
            if trunk_angle > 20:  # Forward lean
                risk_factors.append(('Forward trunk lean', 30))
            elif trunk_angle < -10:  # Backward lean
                risk_factors.append(('Backward trunk lean', 25))
        
        # Neck angle risk
        if 'neck_from_vertical' in angles:
            neck_angle = angles['neck_from_vertical']
            if abs(neck_angle) > 25:
                risk_factors.append(('Neck strain', 20))
        
        # Hip and knee angles
        for side in ['left', 'right']:
            hip_key = f'{side}_hip_angle'
            knee_key = f'{side}_knee_angle'
            
            if hip_key in angles and angles[hip_key] < 80:
                risk_factors.append((f'{side.title()} hip strain', 15))
            
            if knee_key in angles and angles[knee_key] < 95:
                risk_factors.append((f'{side.title()} knee strain', 10))
        
        # Calculate total risk score
        total_risk = sum(risk for _, risk in risk_factors)
        ergonomic_risk = min(100, total_risk)
        
        # 3. Fatigue Indicator (0-100, lower is better)
        fatigue_score = 0
        if posture_history and len(posture_history) > 10:
            # Analyze posture drift over time
            recent_angles = posture_history[-10:]
            trunk_drift = np.std([a.get('trunk_from_vertical', 0) for a in recent_angles])
            neck_drift = np.std([a.get('neck_from_vertical', 0) for a in recent_angles])
            
            # Calculate fatigue based on posture instability
            fatigue_score = min(100, (trunk_drift + neck_drift) * 10)
        
        # 4. Overall Comfort Score
        # Weighted combination of posture quality, risk, and fatigue
        overall_comfort = (
            weighted_posture_score * 0.5 +
            (100 - ergonomic_risk) * 0.3 +
            (100 - fatigue_score) * 0.2
        )
        
        # 5. Comfort Category
        comfort_category = "Poor"
        for category, threshold in self.comfort_thresholds.items():
            if overall_comfort >= threshold['min']:
                comfort_category = category.title()
                break
        
        # 6. Generate Recommendations
        recommendations = self._generate_recommendations(
            angles, posture_scores, risk_factors, fatigue_score, session_duration
        )
        
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
            seat_comparison=seat_comparison
        )
    
    def _generate_recommendations(self, angles: Dict[str, float], 
                                posture_scores: Dict[str, float],
                                risk_factors: List[Tuple[str, int]],
                                fatigue_score: float,
                                session_duration: float) -> List[str]:
        """Generate specific recommendations for driver comfort improvement."""
        
        recommendations = []
        
        # Posture-specific recommendations
        if 'trunk_from_vertical' in angles:
            trunk_angle = angles['trunk_from_vertical']
            if trunk_angle > 15:
                recommendations.append("Adjust seat backrest to reduce forward lean")
            elif trunk_angle < -5:
                recommendations.append("Adjust seat backrest to reduce backward lean")
        
        if 'neck_from_vertical' in angles:
            neck_angle = angles['neck_from_vertical']
            if abs(neck_angle) > 20:
                recommendations.append("Adjust headrest height and seat position for better neck alignment")
        
        # Hip and knee recommendations
        for side in ['left', 'right']:
            hip_key = f'{side}_hip_angle'
            knee_key = f'{side}_knee_angle'
            
            if hip_key in angles and angles[hip_key] < 85:
                recommendations.append(f"Adjust {side} side seat height and angle for better hip support")
            
            if knee_key in angles and angles[knee_key] < 100:
                recommendations.append(f"Adjust {side} side seat position for better knee angle")
        
        # Fatigue-based recommendations
        if fatigue_score > 50:
            recommendations.append("Consider taking a break - posture instability detected")
        
        # Duration-based recommendations
        if session_duration > 3600:  # 1 hour
            recommendations.append("Long driving session - consider seat adjustments or breaks")
        
        # General recommendations
        if len(recommendations) < 3:
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
            comfort_scores = []
            for data_point in posture_data:
                if 'angles' in data_point:
                    score = self.calculate_comfort_score(data_point['angles'])
                    comfort_scores.append(score.overall_comfort)
            
            if comfort_scores:
                comparison_results[seat_type] = {
                    'average_comfort': np.mean(comfort_scores),
                    'comfort_std': np.std(comfort_scores),
                    'min_comfort': np.min(comfort_scores),
                    'max_comfort': np.max(comfort_scores),
                    'sample_size': len(comfort_scores)
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
        
        # Calculate comfort scores for each data point
        comfort_scores = []
        angles_history = []
        
        for data_point in session_data:
            if 'angles' in data_point:
                angles_history.append(data_point['angles'])
                score = self.calculate_comfort_score(
                    data_point['angles'], 
                    angles_history,
                    session_info.get('duration', 0)
                )
                comfort_scores.append(score)
        
        if not comfort_scores:
            return {"error": "No valid posture data found"}
        
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
            'data_points': len(comfort_scores)
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

# Convenience function for quick comfort scoring
def score_driver_comfort(angles: Dict[str, float], 
                        posture_history: List[Dict] = None,
                        session_duration: float = 0) -> DriverComfortScore:
    """Quick function to get driver comfort score."""
    analyzer = DriverComfortAnalyzer()
    return analyzer.calculate_comfort_score(angles, posture_history, session_duration)
