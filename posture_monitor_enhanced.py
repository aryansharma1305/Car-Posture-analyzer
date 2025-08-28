import cv2
import time
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from pose_core import PoseExtractor
from posture_angles import compute_angles
from smoothing import AngleSmoother
from posture_fullbody_rules import detect_body_state, classify_upper_body, BodyConfig
from posture_scoring import PostureQualityScorer, PostureScore
from posture_analytics import PostureAnalytics

class EnhancedPostureMonitor:
    """Enhanced real-time posture monitoring with quality scoring and analytics."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # Initialize components
        self.pose_extractor = PoseExtractor(
            min_detection_confidence=self.config['detection_confidence'],
            min_tracking_confidence=self.config['tracking_confidence']
        )
        self.angle_smoother = AngleSmoother(alpha=self.config['smoothing_alpha'])
        self.body_config = BodyConfig()
        self.quality_scorer = PostureQualityScorer()
        self.analytics = PostureAnalytics()
        
        # Session tracking
        self.session_start = None
        self.current_posture = None
        self.posture_durations = {}
        self.quality_history = []
        self.break_reminders = []
        
        # UI state
        self.show_angles = True
        self.show_quality = True
        self.show_recommendations = True
        self.show_break_timer = True
        
        # Break timer
        self.last_break_time = time.monotonic()
        self.break_interval = self.config['break_interval_minutes'] * 60  # Convert to seconds
        
        # Setup logging
        self._setup_logging()
    
    def _default_config(self) -> Dict:
        """Default configuration for the monitor."""
        return {
            'detection_confidence': 0.6,
            'tracking_confidence': 0.6,
            'smoothing_alpha': 0.25,
            'break_interval_minutes': 30,
            'quality_threshold_good': 80,
            'quality_threshold_warning': 60,
            'logging_interval_seconds': 1.0,
            'save_visualizations': True,
            'alert_poor_posture': True,
            'max_session_duration_hours': 4
        }
    
    def _setup_logging(self):
        """Setup logging directories and files."""
        self.log_dir = Path("posture_logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Create session-specific log files
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log_path = self.log_dir / f"enhanced_{session_id}.csv"
        self.quality_log_path = self.log_dir / f"quality_{session_id}.csv"
        self.session_config_path = self.log_dir / f"config_{session_id}.json"
        
        # Save configuration
        with open(self.session_config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Initialize CSV headers
        self._init_csv_headers()
    
    def _init_csv_headers(self):
        """Initialize CSV files with headers."""
        # Main session log
        with open(self.session_log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 't_sec', 'body_state', 'upper_body', 'lower_visible',
                'lean_side', 'trunk_deg', 'neck_deg', 'knee_L', 'knee_R',
                'hip_L', 'hip_R', 'overall_quality_score', 'risk_level'
            ])
        
        # Quality log
        with open(self.quality_log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 't_sec', 'overall_score', 'upper_body_score',
                'lower_body_score', 'shoulders_score', 'symmetry_score',
                'risk_level', 'recommendations_count'
            ])
    
    def start_session(self):
        """Start a new monitoring session."""
        self.session_start = time.monotonic()
        self.last_break_time = self.session_start
        print(f"Started posture monitoring session at {datetime.now().strftime('%H:%M:%S')}")
        print(f"Break reminders every {self.config['break_interval_minutes']} minutes")
        print(f"Session logs: {self.session_log_path}")
    
    def process_frame(self, frame) -> Dict:
        """Process a single frame and return analysis results."""
        results, pts, (img_h, img_w) = self.pose_extractor.process_bgr(frame)
        
        if results is None:
            return {"status": "no_pose", "message": "No person detected"}
        
        # Draw pose landmarks
        self.pose_extractor.draw(frame, results)
        
        # Get visibility information
        vis = self._get_visibility(results)
        
        # Compute and smooth angles
        raw_angles = compute_angles(pts)
        smoothed_angles = self.angle_smoother(raw_angles)
        
        # Analyze posture
        body_state, extras = detect_body_state(pts, smoothed_angles, vis, (img_h, img_w), self.body_config)
        upper_body_state = classify_upper_body(smoothed_angles)
        
        # Calculate quality score
        quality_score = self.quality_scorer.score_posture(smoothed_angles)
        
        # Update tracking
        self._update_posture_tracking(body_state, quality_score)
        
        # Check break timer
        break_warning = self._check_break_timer()
        
        # Prepare results
        results = {
            "status": "success",
            "body_state": body_state,
            "upper_body": upper_body_state,
            "angles": smoothed_angles,
            "quality_score": quality_score,
            "extras": extras,
            "break_warning": break_warning,
            "posture_durations": self.posture_durations.copy(),
            "session_duration": (time.monotonic() - self.session_start) / 60 if self.session_start else 0
        }
        
        # Log data periodically
        self._log_data(results)
        
        return results
    
    def _get_visibility(self, results) -> Dict:
        """Extract visibility information from pose results."""
        lm = results.pose_landmarks.landmark
        idx = {
            "left_shoulder": 11, "right_shoulder": 12,
            "left_hip": 23, "right_hip": 24,
            "left_knee": 25, "right_knee": 26,
            "left_ankle": 27, "right_ankle": 28,
        }
        return {k: lm[i].visibility for k, i in idx.items()}
    
    def _update_posture_tracking(self, body_state: str, quality_score: PostureScore):
        """Update posture duration tracking and quality history."""
        current_time = time.monotonic()
        
        # Update posture durations
        if self.current_posture != body_state:
            if self.current_posture:
                duration = current_time - self.posture_start_time
                self.posture_durations[self.current_posture] = self.posture_durations.get(self.current_posture, 0) + duration
            
            self.current_posture = body_state
            self.posture_start_time = current_time
        
        # Update quality history
        self.quality_history.append({
            'timestamp': current_time,
            'score': quality_score.overall_score,
            'risk_level': quality_score.risk_level
        })
        
        # Keep only recent history (last 100 samples)
        if len(self.quality_history) > 100:
            self.quality_history = self.quality_history[-100:]
    
    def _check_break_timer(self) -> Optional[str]:
        """Check if it's time for a break reminder."""
        current_time = time.monotonic()
        time_since_break = current_time - self.last_break_time
        
        if time_since_break >= self.break_interval:
            self.last_break_time = current_time
            return f"Time for a {self.config['break_interval_minutes']}-minute break!"
        
        return None
    
    def _log_data(self, results: Dict):
        """Log data to CSV files."""
        current_time = time.monotonic()
        session_time = current_time - self.session_start if self.session_start else 0
        
        # Log main session data
        if int(session_time) != int(session_time - 1):  # Log roughly once per second
            with open(self.session_log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    round(session_time, 2),
                    results['body_state'],
                    results['upper_body'],
                    results['extras']['lower_visible'],
                    results['extras'].get('lean_side', ''),
                    round(results['angles']['trunk_from_vertical'], 1),
                    round(results['angles']['neck_from_vertical'], 1),
                    round(results['angles']['left_knee_angle'], 1),
                    round(results['angles']['right_knee_angle'], 1),
                    round(results['angles']['left_hip_angle'], 1),
                    round(results['angles']['right_hip_angle'], 1),
                    results['quality_score'].overall_score,
                    results['quality_score'].risk_level
                ])
            
            # Log quality data
            with open(self.quality_log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    round(session_time, 2),
                    results['quality_score'].overall_score,
                    results['quality_score'].category_scores['upper_body'],
                    results['quality_score'].category_scores['lower_body'],
                    results['quality_score'].category_scores['shoulders'],
                    results['quality_score'].category_scores['symmetry'],
                    results['quality_score'].risk_level,
                    len(results['quality_score'].recommendations)
                ])
    
    def get_session_summary(self) -> Dict:
        """Get a summary of the current session."""
        if not self.session_start:
            return {"error": "No active session"}
        
        session_duration = (time.monotonic() - self.session_start) / 60
        
        # Calculate average quality score
        if self.quality_history:
            avg_quality = sum(h['score'] for h in self.quality_history) / len(self.quality_history)
            risk_distribution = {}
            for h in self.quality_history:
                risk = h['risk_level']
                risk_distribution[risk] = risk_distribution.get(risk, 0) + 1
        else:
            avg_quality = 0
            risk_distribution = {}
        
        return {
            "session_duration_minutes": round(session_duration, 1),
            "posture_durations": {k: round(v/60, 1) for k, v in self.posture_durations.items()},
            "average_quality_score": round(avg_quality, 1),
            "risk_distribution": risk_distribution,
            "total_samples": len(self.quality_history),
            "break_reminders_given": len(self.break_reminders)
        }
    
    def end_session(self):
        """End the current monitoring session."""
        if not self.session_start:
            return
        
        # Finalize posture duration for current posture
        if self.current_posture:
            current_time = time.monotonic()
            duration = current_time - self.posture_start_time
            self.posture_durations[self.current_posture] = self.posture_durations.get(self.current_posture, 0) + duration
        
        # Generate session summary
        summary = self.get_session_summary()
        
        # Save summary
        summary_path = self.log_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSession ended. Duration: {summary['session_duration_minutes']:.1f} minutes")
        print(f"Average quality score: {summary['average_quality_score']:.1f}/100")
        print(f"Summary saved to: {summary_path}")
        
        # Reset session state
        self.session_start = None
        self.current_posture = None
        self.posture_durations = {}
        self.quality_history = []
        self.break_reminders = []

def draw_enhanced_ui(frame, results: Dict, monitor: EnhancedPostureMonitor):
    """Draw enhanced UI elements on the frame."""
    h, w = frame.shape[:2]
    
    # Colors
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    YELLOW = (0, 255, 255)
    WHITE = (255, 255, 255)
    BLUE = (255, 0, 0)
    ORANGE = (0, 165, 255)
    
    def draw_text(text, x, y, color=WHITE, scale=0.7, thickness=2):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    
    # Header information
    y_offset = 30
    draw_text(f"Body: {results['body_state']}", 10, y_offset, 
              GREEN if results['body_state'] in ['Standing', 'Sitting'] else YELLOW, 0.9, 2)
    y_offset += 25
    
    draw_text(f"Upper: {results['upper_body']}", 10, y_offset, WHITE)
    y_offset += 25
    
    # Quality score display
    quality = results['quality_score']
    quality_color = GREEN if quality.overall_score >= 80 else (YELLOW if quality.overall_score >= 60 else RED)
    draw_text(f"Quality: {quality.overall_score:.1f}/100 ({quality.risk_level} Risk)", 10, y_offset, quality_color, 0.8, 2)
    y_offset += 25
    
    # Category scores
    for category, score in quality.category_scores.items():
        color = GREEN if score >= 80 else (YELLOW if score >= 60 else RED)
        draw_text(f"{category}: {score:.1f}", 10, y_offset, color, 0.6)
        y_offset += 20
    
    y_offset += 10
    
    # Key angles
    angles = results['angles']
    draw_text(f"Neck: {angles['neck_from_vertical']:.1f}°", 10, y_offset, WHITE, 0.6)
    y_offset += 20
    draw_text(f"Trunk: {angles['trunk_from_vertical']:.1f}°", 10, y_offset, WHITE, 0.6)
    y_offset += 20
    draw_text(f"Hips: {angles['left_hip_angle']:.1f}°/{angles['right_hip_angle']:.1f}°", 10, y_offset, WHITE, 0.6)
    y_offset += 20
    
    # Break timer
    if results.get('break_warning'):
        draw_text("BREAK TIME!", w - 200, 30, RED, 1.0, 3)
        draw_text(results['break_warning'], w - 200, 60, YELLOW, 0.7)
    
    # Session duration
    session_min = results['session_duration']
    draw_text(f"Session: {session_min:.1f} min", w - 200, h - 60, WHITE, 0.6)
    
    # Posture duration summary
    y_offset = h - 100
    for posture, duration in results['posture_durations'].items():
        draw_text(f"{posture}: {duration/60:.1f}m", w - 200, y_offset, WHITE, 0.6)
        y_offset += 20
    
    # Recommendations (if quality is poor)
    if quality.overall_score < 60 and quality.recommendations:
        y_offset = 200
        draw_text("RECOMMENDATIONS:", 10, y_offset, YELLOW, 0.7, 2)
        y_offset += 25
        for i, rec in enumerate(quality.recommendations[:3]):  # Show first 3
            draw_text(f"{i+1}. {rec}", 10, y_offset, WHITE, 0.5)
            y_offset += 18

def main():
    """Main function to run the enhanced posture monitor."""
    # Initialize monitor
    monitor = EnhancedPostureMonitor()
    
    # Start camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not found.")
        return
    
    # Start session
    monitor.start_session()
    
    print("Enhanced Posture Monitor Started")
    print("Press 'q' to quit, 'r' to show/hide recommendations, 'a' to show/hide angles")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture frame.")
                break
            
            # Process frame
            results = monitor.process_frame(frame)
            
            # Draw UI
            if results['status'] == 'success':
                draw_enhanced_ui(frame, results, monitor)
            else:
                cv2.putText(frame, results['message'], (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Show frame
            cv2.imshow("Enhanced Posture Monitor", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                monitor.show_recommendations = not monitor.show_recommendations
            elif key == ord('a'):
                monitor.show_angles = not monitor.show_angles
            elif key == ord('s'):
                # Show session summary
                summary = monitor.get_session_summary()
                print("\n" + "="*50)
                print("SESSION SUMMARY")
                print("="*50)
                for key, value in summary.items():
                    print(f"{key}: {value}")
                print("="*50)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        monitor.end_session()
        cap.release()
        cv2.destroyAllWindows()
        print("Enhanced Posture Monitor stopped.")

if __name__ == "__main__":
    main()
