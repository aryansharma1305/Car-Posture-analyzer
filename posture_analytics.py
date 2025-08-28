import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from posture_scoring import PostureScore, score_posture

class PostureAnalytics:
    """Analyzes posture data and generates comprehensive reports."""
    
    def __init__(self, data_dir: str = "posture_logs"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def load_session_data(self, session_file: str) -> pd.DataFrame:
        """Load data from a specific session file."""
        file_path = self.data_dir / session_file
        if not file_path.exists():
            raise FileNotFoundError(f"Session file not found: {session_file}")
        
        df = pd.read_csv(file_path)
        
        # Add timestamp column if not present
        if 'timestamp' not in df.columns and 't_sec' in df.columns:
            df['timestamp'] = pd.to_datetime('now') - pd.to_timedelta(df['t_sec'], unit='s')
        
        return df
    
    def load_all_sessions(self, pattern: str = "*.csv") -> pd.DataFrame:
        """Load and combine data from all session files."""
        all_data = []
        
        for file_path in self.data_dir.glob(pattern):
            try:
                df = pd.read_csv(file_path)
                df['session_file'] = file_path.name
                df['session_date'] = pd.to_datetime(file_path.stem.split('_')[1:3], format='%Y%m%d_%H%M%S')
                
                # Add timestamp if t_sec is present
                if 't_sec' in df.columns:
                    df['timestamp'] = df['session_date'] + pd.to_timedelta(df['t_sec'], unit='s')
                
                all_data.append(df)
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
        
        if not all_data:
            return pd.DataFrame()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values('timestamp')
        return combined_df
    
    def analyze_session(self, session_file: str) -> Dict:
        """Analyze a single session and return comprehensive statistics."""
        df = self.load_session_data(session_file)
        
        if df.empty:
            return {"error": "No data found"}
        
        # Basic session info
        session_info = {
            "file": session_file,
            "duration_minutes": (df['t_sec'].max() - df['t_sec'].min()) / 60 if 't_sec' in df.columns else 0,
            "total_samples": len(df),
            "start_time": df['timestamp'].min() if 'timestamp' in df.columns else None,
            "end_time": df['timestamp'].max() if 'timestamp' in df.columns else None
        }
        
        # Posture classification analysis
        if 'body_state' in df.columns:
            posture_counts = df['body_state'].value_counts().to_dict()
            session_info["posture_distribution"] = posture_counts
            
            # Calculate time spent in each posture
            if 't_sec' in df.columns:
                posture_times = {}
                for posture in df['body_state'].unique():
                    posture_data = df[df['body_state'] == posture]
                    if len(posture_data) > 1:
                        time_spent = (posture_data['t_sec'].max() - posture_data['t_sec'].min())
                        posture_times[posture] = time_spent / 60  # Convert to minutes
                    else:
                        posture_times[posture] = 0
                session_info["posture_times_minutes"] = posture_times
        
        # Angle analysis
        angle_columns = [col for col in df.columns if any(angle in col for angle in 
                        ['neck', 'trunk', 'hip', 'knee', 'shoulder'])]
        
        if angle_columns:
            angle_stats = {}
            for col in angle_columns:
                if col in df.columns:
                    angle_stats[col] = {
                        "mean": df[col].mean(),
                        "std": df[col].std(),
                        "min": df[col].min(),
                        "max": df[col].max(),
                        "median": df[col].median()
                    }
            session_info["angle_statistics"] = angle_stats
            
            # Posture quality scoring (if we have all required angles)
            required_angles = ['neck_from_vertical', 'trunk_from_vertical', 
                             'left_hip_angle', 'right_hip_angle',
                             'left_knee_angle', 'right_knee_angle',
                             'left_shoulder_elev', 'right_shoulder_elev']
            
            if all(angle in df.columns for angle in required_angles):
                scores = []
                for _, row in df.iterrows():
                    try:
                        angles = {angle: row[angle] for angle in required_angles}
                        score = score_posture(angles)
                        scores.append(score.overall_score)
                    except:
                        continue
                
                if scores:
                    session_info["posture_quality"] = {
                        "mean_score": np.mean(scores),
                        "min_score": np.min(scores),
                        "max_score": np.max(scores),
                        "score_std": np.std(scores)
                    }
        
        return session_info
    
    def generate_session_report(self, session_file: str, output_file: Optional[str] = None) -> str:
        """Generate a detailed report for a specific session."""
        analysis = self.analyze_session(session_file)
        
        if "error" in analysis:
            return f"Error: {analysis['error']}"
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append(f"POSTURE ANALYSIS REPORT")
        report_lines.append(f"Session: {analysis['file']}")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Session overview
        report_lines.append("SESSION OVERVIEW:")
        report_lines.append(f"  Duration: {analysis['duration_minutes']:.1f} minutes")
        report_lines.append(f"  Total samples: {analysis['total_samples']}")
        if analysis['start_time']:
            report_lines.append(f"  Start time: {analysis['start_time']}")
        if analysis['end_time']:
            report_lines.append(f"  End time: {analysis['end_time']}")
        report_lines.append("")
        
        # Posture distribution
        if "posture_distribution" in analysis:
            report_lines.append("POSTURE DISTRIBUTION:")
            for posture, count in analysis["posture_distribution"].items():
                percentage = (count / analysis["total_samples"]) * 100
                report_lines.append(f"  {posture}: {count} samples ({percentage:.1f}%)")
            report_lines.append("")
            
            if "posture_times_minutes" in analysis:
                report_lines.append("TIME SPENT IN EACH POSTURE:")
                for posture, time_min in analysis["posture_times_minutes"].items():
                    report_lines.append(f"  {posture}: {time_min:.1f} minutes")
                report_lines.append("")
        
        # Angle statistics
        if "angle_statistics" in analysis:
            report_lines.append("ANGLE STATISTICS (degrees):")
            for angle, stats in analysis["angle_statistics"].items():
                report_lines.append(f"  {angle}:")
                report_lines.append(f"    Mean: {stats['mean']:.1f}°")
                report_lines.append(f"    Std: {stats['std']:.1f}°")
                report_lines.append(f"    Range: {stats['min']:.1f}° - {stats['max']:.1f}°")
            report_lines.append("")
        
        # Posture quality
        if "posture_quality" in analysis:
            report_lines.append("POSTURE QUALITY SCORES:")
            quality = analysis["posture_quality"]
            report_lines.append(f"  Overall mean score: {quality['mean_score']:.1f}/100")
            report_lines.append(f"  Score range: {quality['min_score']:.1f} - {quality['max_score']:.1f}")
            report_lines.append(f"  Score consistency (std): {quality['score_std']:.1f}")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS:")
        if "posture_quality" in analysis:
            quality = analysis["posture_quality"]
            if quality['mean_score'] >= 80:
                report_lines.append("  ✅ Excellent posture! Keep up the good work.")
            elif quality['mean_score'] >= 60:
                report_lines.append("  ⚠️  Good posture with room for improvement.")
                report_lines.append("  Consider the specific angle recommendations above.")
            else:
                report_lines.append("  ❌ Poor posture detected. Immediate attention needed.")
                report_lines.append("  Review ergonomic setup and take frequent breaks.")
        else:
            report_lines.append("  Unable to calculate posture quality scores.")
            report_lines.append("  Ensure all required angle data is available.")
        
        report_lines.append("")
        report_lines.append("=" * 60)
        
        report_text = "\n".join(report_lines)
        
        # Save to file if requested
        if output_file:
            output_path = self.data_dir / output_file
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {output_path}")
        
        return report_text
    
    def generate_summary_report(self, days: int = 7, output_file: Optional[str] = None) -> str:
        """Generate a summary report for the last N days."""
        all_data = self.load_all_sessions()
        
        if all_data.empty:
            return "No data found for summary report."
        
        # Filter by date
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_data = all_data[all_data['session_date'] >= cutoff_date]
        
        if recent_data.empty:
            return f"No data found in the last {days} days."
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append(f"POSTURE ANALYSIS SUMMARY REPORT")
        report_lines.append(f"Period: Last {days} days")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Overall statistics
        total_sessions = recent_data['session_file'].nunique()
        total_duration = recent_data.groupby('session_file')['t_sec'].apply(
            lambda x: (x.max() - x.min()) / 60 if len(x) > 1 else 0
        ).sum()
        
        report_lines.append("OVERALL STATISTICS:")
        report_lines.append(f"  Total sessions: {total_sessions}")
        report_lines.append(f"  Total time tracked: {total_duration:.1f} minutes")
        report_lines.append(f"  Average session length: {total_duration/total_sessions:.1f} minutes")
        report_lines.append("")
        
        # Session list
        report_lines.append("SESSIONS ANALYZED:")
        for session_file in sorted(recent_data['session_file'].unique()):
            session_data = recent_data[recent_data['session_file'] == session_file]
            duration = (session_data['t_sec'].max() - session_data['t_sec'].min()) / 60 if len(session_data) > 1 else 0
            date = session_data['session_date'].iloc[0]
            report_lines.append(f"  {session_file}: {duration:.1f} min on {date.strftime('%Y-%m-%d')}")
        report_lines.append("")
        
        # Posture trends
        if 'body_state' in recent_data.columns:
            report_lines.append("POSTURE TRENDS:")
            posture_counts = recent_data['body_state'].value_counts()
            for posture, count in posture_counts.items():
                percentage = (count / len(recent_data)) * 100
                report_lines.append(f"  {posture}: {count} samples ({percentage:.1f}%)")
            report_lines.append("")
        
        # Angle trends
        angle_columns = [col for col in recent_data.columns if any(angle in col for angle in 
                        ['neck', 'trunk', 'hip', 'knee', 'shoulder'])]
        
        if angle_columns:
            report_lines.append("ANGLE TRENDS (averages):")
            for col in angle_columns:
                if col in recent_data.columns:
                    mean_val = recent_data[col].mean()
                    report_lines.append(f"  {col}: {mean_val:.1f}°")
            report_lines.append("")
        
        report_lines.append("=" * 60)
        
        report_text = "\n".join(report_lines)
        
        # Save to file if requested
        if output_file:
            output_path = self.data_dir / output_file
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"Summary report saved to: {output_path}")
        
        return report_text
    
    def create_visualizations(self, session_file: str, output_dir: Optional[str] = None) -> List[str]:
        """Create visualizations for a session."""
        df = self.load_session_data(session_file)
        
        if df.empty:
            return ["No data found for visualization"]
        
        output_path = Path(output_dir) if output_dir else self.data_dir
        output_path.mkdir(exist_ok=True)
        
        created_files = []
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Posture distribution pie chart
        if 'body_state' in df.columns:
            plt.figure(figsize=(10, 6))
            posture_counts = df['body_state'].value_counts()
            plt.pie(posture_counts.values, labels=posture_counts.index, autopct='%1.1f%%')
            plt.title(f'Posture Distribution - {session_file}')
            pie_file = output_path / f"{session_file.replace('.csv', '_posture_distribution.png')}"
            plt.savefig(pie_file, dpi=300, bbox_inches='tight')
            plt.close()
            created_files.append(str(pie_file))
        
        # 2. Angle trends over time
        angle_columns = [col for col in df.columns if any(angle in col for angle in 
                        ['neck', 'trunk', 'hip', 'knee', 'shoulder'])]
        
        if angle_columns and 't_sec' in df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, angle in enumerate(angle_columns[:4]):  # Show first 4 angles
                if i < len(axes):
                    axes[i].plot(df['t_sec'], df[angle], linewidth=2)
                    axes[i].set_title(f'{angle} Over Time')
                    axes[i].set_xlabel('Time (seconds)')
                    axes[i].set_ylabel('Angle (degrees)')
                    axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            trends_file = output_path / f"{session_file.replace('.csv', '_angle_trends.png')}"
            plt.savefig(trends_file, dpi=300, bbox_inches='tight')
            plt.close()
            created_files.append(str(trends_file))
        
        # 3. Posture quality scores over time (if available)
        required_angles = ['neck_from_vertical', 'trunk_from_vertical', 
                         'left_hip_angle', 'right_hip_angle',
                         'left_knee_angle', 'right_knee_angle',
                         'left_shoulder_elev', 'right_shoulder_elev']
        
        if all(angle in df.columns for angle in required_angles) and 't_sec' in df.columns:
            scores = []
            for _, row in df.iterrows():
                try:
                    angles = {angle: row[angle] for angle in required_angles}
                    score = score_posture(angles)
                    scores.append(score.overall_score)
                except:
                    scores.append(np.nan)
            
            if any(not np.isnan(score) for score in scores):
                plt.figure(figsize=(12, 6))
                plt.plot(df['t_sec'], scores, linewidth=2, color='green')
                plt.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Good Posture Threshold')
                plt.axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='Acceptable Threshold')
                plt.title(f'Posture Quality Score Over Time - {session_file}')
                plt.xlabel('Time (seconds)')
                plt.ylabel('Posture Score (0-100)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 100)
                
                quality_file = output_path / f"{session_file.replace('.csv', '_quality_scores.png')}"
                plt.savefig(quality_file, dpi=300, bbox_inches='tight')
                plt.close()
                created_files.append(str(quality_file))
        
        return created_files

# Convenience functions
def analyze_session(session_file: str) -> Dict:
    """Quick function to analyze a session."""
    analytics = PostureAnalytics()
    return analytics.analyze_session(session_file)

def generate_report(session_file: str, output_file: Optional[str] = None) -> str:
    """Quick function to generate a session report."""
    analytics = PostureAnalytics()
    return analytics.generate_session_report(session_file, output_file)
