import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from posture_scoring import PostureScore, score_posture

_SESSION_STAMP_RE = re.compile(r"_(\d{8})_(\d{6})$")


def session_timestamp_from_name(stem: str) -> Optional[pd.Timestamp]:
    """Parse the trailing _YYYYMMDD_HHMMSS stamp every log filename carries.

    Anchored at the end of the stem so arbitrary prefixes work:
    'driver_comfort_20250829_153444' and 'angles_20250811_082025' both parse.
    Returns None when no stamp is present, so callers can report the file.
    """
    m = _SESSION_STAMP_RE.search(stem)
    if not m:
        return None
    try:
        return pd.to_datetime(m.group(1) + m.group(2), format="%Y%m%d%H%M%S")
    except ValueError:
        return None


class PostureAnalytics:
    """Analyzes posture data and generates comprehensive reports."""
    
    def __init__(self, data_dir: str = "posture_logs"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def load_session_data(self, session_file: str) -> pd.DataFrame:
        """Load one session file, normalised to a single time schema.

        Guarantees both 't_sec' (seconds from session start) and 'timestamp'
        (absolute datetime) whichever historical writer produced the file, so
        callers never have to ask which column means what. See
        _normalise_time_columns for the per-schema handling.
        """
        file_path = self.data_dir / session_file
        if not file_path.exists():
            raise FileNotFoundError(f"Session file not found: {session_file}")

        df = pd.read_csv(file_path)
        if df.empty:
            return df

        session_date = session_timestamp_from_name(file_path.stem)
        if session_date is None:
            # No filename stamp to anchor against. Relative durations still
            # work; only the absolute clock time is arbitrary.
            session_date = pd.Timestamp.now().normalize()

        normalised = self._normalise_time_columns(df, session_date)
        return normalised if normalised is not None else df
    
    def load_all_sessions(self, pattern: str = "*.csv") -> pd.DataFrame:
        """Load and combine data from all session files.

        Session logs exist in several historical schemas: some carry 't_sec'
        only, some an ISO 'timestamp' plus 't_sec', and some a numeric
        'timestamp' column that actually holds seconds since session start.
        All are normalised here to 't_sec' (float seconds) plus 'timestamp'
        (absolute datetime) so downstream code sees one shape.

        Files that cannot be normalised are reported, not silently dropped.
        """
        all_data = []
        failures = []

        for file_path in sorted(self.data_dir.glob(pattern)):
            session_date = session_timestamp_from_name(file_path.stem)
            if session_date is None:
                failures.append((file_path.name, "filename has no _YYYYMMDD_HHMMSS stamp"))
                continue

            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                failures.append((file_path.name, f"unreadable: {e}"))
                continue

            if df.empty:
                failures.append((file_path.name, "no rows"))
                continue

            df['session_file'] = file_path.name
            df['session_date'] = session_date

            normalised = self._normalise_time_columns(df, session_date)
            if normalised is None:
                failures.append((file_path.name, "no usable 't_sec' or 'timestamp' column"))
                continue

            all_data.append(normalised)

        if failures:
            total = len(all_data) + len(failures)
            print(f"Skipped {len(failures)} of {total} file(s) in {self.data_dir}:")
            for name, reason in failures:
                print(f"  - {name}: {reason}")

        if not all_data:
            return pd.DataFrame()

        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df.sort_values('timestamp').reset_index(drop=True)

    @staticmethod
    def _normalise_time_columns(df: pd.DataFrame, session_date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Give df both 't_sec' (seconds from session start) and 'timestamp'.

        Returns None when the frame carries no recoverable time column.
        """
        if 'timestamp' in df.columns:
            numeric = pd.to_numeric(df['timestamp'], errors='coerce')
            if numeric.notna().all():
                # Legacy writer stored elapsed seconds under the name
                # 'timestamp' (posture_live_full.py). Recover it as t_sec and
                # rebuild an absolute timestamp from the filename stamp.
                df['t_sec'] = numeric
                df['timestamp'] = session_date + pd.to_timedelta(numeric, unit='s')
                return df

            parsed = pd.to_datetime(df['timestamp'], errors='coerce', format='mixed')
            if parsed.notna().any():
                df['timestamp'] = parsed
                if 't_sec' in df.columns:
                    df['t_sec'] = pd.to_numeric(df['t_sec'], errors='coerce')
                else:
                    df['t_sec'] = (parsed - session_date).dt.total_seconds()
                return df

        if 't_sec' in df.columns:
            t_sec = pd.to_numeric(df['t_sec'], errors='coerce')
            df['t_sec'] = t_sec
            df['timestamp'] = session_date + pd.to_timedelta(t_sec, unit='s')
            return df

        return None
    
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
                    # The canonical schema writes every column in every file,
                    # so a monitor that does not compute an angle leaves it
                    # entirely empty. Summarising such a column yields NaN and
                    # a "Mean of empty slice" warning, so skip it outright.
                    numeric = pd.to_numeric(df[col], errors='coerce')
                    if numeric.notna().sum() == 0:
                        continue
                    df[col] = numeric
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
