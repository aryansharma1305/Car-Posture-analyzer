#!/usr/bin/env python3
"""
Driver Comfort Reporter for Mahindra & Mahindra
Generates comprehensive reports for seat ergonomics research
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple
import argparse

from driver_comfort_analyzer import DriverComfortAnalyzer
from driver_model import DRIVING, get_reference
from session_schema import SessionMetadata

class DriverComfortReporter:
    """
    Generates comprehensive driver comfort reports for Mahindra's seat ergonomics research.
    """
    
    # Sessions recorded before session_schema v2 carry no posture_reference in
    # their sidecar. They are all in-car recordings, so DRIVING is the right
    # fallback - but it is a fallback, and it is reported as one in
    # session_info['posture_reference_source'].
    DEFAULT_REFERENCE = DRIVING

    def __init__(self, data_dir: str = "posture_logs"):
        self.data_dir = Path(data_dir)
        self.analyzer = DriverComfortAnalyzer(self.DEFAULT_REFERENCE)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_session_data(self, session_file: str) -> Tuple[pd.DataFrame, Dict]:
        """Load session data and metadata."""
        file_path = self.data_dir / session_file
        
        if not file_path.exists():
            raise FileNotFoundError(f"Session file not found: {file_path}")
        
        # Load CSV data
        df = pd.read_csv(file_path)

        # Prefer the session sidecar over guesses. driver_id and seat_type were
        # previously hardcoded to 'test_driver'/'test_seat', which made every
        # report un-attributable and left analyze_seat_comparison unusable.
        session_id = session_file.replace('driver_comfort_', '').replace('.csv', '')
        meta = SessionMetadata.load(file_path)

        # 'duration' must be elapsed seconds. The legacy writer stored those in
        # a column called 'timestamp'; the canonical schema puts them in
        # 't_sec' and keeps 'timestamp' for the wall clock, so reading
        # 'timestamp' here would return a datetime, not a duration.
        if df.empty:
            duration = 0
        elif 't_sec' in df.columns:
            duration = pd.to_numeric(df['t_sec'], errors='coerce').max()
        else:
            duration = pd.to_numeric(df.get('timestamp'), errors='coerce').max()

        # What the WRITER recorded about this session's coverage, read straight
        # from the schema-3 columns. Reported alongside the reporter's own
        # re-scoring so a disagreement between them is visible rather than
        # silently resolved in favour of one.
        logged_coverage = {}
        if 'observation' in df.columns and not df.empty:
            observation = df['observation'].astype(str).str.strip().str.lower()
            recorded = observation.isin({'true', 'false'})
            if recorded.any():
                logged_coverage['rows'] = int(recorded.sum())
                logged_coverage['observations'] = int((observation == 'true').sum())
        if 'measured_fraction' in df.columns and not df.empty:
            fractions = pd.to_numeric(df['measured_fraction'], errors='coerce').dropna()
            if not fractions.empty:
                logged_coverage['mean_measured_fraction'] = round(
                    float(fractions.mean()), 3)
        if 'frame_confidence' in df.columns and not df.empty:
            confidence = pd.to_numeric(df['frame_confidence'], errors='coerce').dropna()
            if not confidence.empty:
                logged_coverage['mean_frame_confidence'] = round(
                    float(confidence.mean()), 3)

        named_reference = meta.posture_reference if meta else None
        session_info = {
            'session_id': meta.session_id if meta else session_id,
            'driver_id': (meta.driver_id if meta and meta.driver_id else 'unknown'),
            'seat_type': (meta.seat_id if meta and meta.seat_id else 'unknown'),
            'duration': duration,
            'metadata_source': 'sidecar' if meta else 'none',
            'posture_reference': named_reference or self.DEFAULT_REFERENCE.name,
            'posture_reference_source': 'sidecar' if named_reference else 'default',
            'logged_coverage': logged_coverage or None,
        }
        
        return df, session_info
    
    def generate_comprehensive_report(self, session_file: str, 
                                   output_dir: str = "comfort_reports") -> Dict:
        """Generate a comprehensive comfort report for a driving session."""
        
        # Load data
        df, session_info = self.load_session_data(session_file)
        
        if df.empty:
            return {"error": "No data found in session file"}
        
        # Score against the reference this session was actually recorded under.
        analyzer = DriverComfortAnalyzer(
            get_reference(session_info['posture_reference']))

        # Convert to format expected by analyzer. The SIGNED columns are passed
        # through whenever they were recorded: they are what distinguishes a
        # supported recline from a hunch, and dropping them forced the
        # analyzer's magnitude backfill to assume every lean was forward.
        signed_available = {
            'trunk_signed': 'trunk_signed' in df.columns,
            'neck_signed': 'neck_signed' in df.columns,
        }
        session_data = []
        for _, row in df.iterrows():
            angles = {
                'trunk_from_vertical': row['trunk_deg'],
                'neck_from_vertical': row['neck_deg'],
                'left_knee_angle': row['knee_L'],
                'right_knee_angle': row['knee_R'],
                'left_hip_angle': row['hip_L'],
                'right_hip_angle': row['hip_R']
            }
            for key, present in signed_available.items():
                if present and pd.notna(row[key]):
                    angles[key] = float(row[key])
            session_data.append({'angles': angles})

        session_info['signed_lean_recorded'] = bool(
            session_data and 'trunk_signed' in session_data[0]['angles'])

        # Generate comfort report
        report = self.analyzer.generate_comfort_report(session_data, session_info)
        
        # Add additional analysis
        report['detailed_analysis'] = self._analyze_posture_patterns(df)
        report['time_series_analysis'] = self._analyze_time_series(df)
        report['ergonomic_insights'] = self._generate_ergonomic_insights(df, report)
        
        return report
    
    def _analyze_posture_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze posture patterns and transitions."""
        
        analysis = {}
        
        # Posture state transitions
        if 'body_state' in df.columns:
            state_changes = df['body_state'].value_counts()
            analysis['posture_distribution'] = state_changes.to_dict()
            
            # Calculate stability (how often posture changes)
            state_transitions = (df['body_state'] != df['body_state'].shift()).sum()
            analysis['posture_stability'] = {
                'total_transitions': int(state_transitions),
                'stability_score': max(0, 100 - (state_transitions * 2))  # Lower transitions = higher stability
            }
        
        # Angle variability analysis
        angle_columns = ['trunk_deg', 'neck_deg', 'knee_L', 'knee_R', 'hip_L', 'hip_R']
        available_angles = [col for col in angle_columns if col in df.columns]
        
        if available_angles:
            angle_stats = {}
            for col in available_angles:
                angle_stats[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'variability': float(df[col].std() / abs(df[col].mean()) if df[col].mean() != 0 else 0)
                }
            analysis['angle_statistics'] = angle_stats
        
        return analysis
    
    def _analyze_time_series(self, df: pd.DataFrame) -> Dict:
        """Analyze how comfort changes over time."""
        
        analysis = {}
        
        if 'comfort_score' in df.columns:
            # Comfort trend analysis
            comfort_scores = df['comfort_score'].dropna()
            if len(comfort_scores) > 10:
                # Split into quarters
                quarter_size = len(comfort_scores) // 4
                quarters = []
                for i in range(4):
                    start_idx = i * quarter_size
                    end_idx = start_idx + quarter_size if i < 3 else len(comfort_scores)
                    quarter_avg = comfort_scores.iloc[start_idx:end_idx].mean()
                    quarters.append(float(quarter_avg))
                
                analysis['comfort_quarters'] = quarters
                
                # Trend analysis
                if quarters[3] > quarters[0] + 3:
                    analysis['comfort_trend'] = "improving"
                elif quarters[3] < quarters[0] - 3:
                    analysis['comfort_trend'] = "declining"
                else:
                    analysis['comfort_trend'] = "stable"
        
        # Time-based comfort distribution.
        # Bin on t_sec, the elapsed-seconds column. This used to bin on
        # 'timestamp', which under the canonical schema is an ISO wall-clock
        # STRING, so pd.cut raised DTypePromotionError on every canonical or
        # migrated session - the same elapsed-vs-clock confusion already fixed
        # in load_session_data. Falling back to 'timestamp' keeps pre-schema
        # files working, where that column did hold elapsed seconds.
        time_column = 't_sec' if 't_sec' in df.columns else 'timestamp'
        if 'comfort_score' in df.columns and time_column in df.columns:
            elapsed = pd.to_numeric(df[time_column], errors='coerce')
            comfort = pd.to_numeric(df['comfort_score'], errors='coerce')
            usable = elapsed.notna() & comfort.notna()
            # pd.cut needs at least two distinct values to form bins.
            if usable.sum() >= 2 and elapsed[usable].nunique() >= 2:
                groups = pd.cut(elapsed[usable], bins=10, labels=False)
                time_comfort = comfort[usable].groupby(groups).mean()
                analysis['time_based_comfort'] = {
                    int(k): float(v) for k, v in time_comfort.items()
                }
            else:
                analysis['time_based_comfort'] = {}
                analysis['time_based_comfort_note'] = (
                    f"not enough usable samples in {time_column}/comfort_score"
                )
        
        return analysis
    
    def _generate_ergonomic_insights(self, df: pd.DataFrame, comfort_report: Dict) -> Dict:
        """Generate specific ergonomic insights for seat design."""
        
        insights = {
            'seat_adjustment_recommendations': [],
            'design_considerations': [],
            'comfort_optimization': []
        }
        
        # Analyze trunk angle patterns.
        # Direction needs the SIGNED column. Reading these bounds off trunk_deg
        # made the backward-lean branch dead code - trunk_deg is a 0..90
        # MAGNITUDE and can never be below -5 - so an over-reclined seat was
        # reported as a forward lean.
        trunk_column = 'trunk_signed' if (
            'trunk_signed' in df.columns and df['trunk_signed'].notna().any()
        ) else 'trunk_deg'
        if trunk_column in df.columns:
            trunk_series = pd.to_numeric(df[trunk_column], errors='coerce').dropna()
        else:
            trunk_series = pd.Series(dtype=float)
        if not trunk_series.empty:
            trunk_mean = trunk_series.mean()
            trunk_std = trunk_series.std()
            ideal = self.DEFAULT_REFERENCE.ideal.get('trunk_signed')

            if trunk_column == 'trunk_signed' and ideal is not None:
                if trunk_mean > ideal.max:
                    insights['seat_adjustment_recommendations'].append(
                        "Consider increasing seat backrest angle to reduce forward lean"
                    )
                elif trunk_mean < ideal.min:
                    insights['seat_adjustment_recommendations'].append(
                        "Consider decreasing seat backrest angle to reduce backward lean"
                    )
            elif trunk_mean > 10:
                # Magnitude only: the direction is unrecoverable, so say so
                # instead of asserting one.
                insights['seat_adjustment_recommendations'].append(
                    "Mean trunk angle is "
                    f"{trunk_mean:.1f} deg off vertical; lean DIRECTION was not "
                    "recorded for this session, so the backrest change cannot "
                    "be signed. Re-record with --camera-position left-side or "
                    "right-side."
                )

            if trunk_std > 8:
                insights['design_considerations'].append(
                    "High trunk angle variability suggests need for better lumbar support"
                )
        
        # Analyze hip and knee angles
        for side in ['L', 'R']:
            hip_col = f'hip_{side}'
            knee_col = f'knee_{side}'
            
            if hip_col in df.columns:
                hip_mean = df[hip_col].mean()
                if hip_mean < 90:
                    insights['seat_adjustment_recommendations'].append(
                        f"Adjust {side.lower()} side seat height for better hip angle support"
                    )
            
            if knee_col in df.columns:
                knee_mean = df[knee_col].mean()
                if knee_mean < 110:
                    insights['seat_adjustment_recommendations'].append(
                        f"Adjust {side.lower()} side seat position for optimal knee angle"
                    )
        
        # Comfort optimization insights
        avg_comfort = comfort_report.get('summary', {}).get('average_comfort', 0)
        if avg_comfort < 70:
            insights['comfort_optimization'].append(
                "Overall comfort below target - consider comprehensive seat redesign"
            )
        elif avg_comfort < 85:
            insights['comfort_optimization'].append(
                "Moderate comfort - focus on specific adjustment mechanisms"
            )
        else:
            insights['comfort_optimization'].append(
                "Good comfort achieved - maintain current design principles"
            )
        
        return insights
    
    def create_visualizations(self, session_file: str, 
                            output_dir: str = "comfort_reports") -> List[str]:
        """Create comprehensive visualizations for the comfort report."""
        
        # Load data
        df, session_info = self.load_session_data(session_file)
        
        if df.empty:
            return []
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        session_id = session_info['session_id']
        created_files = []
        
        # 1. Comfort Score Timeline
        if 'comfort_score' in df.columns:
            plt.figure(figsize=(12, 6))
            plt.plot(df['timestamp'], df['comfort_score'], linewidth=2, color='blue', alpha=0.7)
            plt.axhline(y=85, color='green', linestyle='--', alpha=0.5, label='Excellent (85+)')
            plt.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='Good (70+)')
            plt.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Fair (50+)')
            plt.fill_between(df['timestamp'], df['comfort_score'], alpha=0.3, color='blue')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Comfort Score (0-100)')
            plt.title(f'Driver Comfort Score Over Time - Session {session_id}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            file_path = output_path / f"comfort_timeline_{session_id}.png"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            created_files.append(str(file_path))
        
        # 2. Posture Angle Distribution
        angle_columns = ['trunk_deg', 'neck_deg', 'knee_L', 'knee_R', 'hip_L', 'hip_R']
        available_angles = [col for col in angle_columns if col in df.columns]
        
        if available_angles:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, col in enumerate(available_angles):
                if i < len(axes):
                    axes[i].hist(df[col].dropna(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                    axes[i].axvline(df[col].mean(), color='red', linestyle='--', label=f'Mean: {df[col].mean():.1f}°')
                    axes[i].set_title(col.replace('_', ' ').title())
                    axes[i].set_xlabel('Angle (degrees)')
                    axes[i].set_ylabel('Frequency')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
            
            plt.suptitle(f'Posture Angle Distribution - Session {session_id}', fontsize=16)
            plt.tight_layout()
            
            file_path = output_path / f"angle_distribution_{session_id}.png"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            created_files.append(str(file_path))
        
        # 3. Comfort Score Distribution
        if 'comfort_score' in df.columns:
            plt.figure(figsize=(10, 6))
            comfort_bins = [0, 50, 70, 85, 100]
            comfort_labels = ['Poor', 'Fair', 'Good', 'Excellent']
            
            plt.hist(df['comfort_score'], bins=comfort_bins, alpha=0.7, color='lightcoral', edgecolor='black')
            plt.xlabel('Comfort Score')
            plt.ylabel('Frequency')
            plt.title(f'Comfort Score Distribution - Session {session_id}')
            plt.xticks(comfort_bins)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            file_path = output_path / f"comfort_distribution_{session_id}.png"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            created_files.append(str(file_path))
        
        # 4. Posture State Analysis
        if 'body_state' in df.columns:
            plt.figure(figsize=(10, 6))
            state_counts = df['body_state'].value_counts()
            colors = ['lightgreen', 'lightblue', 'lightcoral', 'lightyellow', 'lightgray']
            
            plt.pie(state_counts.values, labels=state_counts.index, autopct='%1.1f%%', 
                   colors=colors[:len(state_counts)], startangle=90)
            plt.title(f'Posture State Distribution - Session {session_id}')
            plt.axis('equal')
            
            file_path = output_path / f"posture_states_{session_id}.png"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            created_files.append(str(file_path))
        
        return created_files
    
    def generate_research_report(self, session_file: str, 
                               output_dir: str = "comfort_reports") -> str:
        """Generate a comprehensive research report for Mahindra's ergonomics team."""
        
        # Generate analysis
        report_data = self.generate_comprehensive_report(session_file, output_dir)
        
        if 'error' in report_data:
            coverage = report_data.get('measurement_coverage', {})
            excluded = report_data.get('excluded_data_points', 0)
            return (
                f"No report generated for {session_file}: {report_data['error']}. "
                f"{excluded} frame(s) were excluded. {coverage.get('note', '')} "
                "A session whose angles were recorded without lean DIRECTION "
                "cannot be scored against the driving model - see "
                "driver_model.DRIVING.required_angles."
            ).strip()
        
        # Create visualizations
        viz_files = self.create_visualizations(session_file, output_dir)
        
        # Generate report file
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        session_id = report_data['session_info']['session_id']
        report_file = output_path / f"research_report_{session_id}.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# Driver Comfort Research Report\n\n")
            f.write(f"**Session ID:** {session_id}\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Duration:** {report_data['summary']['total_duration_minutes']:.1f} minutes\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write(f"The driver achieved an **{report_data['summary']['final_comfort_category']}** comfort level ")
            f.write(f"with an average score of **{report_data['summary']['average_comfort']:.1f}/100**.\n\n")
            
            f.write("## Key Findings\n\n")
            f.write(f"- **Overall Comfort:** {report_data['summary']['average_comfort']:.1f}/100\n")
            f.write(f"- **Comfort Category:** {report_data['summary']['final_comfort_category']}\n")
            f.write(f"- **Comfort Trend:** {report_data['comfort_trend']['description']}\n")
            f.write(f"- **Data Points:** {report_data['data_points']}\n")
            coverage = report_data.get('measurement_coverage', {})
            f.write(f"- **Excluded Frames:** {report_data.get('excluded_data_points', 0)} "
                    f"(usable fraction {coverage.get('usable_fraction', 'n/a')})\n")
            missing = coverage.get('angles_most_often_missing') or {}
            if missing:
                worst = ", ".join(f"{a} x{n}" for a, n in list(missing.items())[:4])
                f.write(f"- **Most Often Not Measured:** {worst}\n")
            f.write("\n")
            
            f.write("## Detailed Analysis\n\n")
            
            # Posture Quality
            f.write("### Posture Quality Analysis\n")
            f.write(f"- **Average Posture Quality:** {report_data['detailed_scores']['posture_quality']['average']:.1f}/100\n")
            f.write(f"- **Posture Stability:** {report_data['detailed_scores']['posture_quality']['trend']}\n\n")
            
            # Ergonomic Risk
            f.write("### Ergonomic Risk Assessment\n")
            f.write(f"- **Average Risk Score:** {report_data['detailed_scores']['ergonomic_risk']['average']:.1f}/100\n")
            f.write(f"- **Risk Trend:** {report_data['detailed_scores']['ergonomic_risk']['trend']}\n\n")
            
            # Fatigue Analysis
            f.write("### Fatigue Analysis\n")
            f.write(f"- **Average Fatigue Indicator:** {report_data['detailed_scores']['fatigue_indicator']['average']:.1f}/100\n")
            f.write(f"- **Fatigue Trend:** {report_data['detailed_scores']['fatigue_indicator']['trend']}\n\n")
            
            # Recommendations
            f.write("## Recommendations for Seat Design\n\n")
            for rec in report_data['recommendations']:
                f.write(f"- {rec}\n")
            f.write("\n")
            
            # Ergonomic Insights
            if 'ergonomic_insights' in report_data:
                f.write("## Ergonomic Insights\n\n")
                
                f.write("### Seat Adjustment Recommendations\n")
                for insight in report_data['ergonomic_insights']['seat_adjustment_recommendations']:
                    f.write(f"- {insight}\n")
                f.write("\n")
                
                f.write("### Design Considerations\n")
                for insight in report_data['ergonomic_insights']['design_considerations']:
                    f.write(f"- {insight}\n")
                f.write("\n")
                
                f.write("### Comfort Optimization\n")
                for insight in report_data['ergonomic_insights']['comfort_optimization']:
                    f.write(f"- {insight}\n")
                f.write("\n")
            
            # Visualizations
            if viz_files:
                f.write("## Generated Visualizations\n\n")
                for viz_file in viz_files:
                    viz_name = Path(viz_file).name
                    f.write(f"- [{viz_name}]({viz_file})\n")
                f.write("\n")
            
            f.write("## Technical Details\n\n")
            f.write("This report was generated using the Mahindra Driver Comfort Analysis System, ")
            f.write("which employs computer vision and ergonomic research principles to evaluate ")
            f.write("driver comfort through posture analysis.\n\n")
            
            f.write("### Methodology\n")
            f.write("- Real-time pose detection using MediaPipe\n")
            f.write("- Ergonomic angle calculations based on automotive research\n")
            f.write("- Comfort scoring using weighted multi-factor analysis\n")
            f.write("- Fatigue detection through posture stability analysis\n")
        
        return str(report_file)

def main():
    parser = argparse.ArgumentParser(description="Generate driver comfort reports for Mahindra research")
    parser.add_argument("session_file", help="Session CSV file to analyze")
    parser.add_argument("--output-dir", default="comfort_reports", help="Output directory for reports")
    parser.add_argument("--report-type", choices=["basic", "comprehensive", "research"], 
                       default="research", help="Type of report to generate")
    
    args = parser.parse_args()
    
    reporter = DriverComfortReporter()
    
    try:
        if args.report_type == "basic":
            report = reporter.generate_comprehensive_report(args.session_file, args.output_dir)
            print(json.dumps(report, indent=2))
        elif args.report_type == "comprehensive":
            report = reporter.generate_comprehensive_report(args.session_file, args.output_dir)
            viz_files = reporter.create_visualizations(args.session_file, args.output_dir)
            print(f"Generated {len(viz_files)} visualizations")
            print(json.dumps(report, indent=2))
        elif args.report_type == "research":
            report_file = reporter.generate_research_report(args.session_file, args.output_dir)
            print(f"Research report generated: {report_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
