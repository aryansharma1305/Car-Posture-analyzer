#!/usr/bin/env python3
"""
Main application for Posture Analysis System
A comprehensive AI-integrated solution for real-time posture monitoring and analysis.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from posture_analytics import PostureAnalytics
from config_manager import ConfigManager, create_default_config
from session_schema import SessionMetadata
from posture_scoring import PostureQualityScorer
from driver_comfort_analyzer import DriverComfortAnalyzer
from driver_model import DESK, REFERENCES, get_reference

def main():
    """Main entry point for the posture analysis system."""
    parser = argparse.ArgumentParser(
        description="Posture Analysis System - AI-integrated posture monitoring and analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start real-time monitoring
  python main.py monitor

  # Analyze a specific session
  python main.py analyze angles_20250811_082025.csv

  # Generate summary report for last 7 days
  python main.py report --days 7

  # Create visualizations for a session
  python main.py visualize fullbody_20250811_092137.csv

  # Setup configuration
  python main.py config --create-default
  python main.py config --apply-preset driving

  # Quick posture scoring
  python main.py score --neck 15 --trunk 5 --hip 95 --knee 90

  # Score the same driver against the automotive model (signed trunk/neck)
  python main.py score --reference driving --trunk -20 --neck 5 --hip 105 --knee 115
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Start real-time posture monitoring')
    monitor_parser.add_argument('--enhanced', action='store_true', 
                               help='Use enhanced monitor with quality scoring')
    monitor_parser.add_argument('--driver-id', type=str,
                                help='Driver identifier recorded in the session sidecar')
    monitor_parser.add_argument('--seat-id', type=str,
                                help='Seat under test; required to group sessions '
                                     'for seat comparison')
    monitor_parser.add_argument('--vehicle', type=str,
                                help='Vehicle identifier recorded in the session sidecar')
    monitor_parser.add_argument('--camera-position', type=str,
                                choices=['left-side', 'right-side', 'front', 'rear', 'other'],
                                help='Where the camera sits relative to the driver. '
                                     'Lean direction is only meaningful from the side.')
    monitor_parser.add_argument('--config', type=str, default='user',
                               help='Configuration profile to use')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze posture data from CSV files')
    analyze_parser.add_argument('session_file', type=str, help='Session CSV file to analyze')
    analyze_parser.add_argument('--output', '-o', type=str, help='Output report file')
    analyze_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate summary reports')
    report_parser.add_argument('--days', type=int, default=7, help='Number of days to analyze')
    report_parser.add_argument('--output', '-o', type=str, help='Output report file')
    
    # Visualize command
    visualize_parser = subparsers.add_parser('visualize', help='Create visualizations from data')
    visualize_parser.add_argument('session_file', type=str, help='Session CSV file to visualize')
    visualize_parser.add_argument('--output-dir', type=str, help='Output directory for images')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Manage configuration')
    config_parser.add_argument('--create-default', action='store_true', 
                              help='Create default configuration files')
    config_parser.add_argument('--list-profiles', action='store_true', 
                              help='List available configuration profiles')
    config_parser.add_argument('--apply-preset', type=str, 
                              help='Apply ergonomic preset '
                                   '(driving, office, gaming, student)')
    config_parser.add_argument('--reset', action='store_true', 
                              help='Reset user configuration to default')
    
    # Score command
    score_parser = subparsers.add_parser('score', help='Quick posture quality scoring')
    score_parser.add_argument('--neck', type=float, help='Neck angle from vertical')
    score_parser.add_argument('--trunk', type=float, help='Trunk angle from vertical')
    score_parser.add_argument('--hip', type=float, help='Hip angle')
    score_parser.add_argument('--knee', type=float, help='Knee angle')
    score_parser.add_argument('--shoulder', type=float, help='Shoulder elevation angle')
    score_parser.add_argument('--reference', type=str, default=DESK.name,
                              choices=sorted(REFERENCES),
                              help='Posture model to score against. "desk" is '
                                   'upright office seating; "driving" is '
                                   'automotive seating, where 5-30 degrees of '
                                   'recline is the ideal. With "driving", pass '
                                   '--trunk/--neck as SIGNED angles '
                                   '(negative = reclined).')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available data files')
    list_parser.add_argument('--recent', type=int, help='Show only recent files (days)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'monitor':
            run_monitor(args.enhanced, args.config,
                        metadata=build_session_metadata(args))

        elif args.command == 'analyze':
            run_analyze(args.session_file, args.output, args.verbose)
        elif args.command == 'report':
            run_report(args.days, args.output)
        elif args.command == 'visualize':
            run_visualize(args.session_file, args.output_dir)
        elif args.command == 'config':
            run_config(args)
        elif args.command == 'score':
            run_score(args)
        elif args.command == 'list':
            run_list(args.recent)
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def build_session_metadata(args) -> 'SessionMetadata':
    """Assemble session attribution from the CLI flags.

    session_id/started_at/app are filled in by the monitor once it knows its
    own log path, so they are placeholders here.
    """
    camera = getattr(args, 'camera_position', None)
    # The sign of the lean angles is only anatomically meaningful from the
    # side; left-side and right-side map image-right onto opposite body
    # directions. Unknown for front/rear, so left as None rather than guessed.
    forward_is_image_right = None
    if camera == 'left-side':
        forward_is_image_right = True
    elif camera == 'right-side':
        forward_is_image_right = False

    meta = SessionMetadata(
        session_id='', started_at='', app='',
        driver_id=getattr(args, 'driver_id', None),
        seat_id=getattr(args, 'seat_id', None),
        vehicle=getattr(args, 'vehicle', None),
        camera_position=camera,
        forward_is_image_right=forward_is_image_right,
    )
    if not meta.seat_id:
        print("Note: no --seat-id given. The session will be recorded but "
              "cannot be grouped for seat comparison.")
    return meta

def run_monitor(enhanced: bool, config_profile: str, metadata=None):
    """Run real-time posture monitoring.

    config_profile selects a ConfigManager profile and it is now actually
    passed to the monitor. It used to be accepted, printed in --help, and
    dropped, so `--config office` ran the hardcoded defaults.
    """
    config = None
    try:
        config = ConfigManager().get_config(config_profile)
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Falling back to built-in defaults.")

    if enhanced:
        print("Starting Enhanced Posture Monitor...")
        try:
            from posture_monitor_enhanced import main as run_enhanced_monitor
            run_enhanced_monitor(metadata=metadata, config=config)
        except ImportError as e:
            print(f"Enhanced monitor not available: {e}")
            print("Falling back to basic monitor...")
            run_basic_monitor(metadata=metadata, config=config)
    else:
        print("Starting Basic Posture Monitor...")
        run_basic_monitor(metadata=metadata, config=config)

def run_basic_monitor(metadata=None, config=None):
    """Run basic posture monitoring."""
    try:
        from posture_live_full import main as run_full_monitor
        run_full_monitor(metadata=metadata, config=config)
    except ImportError as e:
        print(f"Basic monitor not available: {e}")
        print("Please ensure all required modules are installed")

def run_analyze(session_file: str, output_file: Optional[str], verbose: bool):
    """Analyze a specific session file."""
    print(f"Analyzing session: {session_file}")
    
    analytics = PostureAnalytics()
    
    try:
        # Generate report
        report = analytics.generate_session_report(session_file, output_file)

        # Always show the report - it is the whole point of the command.
        # --verbose only adds the banner, and -o additionally writes a file.
        if verbose:
            print("\n" + "="*60)
            print("ANALYSIS COMPLETE")
            print("="*60)
        print(report)

        if output_file:
            print(f"\nReport also written to: {output_file}")
    
    except FileNotFoundError:
        print(f"Error: Session file not found: {session_file}")
        print("Available files:")
        list_available_files()
        sys.exit(1)

def run_report(days: int, output_file: Optional[str]):
    """Generate summary report."""
    print(f"Generating summary report for last {days} days...")
    
    analytics = PostureAnalytics()
    report = analytics.generate_summary_report(days, output_file)
    
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    print(report)

def run_visualize(session_file: str, output_dir: Optional[str]):
    """Create visualizations for a session."""
    print(f"Creating visualizations for: {session_file}")
    
    analytics = PostureAnalytics()
    created_files = analytics.create_visualizations(session_file, output_dir)
    
    print(f"Created {len(created_files)} visualization files:")
    for file_path in created_files:
        print(f"  - {file_path}")

def run_config(args):
    """Handle configuration commands."""
    if args.create_default:
        create_default_config()
    
    elif args.list_profiles:
        manager = ConfigManager()
        profiles = manager.list_profiles()
        print("Available configuration profiles:")
        for profile in profiles:
            print(f"  - {profile}")
    
    elif args.apply_preset:
        manager = ConfigManager()
        try:
            manager.apply_preset(args.apply_preset)
        except ValueError as e:
            print(f"Error: {e}")
            print("Available presets: " + ", ".join(manager.get_ergonomic_presets()))
    
    elif args.reset:
        manager = ConfigManager()
        manager.reset_to_default()
    
    else:
        print("Configuration management options:")
        print("  --create-default    Create default configuration files")
        print("  --list-profiles     List available configuration profiles")
        print("  --apply-preset      Apply ergonomic preset (driving, office, gaming, student)")
        print("  --reset             Reset user configuration to default")

def run_score(args):
    """Quick posture quality scoring against a named posture reference."""
    reference = get_reference(args.reference)

    # Which view of the torso angle this reference reads. DRIVING reads the
    # SIGNED angle, because recline against a backrest is not slouch; DESK
    # reads the magnitude. Feeding a magnitude to the driving model is what
    # made a properly reclined driver look like a hunched one.
    trunk_key = reference.slouch_key
    neck_key = reference.forward_head_key

    angles = {}
    if args.neck is not None:
        angles[neck_key] = args.neck
    if args.trunk is not None:
        angles[trunk_key] = args.trunk
    if args.hip is not None:
        angles['left_hip_angle'] = args.hip
        angles['right_hip_angle'] = args.hip
    if args.knee is not None:
        angles['left_knee_angle'] = args.knee
        angles['right_knee_angle'] = args.knee
    if args.shoulder is not None:
        angles['left_shoulder_elev'] = args.shoulder
        angles['right_shoulder_elev'] = args.shoulder

    # Fill in anything the reference scores but the caller did not give, using
    # the midpoint of its own ideal range so an unspecified angle is neutral
    # rather than a hardcoded guess.
    for angle, ideal in reference.ideal.items():
        angles.setdefault(angle, (ideal.min + ideal.max) / 2.0)

    # Each model has its own scorer: the desk model scores angle magnitudes by
    # category, the driving model scores signed torso angles for comfort. They
    # are not interchangeable, so route rather than fake one with the other.
    try:
        if reference is DESK:
            score = PostureQualityScorer(reference).score_posture(angles)
            headline = f"Overall Score: {score.overall_score:.1f}/100"
            risk = f"Risk Level: {score.risk_level}"
            breakdown = {k.replace('_', ' ').title(): v
                         for k, v in score.category_scores.items()}
            recommendations = score.recommendations
        else:
            comfort = DriverComfortAnalyzer(reference).calculate_comfort_score(angles)
            headline = f"Overall Comfort: {comfort.overall_comfort:.1f}/100"
            risk = f"Category: {comfort.comfort_category}"
            breakdown = {
                "Posture Quality": comfort.posture_quality,
                "Ergonomic Risk (lower is better)": comfort.ergonomic_risk,
                "Fatigue Indicator (single sample)": comfort.fatigue_indicator,
            }
            recommendations = comfort.recommendations

        print("\n" + "="*50)
        print(f"POSTURE SCORE ({reference.name})")
        print("="*50)
        print(headline)
        print(risk)
        print()

        print("Breakdown:")
        for label, value in breakdown.items():
            print(f"  {label}: {value:.1f}/100")
        print()

        if recommendations:
            print("Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        else:
            print("Excellent posture! No recommendations needed.")

        print("="*50)

    except Exception as e:
        print(f"Error calculating score: {e}")

def run_list(recent_days: Optional[int]):
    """List available data files."""
    list_available_files(recent_days)

def list_available_files(recent_days: Optional[int] = None):
    """List available posture data files."""
    log_dir = Path("posture_logs")
    
    if not log_dir.exists():
        print("No posture logs directory found.")
        return
    
    csv_files = list(log_dir.glob("*.csv"))
    
    if not csv_files:
        print("No CSV data files found.")
        return
    
    # Filter by recent files if requested
    if recent_days:
        from datetime import datetime, timedelta
        cutoff_date = datetime.now() - timedelta(days=recent_days)
        
        recent_files = []
        for file_path in csv_files:
            try:
                # Extract date from filename
                parts = file_path.stem.split('_')
                if len(parts) >= 3:
                    date_str = f"{parts[1]}_{parts[2]}"
                    file_date = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                    if file_date >= cutoff_date:
                        recent_files.append(file_path)
            except:
                continue
        
        csv_files = recent_files
    
    print(f"Available data files ({len(csv_files)} total):")
    for file_path in sorted(csv_files):
        file_size = file_path.stat().st_size
        print(f"  - {file_path.name} ({file_size} bytes)")
    
    # Show summary by file type
    file_types = {}
    for file_path in csv_files:
        file_type = file_path.stem.split('_')[0] if '_' in file_path.stem else 'unknown'
        file_types[file_type] = file_types.get(file_type, 0) + 1
    
    print(f"\nFile type summary:")
    for file_type, count in sorted(file_types.items()):
        print(f"  {file_type}: {count} files")

if __name__ == "__main__":
    main()
