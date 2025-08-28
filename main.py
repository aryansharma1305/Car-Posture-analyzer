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
from posture_scoring import score_posture

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
  python main.py config --apply-preset office

  # Quick posture scoring
  python main.py score --neck 15 --trunk 5 --hip 95 --knee 90
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Start real-time posture monitoring')
    monitor_parser.add_argument('--enhanced', action='store_true', 
                               help='Use enhanced monitor with quality scoring')
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
                              help='Apply ergonomic preset (office, gaming, student)')
    config_parser.add_argument('--reset', action='store_true', 
                              help='Reset user configuration to default')
    
    # Score command
    score_parser = subparsers.add_parser('score', help='Quick posture quality scoring')
    score_parser.add_argument('--neck', type=float, help='Neck angle from vertical')
    score_parser.add_argument('--trunk', type=float, help='Trunk angle from vertical')
    score_parser.add_argument('--hip', type=float, help='Hip angle')
    score_parser.add_argument('--knee', type=float, help='Knee angle')
    score_parser.add_argument('--shoulder', type=float, help='Shoulder elevation angle')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available data files')
    list_parser.add_argument('--recent', type=int, help='Show only recent files (days)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'monitor':
            run_monitor(args.enhanced, args.config)
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

def run_monitor(enhanced: bool, config_profile: str):
    """Run real-time posture monitoring."""
    if enhanced:
        print("Starting Enhanced Posture Monitor...")
        try:
            from posture_monitor_enhanced import main as run_enhanced_monitor
            run_enhanced_monitor()
        except ImportError as e:
            print(f"Enhanced monitor not available: {e}")
            print("Falling back to basic monitor...")
            run_basic_monitor()
    else:
        print("Starting Basic Posture Monitor...")
        run_basic_monitor()

def run_basic_monitor():
    """Run basic posture monitoring."""
    try:
        from posture_live_full import main as run_basic_monitor
        run_basic_monitor()
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
        
        if verbose:
            print("\n" + "="*60)
            print("ANALYSIS COMPLETE")
            print("="*60)
            print(report)
        else:
            print(f"Analysis complete. Report saved to: {output_file or 'console'}")
    
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
            print("Available presets: office, gaming, student")
    
    elif args.reset:
        manager = ConfigManager()
        manager.reset_to_default()
    
    else:
        print("Configuration management options:")
        print("  --create-default    Create default configuration files")
        print("  --list-profiles     List available configuration profiles")
        print("  --apply-preset      Apply ergonomic preset (office, gaming, student)")
        print("  --reset             Reset user configuration to default")

def run_score(args):
    """Quick posture quality scoring."""
    # Build angles dictionary from arguments
    angles = {}
    
    if args.neck is not None:
        angles['neck_from_vertical'] = args.neck
    if args.trunk is not None:
        angles['trunk_from_vertical'] = args.trunk
    if args.hip is not None:
        angles['left_hip_angle'] = args.hip
        angles['right_hip_angle'] = args.hip
    if args.knee is not None:
        angles['left_knee_angle'] = args.knee
        angles['right_knee_angle'] = args.knee
    if args.shoulder is not None:
        angles['left_shoulder_elev'] = args.shoulder
        angles['right_shoulder_elev'] = args.shoulder
    
    # Fill in missing angles with default values
    required_angles = [
        'neck_from_vertical', 'trunk_from_vertical',
        'left_hip_angle', 'right_hip_angle',
        'left_knee_angle', 'right_knee_angle',
        'left_shoulder_elev', 'right_shoulder_elev'
    ]
    
    for angle in required_angles:
        if angle not in angles:
            if 'hip' in angle:
                angles[angle] = 95.0  # Default seated hip angle
            elif 'knee' in angle:
                angles[angle] = 95.0  # Default seated knee angle
            elif 'shoulder' in angle:
                angles[angle] = 15.0  # Default shoulder elevation
            else:
                angles[angle] = 5.0   # Default neck/trunk angles
    
    # Calculate score
    try:
        score = score_posture(angles)
        
        print("\n" + "="*50)
        print("POSTURE QUALITY SCORE")
        print("="*50)
        print(f"Overall Score: {score.overall_score:.1f}/100")
        print(f"Risk Level: {score.risk_level}")
        print()
        
        print("Category Scores:")
        for category, cat_score in score.category_scores.items():
            print(f"  {category.replace('_', ' ').title()}: {cat_score:.1f}/100")
        print()
        
        if score.recommendations:
            print("Recommendations:")
            for i, rec in enumerate(score.recommendations, 1):
                print(f"  {i}. {rec}")
        else:
            print("✅ Excellent posture! No recommendations needed.")
        
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
