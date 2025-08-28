#!/usr/bin/env python3
"""
Demo script for the Posture Analysis System
Demonstrates key features and provides interactive examples
"""

import time
import sys
from pathlib import Path

def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_section(title: str):
    """Print a formatted section"""
    print(f"\n--- {title} ---")

def demo_posture_scoring():
    """Demonstrate posture scoring functionality"""
    print_section("Posture Quality Scoring Demo")
    
    try:
        from posture_scoring import score_posture
        
        # Example 1: Good posture
        print("Example 1: Good Posture")
        good_angles = {
            "neck_from_vertical": 5.0,
            "trunk_from_vertical": 3.0,
            "left_hip_angle": 95.0,
            "right_hip_angle": 95.0,
            "left_knee_angle": 90.0,
            "right_knee_angle": 90.0,
            "left_shoulder_elev": 10.0,
            "right_shoulder_elev": 10.0
        }
        
        score = score_posture(good_angles)
        print(f"Overall Score: {score.overall_score:.1f}/100")
        print(f"Risk Level: {score.risk_level}")
        print(f"Recommendations: {', '.join(score.recommendations[:2])}")
        
        # Example 2: Poor posture
        print("\nExample 2: Poor Posture")
        poor_angles = {
            "neck_from_vertical": 35.0,
            "trunk_from_vertical": 25.0,
            "left_hip_angle": 120.0,
            "right_hip_angle": 120.0,
            "left_knee_angle": 60.0,
            "right_knee_angle": 60.0,
            "left_shoulder_elev": 45.0,
            "right_shoulder_elev": 45.0
        }
        
        score = score_posture(poor_angles)
        print(f"Overall Score: {score.overall_score:.1f}/100")
        print(f"Risk Level: {score.risk_level}")
        print(f"Recommendations: {', '.join(score.recommendations[:2])}")
        
    except ImportError as e:
        print(f"Could not import posture_scoring: {e}")
        print("Make sure to install dependencies: pip install -r requirements.txt")

def demo_analytics():
    """Demonstrate analytics functionality"""
    print_section("Posture Analytics Demo")
    
    try:
        from posture_analytics import PostureAnalytics
        
        analytics = PostureAnalytics()
        
        # List available data files
        print("Available data files:")
        data_files = list(Path("posture_logs").glob("*.csv"))
        if data_files:
            for file in data_files[:5]:  # Show first 5 files
                print(f"  - {file.name}")
        else:
            print("  No data files found in posture_logs/")
        
        # Try to analyze a sample file if available
        if data_files:
            sample_file = data_files[0].name
            print(f"\nAnalyzing sample file: {sample_file}")
            try:
                analysis = analytics.analyze_session(sample_file)
                print(f"Session duration: {analysis.get('duration_minutes', 'N/A'):.1f} minutes")
                print(f"Posture changes: {analysis.get('posture_changes', 'N/A')}")
            except Exception as e:
                print(f"Analysis failed: {e}")
        
    except ImportError as e:
        print(f"Could not import posture_analytics: {e}")
        print("Make sure to install dependencies: pip install -r requirements.txt")

def demo_configuration():
    """Demonstrate configuration management"""
    print_section("Configuration Management Demo")
    
    try:
        from config_manager import ConfigManager
        
        manager = ConfigManager()
        
        # List available profiles
        profiles = manager.list_profiles()
        print(f"Available profiles: {', '.join(profiles)}")
        
        # List available presets
        presets = manager.get_ergonomic_presets()
        print(f"Available presets: {', '.join(presets.keys())}")
        
        # Show current user config
        user_config = manager.get_config("user")
        print(f"\nCurrent user configuration:")
        print(f"  Camera ID: {user_config.camera_id}")
        print(f"  Detection confidence: {user_config.detection_confidence}")
        print(f"  Break interval: {user_config.break_interval_minutes} minutes")
        
    except ImportError as e:
        print(f"Could not import config_manager: {e}")
        print("Make sure to install dependencies: pip install -r requirements.txt")

def demo_quick_analysis():
    """Demonstrate basic data analysis using pandas"""
    print_section("Quick Data Analysis Demo")
    
    try:
        import pandas as pd
        
        # Look for CSV files in posture_logs
        csv_files = list(Path("posture_logs").glob("*.csv"))
        if csv_files:
            # Use the most recent file
            latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
            print(f"Analyzing: {latest_file.name}")
            
            df = pd.read_csv(latest_file)
            print(f"Data shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            if 't_sec' in df.columns:
                duration = df['t_sec'].max() - df['t_sec'].min()
                print(f"Session duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            
            if 'label' in df.columns:
                posture_counts = df['label'].value_counts()
                print(f"Posture distribution:")
                for posture, count in posture_counts.items():
                    print(f"  {posture}: {count}")
                    
        else:
            print("No CSV files found in posture_logs/")
            
    except ImportError as e:
        print(f"Could not import pandas: {e}")
        print("Make sure to install dependencies: pip install -r requirements.txt")

def demo_system_info():
    """Show system information and project status"""
    print_section("System Information")
    
    print(f"Python version: {sys.version}")
    
    # Check key dependencies
    dependencies = [
        ('opencv-python', 'cv2'),
        ('mediapipe', 'mediapipe'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn')
    ]
    
    print("\nDependency status:")
    for package, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (not installed)")
    
    # Check project files
    print("\nProject files:")
    required_files = [
        'main.py', 'posture_scoring.py', 'posture_analytics.py',
        'posture_monitor_enhanced.py', 'config_manager.py'
    ]
    
    for file in required_files:
        if Path(file).exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (missing)")

def interactive_demo():
    """Interactive demo for posture scoring"""
    print_section("Interactive Posture Scoring")
    
    try:
        from posture_scoring import score_posture
        
        print("Enter your posture angles to get a real-time score!")
        print("(Press Enter to use default values)")
        
        # Get user input or use defaults
        neck_angle = input("Neck angle from vertical (0-90°): ").strip()
        trunk_angle = input("Trunk angle from vertical (0-90°): ").strip()
        
        if not neck_angle:
            neck_angle = 15.0
        else:
            neck_angle = float(neck_angle)
            
        if not trunk_angle:
            trunk_angle = 8.0
        else:
            trunk_angle = float(trunk_angle)
        
        # Create angles dictionary with default values for others
        angles = {
            "neck_from_vertical": neck_angle,
            "trunk_from_vertical": trunk_angle,
            "left_hip_angle": 95.0,
            "right_hip_angle": 95.0,
            "left_knee_angle": 90.0,
            "right_knee_angle": 90.0,
            "left_shoulder_elev": 15.0,
            "right_shoulder_elev": 15.0
        }
        
        print(f"\nAnalyzing posture with angles: {angles}")
        score = score_posture(angles)
        
        print(f"\nResults:")
        print(f"  Overall Score: {score.overall_score:.1f}/100")
        print(f"  Risk Level: {score.risk_level}")
        print(f"  Recommendations:")
        for rec in score.recommendations[:3]:
            print(f"    • {rec}")
            
    except ImportError as e:
        print(f"Could not import posture_scoring: {e}")
        print("Make sure to install dependencies: pip install -r requirements.txt")
    except ValueError as e:
        print(f"Invalid input: {e}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Run all demo functions"""
    print_header("Posture Analysis System - Demo")
    
    print("This demo showcases the key features of the Posture Analysis System.")
    print("Make sure you have installed all dependencies: pip install -r requirements.txt")
    
    # Run demos
    demo_system_info()
    demo_posture_scoring()
    demo_analytics()
    demo_configuration()
    demo_quick_analysis()
    
    # Interactive demo
    try:
        interactive_demo()
    except KeyboardInterrupt:
        print("\nInteractive demo interrupted.")
    
    print_header("Demo Complete")
    print("To run the full system:")
    print("  python main.py monitor --enhanced")
    print("\nFor more options:")
    print("  python main.py --help")

if __name__ == "__main__":
    main()
