"""Application settings for the posture tools.

This module owns RUNTIME settings - camera, confidence, cadence, UI, alerts.
It does not own ergonomics. The threshold fields below default from
driver_model.DEFAULT_REFERENCE and exist so a profile can OVERRIDE the named
reference, not so the numbers can be written down a seventh time; before this
they were a dead copy that no monitor ever read.
"""
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

import driver_model
import posture_rules

@dataclass
class PostureConfig:
    """Configuration for posture analysis system."""
    
    # Camera settings
    camera_id: int = 0
    frame_width: int = 640
    frame_height: int = 480
    fps_target: int = 30
    
    # Detection settings
    detection_confidence: float = 0.6
    tracking_confidence: float = 0.6
    model_complexity: int = 1
    enable_segmentation: bool = False
    smooth_landmarks: bool = True
    
    # Posture analysis
    smoothing_alpha: float = 0.25
    quality_threshold_good: float = 80.0
    quality_threshold_warning: float = 60.0
    quality_threshold_poor: float = 40.0
    
    # Break reminders
    break_interval_minutes: int = 30
    max_session_duration_hours: float = 4.0
    microbreak_duration_seconds: int = 20
    
    # Logging
    logging_interval_seconds: float = 1.0
    save_visualizations: bool = True
    save_quality_scores: bool = True
    log_directory: str = "posture_logs"
    
    # UI settings
    show_angles: bool = True
    show_quality: bool = True
    show_recommendations: bool = True
    show_break_timer: bool = True
    show_posture_durations: bool = True
    ui_scale: float = 1.0
    
    # Alert settings
    alert_poor_posture: bool = True
    alert_break_time: bool = True
    alert_session_duration: bool = True
    sound_alerts: bool = False
    
    # Which driver_model.PostureReference this profile scores against:
    # "driving" for in-car seating, "desk" for upright office seating.
    # None means "whatever the monitor's own default is", which is how a
    # general-purpose profile avoids silently switching the in-car monitor to
    # the desk model.
    posture_reference: Optional[str] = None

    # Ergonomic thresholds. None means "take the reference's value" - the
    # reason these are Optional rather than numbers is that a default number
    # here would be a copy of the model, and copies drift. Set one only to
    # deviate from the reference deliberately.
    neck_forward_threshold: Optional[float] = None
    trunk_slouch_threshold: Optional[float] = None
    shoulder_elevation_threshold: Optional[float] = None
    hip_angle_target: Optional[float] = None
    hip_angle_tolerance: Optional[float] = None
    knee_angle_target: Optional[float] = None
    knee_angle_tolerance: Optional[float] = None
    
    # Analysis settings
    min_samples_for_analysis: int = 10
    outlier_detection_enabled: bool = True
    outlier_std_threshold: float = 3.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def reference(
        self,
        default: Optional["driver_model.PostureReference"] = None,
    ) -> "driver_model.PostureReference":
        """The PostureReference this profile names.

        `default` is the calling monitor's own reference, used when the profile
        does not name one.
        """
        if self.posture_reference is None:
            return default or driver_model.DEFAULT_REFERENCE
        return driver_model.get_reference(self.posture_reference)

    def to_posture_config(
        self,
        default_reference: Optional["driver_model.PostureReference"] = None,
    ) -> "posture_rules.PostureConfig":
        """The labelling thresholds, as posture_rules consumes them.

        Starts from the reference - so the angle VIEW each threshold reads
        (signed for driving, magnitude for desk) comes from the model - then
        applies whichever of this profile's overrides are actually set.
        """
        cfg = posture_rules.PostureConfig.from_reference(
            self.reference(default_reference))
        for attr, override in (
            ("trunk_slouch_deg", self.trunk_slouch_threshold),
            ("neck_forward_deg", self.neck_forward_threshold),
            ("shoulder_elev_deg", self.shoulder_elevation_threshold),
            ("hip_angle_target", self.hip_angle_target),
            ("hip_angle_tol", self.hip_angle_tolerance),
            ("knee_angle_target", self.knee_angle_target),
            ("knee_angle_tol", self.knee_angle_tolerance),
        ):
            if override is not None:
                setattr(cfg, attr, override)
        return cfg
    
    def save(self, filepath: str):
        """Save configuration to file."""
        filepath = Path(filepath)
        
        if filepath.suffix.lower() == '.json':
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        elif filepath.suffix.lower() in ['.yml', '.yaml']:
            with open(filepath, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    @classmethod
    def load(cls, filepath: str) -> 'PostureConfig':
        """Load configuration from file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        if filepath.suffix.lower() == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
        elif filepath.suffix.lower() in ['.yml', '.yaml']:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        return cls(**data)
    
    @classmethod
    def create_default(cls, filepath: str):
        """Create default configuration file."""
        config = cls()
        config.save(filepath)
        print(f"Default configuration saved to: {filepath}")
        return config

class ConfigManager:
    """Manages configuration files and provides easy access to settings."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.default_config_path = self.config_dir / "posture_config.json"
        self.user_config_path = self.config_dir / "user_config.json"
        
        # Load or create default configuration
        if not self.default_config_path.exists():
            PostureConfig.create_default(self.default_config_path)
        
        self.default_config = PostureConfig.load(self.default_config_path)
        self.user_config = self._load_user_config()
    
    def _load_user_config(self) -> PostureConfig:
        """Load user configuration or create from default."""
        if self.user_config_path.exists():
            try:
                return PostureConfig.load(self.user_config_path)
            except Exception as e:
                print(f"Warning: Could not load user config: {e}")
                print("Using default configuration")
        
        # Create user config from default
        user_config = PostureConfig()
        user_config.save(self.user_config_path)
        return user_config
    
    def get_config(self, profile: str = "user") -> PostureConfig:
        """Get configuration for specified profile."""
        if profile == "default":
            return self.default_config
        elif profile == "user":
            return self.user_config
        else:
            # Load custom profile
            profile_path = self.config_dir / f"{profile}_config.json"
            if profile_path.exists():
                return PostureConfig.load(profile_path)
            else:
                raise ValueError(f"Profile not found: {profile}")
    
    def save_user_config(self, config: PostureConfig):
        """Save user configuration."""
        config.save(self.user_config_path)
        self.user_config = config
        print(f"User configuration saved to: {self.user_config_path}")
    
    def create_profile(self, profile_name: str, base_config: Optional[PostureConfig] = None):
        """Create a new configuration profile."""
        if base_config is None:
            base_config = self.user_config
        
        profile_path = self.config_dir / f"{profile_name}_config.json"
        base_config.save(profile_path)
        print(f"Profile '{profile_name}' created: {profile_path}")
    
    def list_profiles(self) -> list:
        """List all available configuration profiles."""
        profiles = ["default", "user"]
        
        for config_file in self.config_dir.glob("*_config.json"):
            profile_name = config_file.stem.replace("_config", "")
            if profile_name not in ["default", "user"]:
                profiles.append(profile_name)
        
        return profiles
    
    def reset_to_default(self):
        """Reset user configuration to default values."""
        self.user_config = PostureConfig()
        self.save_user_config(self.user_config)
        print("User configuration reset to default values")
    
    def validate_config(self, config: PostureConfig) -> list:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check numeric ranges
        if not (0.0 <= config.detection_confidence <= 1.0):
            issues.append("detection_confidence must be between 0.0 and 1.0")
        
        if not (0.0 <= config.tracking_confidence <= 1.0):
            issues.append("tracking_confidence must be between 0.0 and 1.0")
        
        if not (0.0 <= config.smoothing_alpha <= 1.0):
            issues.append("smoothing_alpha must be between 0.0 and 1.0")
        
        if config.break_interval_minutes <= 0:
            issues.append("break_interval_minutes must be positive")
        
        if config.max_session_duration_hours <= 0:
            issues.append("max_session_duration_hours must be positive")
        
        if (config.posture_reference is not None
                and config.posture_reference not in driver_model.REFERENCES):
            issues.append(
                "posture_reference must be one of "
                f"{sorted(driver_model.REFERENCES)}, got {config.posture_reference!r}"
            )

        # Check file paths
        log_dir = Path(config.log_directory)
        if not log_dir.parent.exists():
            issues.append(f"Log directory parent does not exist: {log_dir.parent}")
        
        return issues
    
    def get_ergonomic_presets(self) -> Dict[str, PostureConfig]:
        """Get predefined ergonomic configurations for different scenarios."""
        presets = {}
        
        # Office worker preset
        office_config = PostureConfig()
        office_config.break_interval_minutes = 30
        office_config.neck_forward_threshold = 15.0
        office_config.trunk_slouch_threshold = 15.0
        office_config.hip_angle_target = 95.0
        office_config.knee_angle_target = 95.0
        presets["office"] = office_config
        
        # Gaming preset
        gaming_config = PostureConfig()
        gaming_config.break_interval_minutes = 45
        gaming_config.neck_forward_threshold = 25.0
        gaming_config.trunk_slouch_threshold = 20.0
        gaming_config.hip_angle_target = 100.0
        gaming_config.knee_angle_target = 100.0
        presets["gaming"] = gaming_config
        
        # Driving preset. The only one that selects the automotive reference,
        # and therefore the only one under which recline is not slouch.
        driving_config = PostureConfig()
        # Names the reference and overrides nothing: the automotive numbers
        # belong to driver_model.DRIVING, not to this preset.
        driving_config.posture_reference = driver_model.DRIVING.name
        driving_config.break_interval_minutes = 120
        presets["driving"] = driving_config

        # Student preset
        student_config = PostureConfig()
        student_config.break_interval_minutes = 25
        student_config.neck_forward_threshold = 18.0
        student_config.trunk_slouch_threshold = 16.0
        student_config.hip_angle_target = 90.0
        student_config.knee_angle_target = 90.0
        presets["student"] = student_config
        
        return presets
    
    def apply_preset(self, preset_name: str):
        """Apply a predefined ergonomic preset."""
        presets = self.get_ergonomic_presets()
        
        if preset_name not in presets:
            available = ", ".join(presets.keys())
            raise ValueError(f"Preset '{preset_name}' not found. Available: {available}")
        
        preset_config = presets[preset_name]
        
        # Apply preset values to user config
        for key, value in preset_config.to_dict().items():
            if hasattr(self.user_config, key):
                setattr(self.user_config, key, value)
        
        self.save_user_config(self.user_config)
        print(f"Applied '{preset_name}' preset to user configuration")

# Convenience functions
def load_config(profile: str = "user") -> PostureConfig:
    """Quick function to load configuration."""
    manager = ConfigManager()
    return manager.get_config(profile)

def save_config(config: PostureConfig, profile: str = "user"):
    """Quick function to save configuration."""
    manager = ConfigManager()
    if profile == "user":
        manager.save_user_config(config)
    else:
        config.save(manager.config_dir / f"{profile}_config.json")

def create_default_config():
    """Create default configuration files."""
    manager = ConfigManager()
    print("Default configuration created in 'config' directory")
    print("Available profiles:", manager.list_profiles())

if __name__ == "__main__":
    # Example usage
    create_default_config()
    
    # Load and modify configuration
    config = load_config()
    config.break_interval_minutes = 45
    config.alert_poor_posture = True
    
    # Save modified configuration
    save_config(config)
    
    # Apply a preset
    manager = ConfigManager()
    manager.apply_preset("office")
    
    print("Configuration setup complete!")
