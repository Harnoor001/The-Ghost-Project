from dataclasses import dataclass, field

@dataclass
class CameraConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    calibration_seconds: int = 3

@dataclass
class SegmentationConfig:
    width: int = 320
    height: int = 180
    threshold: float = 0.25

@dataclass
class GestureConfig:
    hand_width: int = 480
    pinch_enter_threshold: float = 0.40
    pinch_exit_threshold: float = 0.50
    pinch_confirmation_sec: float = 0.15
    gesture_cooldown_sec: float = 0.50

@dataclass
class BackgroundConfig:
    """Configuration for the streaming background model."""
    # Continuous EMA update after calibration
    update_enabled: bool = True
    update_rate: float = 0.02            # EMA alpha (~30 FPS → slow drift)
    foreground_protection: bool = True   # Skip EMA update on fg pixels
    stale_threshold_sec: float = 5.0     # Seconds without bg update → stale

    # Calibration behaviour
    calibration_min_frames: int = 15     # Frames before mask-based accumulation starts
    calibration_update_interval: int = 6 # (kept for API compat; not used by Phase-2 model)

@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)

config = AppConfig()
