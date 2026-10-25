from dataclasses import dataclass, field


@dataclass
class CameraConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    calibration_seconds: int = 3

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("Camera dimensions and fps must be positive")
        if self.calibration_seconds <= 0:
            raise ValueError("calibration_seconds must be positive")


@dataclass
class SegmentationConfig:
    width: int = 320
    height: int = 180
    threshold: float = 0.25

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Segmentation dimensions must be positive")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Segmentation threshold must be between 0 and 1")


@dataclass
class GestureConfig:
    hand_width: int = 480
    pinch_enter_threshold: float = 0.40
    pinch_exit_threshold: float = 0.50
    pinch_confirmation_sec: float = 0.15
    gesture_cooldown_sec: float = 0.50

    def __post_init__(self):
        if self.hand_width <= 0:
            raise ValueError("hand_width must be positive")
        if not 0.0 <= self.pinch_enter_threshold <= self.pinch_exit_threshold:
            raise ValueError("Pinch thresholds must satisfy 0 <= enter <= exit")
        if self.pinch_confirmation_sec < 0 or self.gesture_cooldown_sec < 0:
            raise ValueError("Gesture durations cannot be negative")


@dataclass
class BackgroundConfig:
    """Configuration for the streaming background model."""

    # Continuous EMA update after calibration
    update_enabled: bool = True
    update_rate: float = 0.02  # EMA alpha (~30 FPS → slow drift)
    foreground_protection: bool = True  # Skip EMA update on fg pixels
    stale_threshold_sec: float = 5.0  # Seconds without bg update → stale

    # Calibration behaviour
    calibration_min_frames: int = 15  # Frames before mask-based accumulation starts
    calibration_update_interval: int = 6  # (kept for API compat; not used by Phase-2 model)

    def __post_init__(self):
        if not 0.0 <= self.update_rate <= 1.0:
            raise ValueError("update_rate must be between 0 and 1")
        if self.calibration_min_frames < 0 or self.calibration_update_interval <= 0:
            raise ValueError("Calibration frame settings are invalid")
        if self.stale_threshold_sec < 0:
            raise ValueError("stale_threshold_sec cannot be negative")


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)


config = AppConfig()
