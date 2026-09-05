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
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)

config = AppConfig()
