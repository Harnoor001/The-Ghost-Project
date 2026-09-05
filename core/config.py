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
    pinch_ratio_threshold: float = 0.45
    touch_cooldown_frames: int = 20

@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)

config = AppConfig()
