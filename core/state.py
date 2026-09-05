from dataclasses import dataclass, field
import numpy as np

@dataclass
class FrameContext:
    frame: np.ndarray
    background: np.ndarray | None = None
    segmentation_mask: np.ndarray | None = None
    hand_count: int = 0
    pinch_ratios: list = field(default_factory=list)
    fingers_touching: bool = False
    tips: tuple | None = None
    all_points: list = field(default_factory=list)
