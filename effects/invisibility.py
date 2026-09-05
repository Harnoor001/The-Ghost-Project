import numpy as np
from core.state import FrameContext

class InvisibilityEffect:
    def __init__(self):
        self.alpha = 0.0

    def update_alpha(self, invisible: bool):
        target = 1.0 if invisible else 0.0
        speed = 0.10
        if self.alpha < target:
            self.alpha = min(target, self.alpha + speed)
        else:
            self.alpha = max(target, self.alpha - speed)

    def apply(self, frame: np.ndarray, ctx: FrameContext) -> np.ndarray:
        if self.alpha > 0.01 and ctx.background is not None and ctx.segmentation_mask is not None:
            roi_f = frame.astype(np.float32)
            roi_bg = ctx.background
            
            eff_mask = ctx.segmentation_mask if self.alpha < 0.97 else np.minimum(ctx.segmentation_mask * 1.6, 1.0)
            roi_mask = eff_mask[:, :, np.newaxis]
            
            blend = roi_f * (1 - roi_mask * self.alpha) + roi_bg * (roi_mask * self.alpha)
            np.clip(blend, 0, 255, out=blend)
            frame[:, :] = blend.astype(np.uint8)
            
        return frame
