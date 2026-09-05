import cv2
import numpy as np
from core.config import SegmentationConfig

class SegmentationEngine:
    def __init__(self, config: SegmentationConfig):
        import mediapipe as mp
        self.config = config
        self.seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        self._prev_mask = None
        self._frame_idx = 0

    def get_mask(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        self._frame_idx += 1
        if self._frame_idx % 2 == 0 and self._prev_mask is not None:
            return self._prev_mask
            
        small = cv2.resize(frame_bgr, (self.config.width, self.config.height), interpolation=cv2.INTER_AREA)
        res = self.seg.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        
        if res.segmentation_mask is None:
            return self._prev_mask if self._prev_mask is not None else np.zeros((h, w), dtype=np.float32)
            
        mask = cv2.resize(res.segmentation_mask, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        if self._prev_mask is not None:
            mask = 0.6 * mask + 0.4 * self._prev_mask
            
        _, hard = cv2.threshold(mask, self.config.threshold, 1.0, cv2.THRESH_BINARY)
        hard8 = (hard * 255).astype(np.uint8)
        kernel = np.ones((15, 15), np.uint8)
        hard8 = cv2.morphologyEx(hard8, cv2.MORPH_CLOSE, kernel)
        hard8 = cv2.dilate(hard8, kernel, iterations=1)
        hard = hard8.astype(np.float32) / 255.0
        mask = cv2.GaussianBlur(hard, (9, 9), 0)
        mask = np.clip(mask * 1.3, 0, 1).astype(np.float32)
        
        self._prev_mask = mask
        return mask
        
    def close(self):
        if hasattr(self, 'seg') and self.seg is not None:
            self.seg.close()
