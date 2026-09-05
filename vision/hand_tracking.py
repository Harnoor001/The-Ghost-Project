import cv2
from core.config import GestureConfig
from core.state import FrameContext

class HandTracker:
    def __init__(self, config: GestureConfig):
        import mediapipe as mp
        self.config = config
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4,
        )

    def process(self, frame_bgr, ctx: FrameContext):
        h, w = frame_bgr.shape[:2]
        sw = min(w, self.config.hand_width)
        sh = int(h * sw / w)
        small = cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        results = self.hands.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        
        ctx.hand_count = 0
        ctx.tips = None
        ctx.fingers_touching = False
        ctx.all_points = []
        
        if not results or not results.multi_hand_landmarks:
            return

        ctx.hand_count = len(results.multi_hand_landmarks)
        index_tips = []
        pinch_ratios = []

        for lms in results.multi_hand_landmarks:
            lm = lms.landmark
            thumb_tip = lm[4]
            index_tip = lm[8]
            wrist = lm[0]
            mid_mcp = lm[9]

            index_tips.append((int(index_tip.x * w), int(index_tip.y * h)))
            ctx.all_points.append([(int(p.x * w), int(p.y * h)) for p in lm])

            palm_size = ((wrist.x - mid_mcp.x)**2 + (wrist.y - mid_mcp.y)**2)**0.5
            pinch_dist = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5

            if palm_size > 1e-6:
                ratio = pinch_dist / palm_size
                pinch_ratios.append(ratio)
            else:
                pinch_ratios.append(999.0)

        ctx.pinch_ratios = pinch_ratios

        if len(index_tips) >= 2:
            ctx.tips = (index_tips[0], index_tips[1])
            
    def close(self):
        if hasattr(self, 'hands') and self.hands is not None:
            self.hands.close()
