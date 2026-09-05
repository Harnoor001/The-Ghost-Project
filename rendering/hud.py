import cv2
import numpy as np
import time
import collections

class HUD:
    FONT  = cv2.FONT_HERSHEY_SIMPLEX
    MONO  = cv2.FONT_HERSHEY_DUPLEX
    CYAN  = (0, 230, 255)
    WHITE = (220, 220, 220)
    GREEN = (0, 255, 140)
    DIM   = (80,  80,  80)
    AMBER = (0, 165, 255)   # BGR amber for stale warning

    def __init__(self, h, w, dev_str):
        self.h, self.w = h, w
        self.dev_str   = dev_str
        self._fps_buf  = collections.deque(maxlen=30)
        self._last_t   = time.time()

    def tick(self):
        now = time.time()
        self._fps_buf.append(1.0 / max(now - self._last_t, 1e-6))
        self._last_t = now

    @property
    def fps(self):
        return np.mean(self._fps_buf) if self._fps_buf else 0.0

    def draw(self, frame, alpha: float, is_active: bool, hand_count: int, touching: bool,
             background_valid: bool = True):
        h, w = self.h, self.w

        frame[0:48, :] = (frame[0:48, :] * 0.35).astype(np.uint8)
        cv2.putText(frame, "Ghost / Invisibility Mode  by Tuba",
                    (16, 30), self.FONT, 0.60, self.CYAN, 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS {self.fps:05.1f}",
                    (w-160, 30), self.FONT, 0.52, self.GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, f"[ {self.dev_str} ]",
                    (w-310, 30), self.FONT, 0.42, self.WHITE, 1, cv2.LINE_AA)

        if alpha > 0.05:
            pulse = int(abs(np.sin(time.time() * 5)) * 80 + 175)
            cv2.circle(frame, (w-20, 12), 7, (0, pulse, 255), -1)
            cv2.putText(frame, "GHOST ACTIVE",
                        (w-185, 16), self.FONT, 0.38, (0,230,255), 1, cv2.LINE_AA)
        elif is_active:
            cv2.circle(frame, (w-20, 12), 7, (0,255,80), -1)
            cv2.putText(frame, "PORTAL READY  — PINCH TO VANISH",
                        (w-340, 16), self.FONT, 0.38, (0,255,80), 1, cv2.LINE_AA)

        frame[h-46:h, :] = (frame[h-46:h, :] * 0.35).astype(np.uint8)
        bx, by, bw2 = 16, h-28, 160
        cv2.rectangle(frame, (bx, by), (bx+bw2, by+10), (40,40,40), -1)
        filled = int(bw2 * alpha)
        if filled > 0:
            cv2.rectangle(frame, (bx, by), (bx+filled, by+10),
                          (int(80*(1-alpha)), int(200*alpha), 255), -1)
        cv2.putText(frame, f"GHOST {int(alpha*100):3d}%",
                    (bx, by-4), self.FONT, 0.40, self.CYAN, 1, cv2.LINE_AA)
        cv2.putText(frame, f"HANDS  {hand_count}",
                    (bx+200, by+8), self.FONT, 0.40, self.WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, time.strftime("%H:%M:%S"),
                    (w-110, by+8), self.MONO, 0.42, self.DIM, 1, cv2.LINE_AA)

        # ── Stale background warning ─────────────────────────────────────
        if not background_valid:
            warn = "Background stale — press R to recalibrate"
            tw = cv2.getTextSize(warn, self.FONT, 0.42, 1)[0][0]
            cv2.putText(frame, warn, ((w - tw) // 2, h - 52),
                        self.FONT, 0.42, self.AMBER, 1, cv2.LINE_AA)
        elif touching:
            tw = cv2.getTextSize("PINCH DETECTED!", self.FONT, 0.55, 2)[0][0]
            cv2.putText(frame, "PINCH DETECTED!", ((w-tw)//2, h-52),
                        self.FONT, 0.55, (0,255,120), 2, cv2.LINE_AA)
        elif alpha < 0.3 and not is_active:
            if hand_count == 0:
                hint = "SHOW BOTH HANDS  ->  SPREAD APART  ->  PINCH TO VANISH"
            else:
                hint = "PINCH THUMB + INDEX TOGETHER ON EITHER HAND"
            tw = cv2.getTextSize(hint, self.FONT, 0.40, 1)[0][0]
            cv2.putText(frame, hint, ((w-tw)//2, h-52),
                        self.FONT, 0.40, (120,120,120), 1, cv2.LINE_AA)

        return frame
