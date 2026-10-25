import cv2
import numpy as np

from core.state import FrameContext


class PortalRenderer:
    def __init__(self):
        self._scan_offset = 0

    def render(
        self,
        frame: np.ndarray,
        ctx: FrameContext,
        alpha: float,
        is_active: bool,
        tl,
        br,
        invisible: bool,
    ):
        h, w = frame.shape[:2]

        if alpha > 0.05 and ctx.background is not None and ctx.segmentation_mask is not None:
            self._draw_scanlines(frame, ctx.segmentation_mask, 0, 0, w, h, alpha)

        if is_active and tl and br:
            a = max(0.4, 1.0 - alpha * 0.5)
            x1, y1 = tl
            x2, y2 = br

            for thick, blend_str, color in [
                (11, 0.12, (0, int(180 * a), int(255 * a))),
                (6, 0.30, (0, int(210 * a), int(255 * a))),
                (2, 0.70, (0, int(230 * a), int(255 * a))),
                (1, 1.00, (255, 255, 255)),
            ]:
                ov = frame.copy()
                cv2.rectangle(ov, (x1, y1), (x2, y2), color, thick)
                cv2.addWeighted(ov, blend_str * a, frame, 1 - blend_str * a, 0, frame)

            corner_length = max(25, min(50, (x2 - x1) // 5, (y2 - y1) // 5))
            ccol = (0, int(240 * a), int(255 * a))
            for (cx, cy), (ddx1, ddy1), (ddx2, ddy2) in [
                ((x1, y1), (corner_length, 0), (0, corner_length)),
                ((x2, y1), (-corner_length, 0), (0, corner_length)),
                ((x1, y2), (corner_length, 0), (0, -corner_length)),
                ((x2, y2), (-corner_length, 0), (0, -corner_length)),
            ]:
                cv2.line(frame, (cx, cy), (cx + ddx1, cy + ddy1), ccol, 3, cv2.LINE_AA)
                cv2.line(frame, (cx, cy), (cx + ddx2, cy + ddy2), ccol, 3, cv2.LINE_AA)

            msg = (
                "PINCH THUMB + INDEX  ->  VANISH" if not invisible else "PINCH AGAIN  ->  REAPPEAR"
            )
            tw = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)[0][0]
            mx = (x1 + x2) // 2 - tw // 2
            my = (y1 + y2) // 2
            cv2.putText(
                frame,
                msg,
                (mx + 1, my + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                msg,
                (mx, my),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, int(230 * a), int(255 * a)),
                1,
                cv2.LINE_AA,
            )

        self._scan_offset = (self._scan_offset + 3) % max(1, h)

    def _draw_scanlines(self, frame, seg_mask, x1, y1, x2, y2, alpha):
        scan_col = np.array([0, 160, 220], dtype=np.float32)
        roi_h = y2 - y1
        if roi_h <= 0:
            return
        rows = (np.arange(0, roi_h, 6) + self._scan_offset) % roi_h + y1
        rows = rows[rows < y2].astype(int)
        if rows.size == 0:
            return
        sm = seg_mask[rows, x1:x2]
        strip = frame[rows, x1:x2].astype(np.float32)
        t = 0.10 * alpha
        strip = strip * (1 - t * sm[:, :, None]) + scan_col * t * sm[:, :, None]
        frame[rows, x1:x2] = strip.astype(np.uint8)
