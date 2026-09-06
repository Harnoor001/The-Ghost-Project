import sys
import time
import cv2
import numpy as np

from core.config import config
from core.state import FrameContext
from core.camera import Camera
from vision.segmentation import SegmentationEngine
from vision.hand_tracking import HandTracker
from vision.background import BackgroundModel
from interaction.gestures import GestureDetector, GestureEvent
from effects.invisibility import InvisibilityEffect
from rendering.compositor import Compositor

BANNER = """
Ghost / Invisibility Mode


  1. Stand still 3s  ->  background captured
  2. Show BOTH hands spread apart  ->  yellow box appears
  3. PINCH thumb + index together  ->  YOU VANISH
  4. Pinch again  ->  reappear
  R = recalibrate  |  S = screenshot  |  Q = quit
"""
WINDOW = "Ghost / Invisibility Mode"

def run_calibration(
    cap: Camera,
    bg_model: BackgroundModel,
    seg: SegmentationEngine,
    w: int,
    h: int,
    seconds: int,
):
    print(f"[CAL] Stand still {seconds}s ...")
    start = time.time()
    while time.time() - start < seconds:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_f32 = frame.astype(np.float32)
        # Get segmentation mask for foreground exclusion.
        # For the first calibration_min_frames frames the model ignores the
        # mask (warm-up period), but we still pass it so the BackgroundModel
        # can decide internally.
        mask = seg.get_mask(frame)
        bg_model.update(frame_f32, foreground_mask=mask, allow_update=False)
        rem = int(seconds - (time.time() - start)) + 1
        disp = (frame * 0.5).astype(np.uint8)
        msg = "STAND STILL  —  CALIBRATING BACKGROUND"
        tw = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
        cv2.putText(disp, msg, ((w - tw) // 2, h // 2 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2, cv2.LINE_AA)
        tw2 = cv2.getTextSize(str(rem), cv2.FONT_HERSHEY_SIMPLEX, 4.0, 5)[0][0]
        cv2.putText(disp, str(rem), ((w - tw2) // 2, h // 2 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 120), 5, cv2.LINE_AA)
        cv2.imshow(WINDOW, disp)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            return False
    print("[CAL] Done\n")
    return True

def main():
    print(BANNER)
    try:
        cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        if cam_idx < 0:
            raise ValueError
    except ValueError:
        print("[ERROR] Camera index must be a non-negative integer.")
        sys.exit(2)

    try:
        cap = Camera(cam_idx, config.camera)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    h, w = cap.actual_h, cap.actual_w
    print("[INIT] Loading models ...")

    seg = SegmentationEngine(config.segmentation)
    tracker = HandTracker(config.gesture)
    bg_cfg = config.background
    bg_model = BackgroundModel(
        h, w,
        calibration_min_frames=bg_cfg.calibration_min_frames,
        calibration_update_interval=bg_cfg.calibration_update_interval,
        update_enabled=bg_cfg.update_enabled,
        update_rate=bg_cfg.update_rate,
        foreground_protection=bg_cfg.foreground_protection,
        stale_threshold_sec=bg_cfg.stale_threshold_sec,
    )
    
    detector = GestureDetector(config.gesture)
    effect = InvisibilityEffect()
    compositor = Compositor(h, w, cap.device_name)

    print("[INIT] Ready!\n")
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, min(w, 1280), min(h, 720))

    try:
        if not run_calibration(cap, bg_model, seg, w, h, config.camera.calibration_seconds):
            return

        bg_model.finalize()
        sc_idx = 0
        ctx = FrameContext(frame=np.zeros((h, w, 3), dtype=np.uint8))

        app_invisible = False

        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            compositor.tick()

            # ── Vision ────────────────────────────────────────────────────
            ctx.frame = frame
            ctx.segmentation_mask = seg.get_mask(frame)

            # EMA update of background (only meaningful pixels; fg is protected)
            frame_f32 = frame.astype(np.float32)
            bg_model.update(frame_f32, foreground_mask=ctx.segmentation_mask, allow_update=True)

            # Safety guard: only expose background when the model is valid
            ctx.background_valid = bg_model.is_valid()
            ctx.background = bg_model.get() if ctx.background_valid else None
            if not ctx.background_valid:
                # Do not leave the UI in ghost mode when its reference
                # background is unavailable or stale.
                app_invisible = False

            # Vision processing
            tracker.process(frame, ctx)

            # Interaction
            events = detector.update(ctx)

            if GestureEvent.PINCH_CONFIRMED in events:
                app_invisible = not app_invisible

            # Effects
            effect.update_alpha(app_invisible)
            frame = effect.apply(frame, ctx)

            # Rendering
            out = compositor.render(frame, ctx, effect.alpha, detector.zone.is_active, detector.zone.tl, detector.zone.br, app_invisible)

            cv2.imshow(WINDOW, out)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                fn = f"screenshot_{sc_idx:04d}.png"
                cv2.imwrite(fn, out)
                print(f"[SCREENSHOT] {fn}")
                sc_idx += 1
            elif key == ord('r'):
                bg_model.reset()
                detector.reset()
                app_invisible = False
                effect.alpha = 0.0
                if not run_calibration(cap, bg_model, seg, w, h, config.camera.calibration_seconds):
                    break
                bg_model.finalize()

    finally:
        cap.release()
        seg.close()
        tracker.close()
        cv2.destroyAllWindows()
        print("\n[DONE]")

if __name__ == "__main__":
    main()
