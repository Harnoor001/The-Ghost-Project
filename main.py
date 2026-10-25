"""Executable entry point for Ghost / Invisibility Mode."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.camera import Camera
    from vision.background import BackgroundModel
    from vision.segmentation import SegmentationEngine


APP_VERSION = "1.0.0"
WINDOW = "Ghost / Invisibility Mode"
LOG = logging.getLogger("ghost")

BANNER = """
Ghost / Invisibility Mode

  1. Keep the camera still and clear the scene for 3 seconds.
  2. Spread both hands to display the interaction area.
  3. Pinch thumb + index finger to toggle invisibility.
  R = recalibrate  |  S = screenshot  |  Q / Esc = quit
"""


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI without importing optional camera dependencies."""
    parser = argparse.ArgumentParser(
        prog="ghost", description="Run the Ghost real-time invisibility camera application."
    )
    parser.add_argument(
        "camera_index",
        nargs="?",
        type=int,
        default=None,
        help="Camera index (legacy positional form, e.g. ghost 1).",
    )
    parser.add_argument("--camera", type=int, default=None, help="Camera index (default: 0).")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested camera frame rate.")
    parser.add_argument(
        "--calibration-seconds",
        type=int,
        default=3,
        help="Background calibration duration in seconds.",
    )
    parser.add_argument(
        "--screenshot-dir",
        type=Path,
        default=Path("screenshots"),
        help="Directory for saved screenshots (default: screenshots).",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Console logging level.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {APP_VERSION}")
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.camera is not None and args.camera_index is not None:
        parser.error("use either the positional camera index or --camera, not both")
    camera_index = args.camera if args.camera is not None else (args.camera_index or 0)
    if camera_index < 0:
        parser.error("camera index must be a non-negative integer")
    for name in ("width", "height", "fps", "calibration_seconds"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return camera_index


def _load_runtime_dependencies() -> bool:
    """Import runtime dependencies and print an actionable error if absent."""
    try:
        import cv2  # noqa: F401
        import mediapipe  # noqa: F401
        import numpy  # noqa: F401
    except ModuleNotFoundError as exc:
        print(
            f"[ERROR] Missing runtime dependency: {exc.name}. "
            "Create a Python 3.10–3.12 virtual environment and run: python -m pip install .",
            file=sys.stderr,
        )
        return False
    return True


def run_calibration(
    cap: Camera,
    bg_model: BackgroundModel,
    seg: SegmentationEngine,
    w: int,
    h: int,
    seconds: int,
    cv2,
) -> bool:
    """Capture a clean reference background; return False on cancel/failure."""
    LOG.info("Calibrating background for %s seconds", seconds)
    start = time.monotonic()
    failed_reads = 0
    while time.monotonic() - start < seconds:
        ret, frame = cap.read()
        if not ret or frame is None:
            failed_reads += 1
            if failed_reads >= 30:
                LOG.error("Camera stopped returning frames during calibration")
                return False
            time.sleep(0.02)
            continue
        failed_reads = 0
        mask = seg.get_mask(frame)
        bg_model.update(frame.astype("float32"), foreground_mask=mask, allow_update=False)
        remaining = max(1, int(seconds - (time.monotonic() - start)) + 1)
        display = (frame * 0.5).astype("uint8")
        message = "KEEP THE SCENE CLEAR — CALIBRATING BACKGROUND"
        text_width = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
        cv2.putText(
            display,
            message,
            ((w - text_width) // 2, h // 2 - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 230, 255),
            2,
            cv2.LINE_AA,
        )
        number_width = cv2.getTextSize(str(remaining), cv2.FONT_HERSHEY_SIMPLEX, 4.0, 5)[0][0]
        cv2.putText(
            display,
            str(remaining),
            ((w - number_width) // 2, h // 2 + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            4.0,
            (0, 255, 120),
            5,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW, display)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            return False
    bg_model.finalize()
    if not bg_model.is_valid():
        LOG.error("Unable to build a valid background model; retry calibration")
        return False
    LOG.info("Background calibration complete")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    camera_index = _validate_args(args, parser)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    if not _load_runtime_dependencies():
        return 1

    import cv2
    import numpy as np

    from core.camera import Camera
    from core.config import config
    from core.state import FrameContext
    from effects.invisibility import InvisibilityEffect
    from interaction.gestures import GestureDetector, GestureEvent
    from rendering.compositor import Compositor
    from vision.background import BackgroundModel
    from vision.hand_tracking import HandTracker
    from vision.segmentation import SegmentationEngine

    config.camera.width = args.width
    config.camera.height = args.height
    config.camera.fps = args.fps
    config.camera.calibration_seconds = args.calibration_seconds
    output_dir = args.screenshot_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(BANNER)
    try:
        cap = Camera(camera_index, config.camera)
    except RuntimeError as exc:
        LOG.error("%s", exc)
        return 1

    seg = tracker = None
    try:
        h, w = cap.actual_h, cap.actual_w
        LOG.info("Loading vision models")
        seg = SegmentationEngine(config.segmentation)
        tracker = HandTracker(config.gesture)
        bg_cfg = config.background
        bg_model = BackgroundModel(
            h,
            w,
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
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, min(w, 1280), min(h, 720))
        if not run_calibration(cap, bg_model, seg, w, h, args.calibration_seconds, cv2):
            return 1

        screenshot_index = 0
        context = FrameContext(frame=np.zeros((h, w, 3), dtype=np.uint8))
        invisible = False
        failed_reads = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                failed_reads += 1
                if failed_reads >= 60:
                    LOG.error("Camera stopped returning frames; exiting")
                    return 1
                time.sleep(0.02)
                continue
            failed_reads = 0
            compositor.tick()
            context.frame = frame
            context.segmentation_mask = seg.get_mask(frame)
            bg_model.update(
                frame.astype(np.float32),
                foreground_mask=context.segmentation_mask,
                allow_update=True,
            )
            # A lack of fresh EMA samples while a person occupies the frame
            # must not disable ghost mode. The calibrated reference remains
            # available until it is deliberately replaced via recalibration.
            context.background = bg_model.get_reference()
            context.background_valid = context.background is not None
            if not context.background_valid:
                invisible = False
            tracker.process(frame, context)
            if GestureEvent.PINCH_CONFIRMED in detector.update(context):
                invisible = not invisible
            effect.update_alpha(invisible)
            output = compositor.render(
                effect.apply(frame, context),
                context,
                effect.alpha,
                detector.zone.is_active,
                detector.zone.tl,
                detector.zone.br,
                invisible,
            )
            cv2.imshow(WINDOW, output)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 0
            if key == ord("s"):
                filename = output_dir / f"screenshot_{screenshot_index:04d}.png"
                if not cv2.imwrite(str(filename), output):
                    LOG.error("Could not save screenshot to %s", filename)
                else:
                    LOG.info("Screenshot saved to %s", filename)
                    screenshot_index += 1
            if key == ord("r"):
                bg_model.reset()
                detector.reset()
                invisible = False
                effect.alpha = 0.0
                if not run_calibration(cap, bg_model, seg, w, h, args.calibration_seconds, cv2):
                    return 1
    except KeyboardInterrupt:
        LOG.info("Interrupted by user")
        return 130
    except Exception:
        LOG.exception("Unexpected application error")
        return 1
    finally:
        cap.release()
        if seg is not None:
            seg.close()
        if tracker is not None:
            tracker.close()
        cv2.destroyAllWindows()
        LOG.info("Shutdown complete")


if __name__ == "__main__":
    raise SystemExit(main())
