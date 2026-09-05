"""
vision/background.py — Background Model 2.0

Design decisions
────────────────
* Per-pixel accumulation during calibration
  First `calibration_min_frames` frames are accumulated without a mask so
  that the segmentation model has time to warm up.  After that, each frame
  can carry a foreground mask (0 = background, 1 = foreground).  We track
  a per-pixel background weight (`_pixel_count`, float32) so that foreground
  pixels are excluded from the average without poisoning it.

* Memory: O(frame_size) — three arrays of shape (H, W, 3) or (H, W) at most.
  No frame buffer, no deque, no list of frames.

* EMA after calibration
  alpha = update_rate (default 0.02).  When foreground_protection is on,
  effective_alpha[h,w] = alpha * (1 − mask[h,w]), so person pixels are
  never absorbed into the background.

* Stale detection
  _last_update_ts is updated only when at least some background-weight pixels
  received real information.  A fully-foreground frame does NOT refresh the
  timestamp.  The default clock is time.monotonic(); tests can inject a
  callable via the `clock` constructor parameter.

* dtype rationale for _pixel_count
  float32 — masks from MediaPipe are continuous (e.g. 0.7 background), not
  binary.  Using float32 allows fractional weights and avoids any uint16
  overflow issues.  Memory cost: H × W × 4 bytes ≈ 3.7 MB for 1080p.
"""

from __future__ import annotations

import time
from enum import Enum, auto
from typing import Callable, Optional

import numpy as np


class _State(Enum):
    EMPTY      = auto()
    CALIBRATING = auto()
    READY      = auto()


class BackgroundModel:
    """Streaming background estimator with per-pixel masked calibration.

    Args:
        h, w: Frame height / width.
        calibration_min_frames: Number of unmasked warm-up frames before the
            mask-aware accumulation starts.
        calibration_update_interval: Kept for API compatibility.
        update_enabled: Whether to run EMA updates after calibration.
        update_rate: EMA alpha (0–1). 0.02 at 30 FPS gives slow adaptation.
        foreground_protection: If True, EMA alpha is scaled by background
            weight so person pixels are not absorbed into the background.
        stale_threshold_sec: Seconds of no meaningful background update
            before is_valid() returns False.
        clock: Callable that returns current time in seconds.
            Defaults to time.monotonic. Inject a mock for testing.
    """

    def __init__(
        self,
        h: int,
        w: int,
        *,
        calibration_min_frames: int = 15,
        calibration_update_interval: int = 6,   # not used, kept for compat
        update_enabled: bool = True,
        update_rate: float = 0.02,
        foreground_protection: bool = True,
        stale_threshold_sec: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.h = h
        self.w = w
        self._calibration_min_frames    = calibration_min_frames
        self._calibration_update_interval = calibration_update_interval
        self._update_enabled            = update_enabled
        self._update_rate               = update_rate
        self._foreground_protection     = foreground_protection
        self._stale_threshold_sec       = stale_threshold_sec
        self._clock                     = clock

        self.reset()

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all calibration state. Model returns to EMPTY."""
        self._state: _State = _State.EMPTY

        # Streaming accumulators — lazily allocated on first frame
        self._pixel_sum:   Optional[np.ndarray] = None   # float32, (H, W, 3)
        self._pixel_count: Optional[np.ndarray] = None   # float32, (H, W)
        self._global_count: int = 0                      # unmasked frame count

        # Finalised background
        self.bg:    Optional[np.ndarray] = None
        self.ready: bool = False

        self._last_update_ts: float = 0.0
        self._frame_idx: int = 0

    def update(
        self,
        frame_f32: np.ndarray,
        foreground_mask: Optional[np.ndarray] = None,
        allow_update: bool = False,
    ) -> None:
        """Feed a frame to the model.

        During calibration (`allow_update=False`):
            Accumulates frames for background estimation.

        After calibration (`allow_update=True` and model is READY):
            Optionally refines the background with an EMA step.

        Args:
            frame_f32: Frame as float32 array of shape (H, W, 3).
            foreground_mask: Float32 array of shape (H, W).
                0.0 = background, 1.0 = foreground.
                None means no mask is available.
            allow_update: Set True in the main loop after calibration to
                enable EMA background updates.
        """
        self._validate_frame(frame_f32)

        if not allow_update:
            # ── CALIBRATION PHASE ────────────────────────────────────────
            self._state = _State.CALIBRATING
            self._accumulate(frame_f32, foreground_mask)
        else:
            # ── EMA UPDATE PHASE (READY) ─────────────────────────────────
            if self.ready and self._update_enabled:
                self._ema_update(frame_f32, foreground_mask)

    def finalize(self) -> None:
        """Compute the background mean from accumulated frames.

        Pixels with no background observations fall back to:
          1. The global unmasked mean (available when calibration_min_frames > 0
             and frames were accepted without masks during warm-up).
          2. The spatial mean of all *observed* background pixels in the frame,
             as a last resort when even the global count is zero (e.g. when
             calibration_min_frames=0 and masks were provided from frame 1).
          3. As an absolute fallback, use 128.0 (mid-grey) — prevents black pixels.
        """
        if self._pixel_sum is None or self._global_count == 0 and not np.any(self._pixel_count > 0):
            return

        # ── Build fallback value for unobserved pixels ─────────────────────
        # We use the spatial mean of *observed* background pixels as the fallback.
        # Using _pixel_sum / _global_count is wrong when masks are present because
        # unobserved pixels (weight=0 every frame) have pixel_sum=0 regardless of
        # global_count — that produces black pixels, not the background mean.
        observed = self._pixel_count > 0   # (H, W)
        if np.any(observed):
            # Per-channel mean of only the observed background pixels
            obs_mean_per_ch = (
                self._pixel_sum[observed].sum(axis=0)         # (3,)
                / self._pixel_count[observed].sum()            # scalar
            )  # shape (3,)
            global_mean = np.broadcast_to(
                obs_mean_per_ch[np.newaxis, np.newaxis, :],   # (1, 1, 3)
                (self.h, self.w, 3)
            ).astype(np.float32).copy()
        elif self._global_count > 0:
            # All pixels fully foreground — fall back to unmasked mean
            global_mean = (self._pixel_sum / self._global_count).astype(np.float32)
        else:
            global_mean = np.full((self.h, self.w, 3), 128.0, dtype=np.float32)

        # ── Per-pixel mean where observations exist ────────────────────────
        count = self._pixel_count[..., np.newaxis]          # (H, W, 1)
        has_obs = self._pixel_count > 0                     # (H, W) bool
        bg = np.where(
            has_obs[..., np.newaxis],
            self._pixel_sum / np.maximum(count, 1e-6),
            global_mean,
        ).astype(np.float32)

        self.bg = bg
        self.ready = True
        self._state = _State.READY
        self._last_update_ts = self._clock()

    def get(self) -> Optional[np.ndarray]:
        """Return the current background, or None if not ready or stale."""
        return self.bg if self.is_valid() else None

    def is_valid(self) -> bool:
        """True when the background is ready AND not stale."""
        if not self.ready:
            return False
        elapsed = self._clock() - self._last_update_ts
        return elapsed <= self._stale_threshold_sec

    @property
    def stale(self) -> bool:
        """True when the background is ready but the timestamp has expired."""
        if not self.ready:
            return False
        return not self.is_valid()

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _validate_frame(self, frame: np.ndarray) -> None:
        if frame.ndim != 3 or frame.shape[0] != self.h or frame.shape[1] != self.w or frame.shape[2] != 3:
            raise ValueError(
                f"BackgroundModel expects frame shape ({self.h}, {self.w}, 3), "
                f"got {frame.shape}"
            )

    def _accumulate(
        self,
        frame_f32: np.ndarray,
        foreground_mask: Optional[np.ndarray],
    ) -> None:
        """Streaming accumulation during calibration.

        Phase 1 — first `calibration_min_frames` frames:
            Accept without mask (warm-up period).
            Uses global count so finalize() has a fallback for every pixel.

        Phase 2 — subsequent frames:
            If a foreground mask is provided, use per-pixel weighted
            accumulation so foreground pixels do not contaminate the average.
            If no mask is provided, fall back to unmasked accumulation.
        """
        self._frame_idx += 1

        # Lazy array allocation on first call
        if self._pixel_sum is None:
            self._pixel_sum   = np.zeros((self.h, self.w, 3), dtype=np.float32)
            self._pixel_count = np.zeros((self.h, self.w),    dtype=np.float32)

        if self._frame_idx <= self._calibration_min_frames or foreground_mask is None:
            # ── Unmasked warm-up ─────────────────────────────────────────
            self._pixel_sum   += frame_f32
            self._pixel_count += 1.0
            self._global_count += 1
        else:
            # ── Masked per-pixel accumulation ────────────────────────────
            bg_weight = (1.0 - foreground_mask).astype(np.float32)   # (H, W)

            # Only accumulate frames that have some background content.
            # Avoids inflating _pixel_count for fully-foreground frames.
            if np.any(bg_weight > 0.05):
                self._pixel_sum   += frame_f32 * bg_weight[..., np.newaxis]
                self._pixel_count += bg_weight
                self._global_count += 1   # for fallback mean computation

    def _ema_update(
        self,
        frame_f32: np.ndarray,
        foreground_mask: Optional[np.ndarray],
    ) -> None:
        """EMA refinement after calibration.

        background ← (1 − eff_alpha) * background + eff_alpha * frame

        When foreground_protection is enabled:
            eff_alpha[h,w] = update_rate * (1 − mask[h,w])
        So foreground pixels contribute zero to the EMA.
        """
        alpha = self._update_rate

        if foreground_mask is not None and self._foreground_protection:
            bg_weight = (1.0 - foreground_mask).astype(np.float32)   # (H, W)
            eff_alpha = (alpha * bg_weight)[..., np.newaxis]          # (H, W, 1)
        else:
            eff_alpha = alpha
            bg_weight = None

        self.bg = (
            (1.0 - eff_alpha) * self.bg + eff_alpha * frame_f32
        ).astype(np.float32)

        # Only refresh timestamp if some background pixels were updated
        updated = (
            bg_weight is None or np.any(bg_weight > 0.05)
        )
        if updated:
            self._last_update_ts = self._clock()
