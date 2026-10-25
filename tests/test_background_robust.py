"""tests/test_background_robust.py — Phase 2 Task 1 robustness tests.

All tests run without webcam, MediaPipe, or GPU.
Synthetic NumPy arrays are used throughout.

Clock injection pattern
-----------------------
BackgroundModel accepts a `clock` callable.  Tests pass a simple list-based
mock that increments on each call so we can simulate time passage without
sleeping.
"""

from __future__ import annotations

import numpy as np

from vision.background import BackgroundModel

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_frame(h: int, w: int, value: float) -> np.ndarray:
    """Uniform float32 frame of shape (H, W, 3)."""
    return np.full((h, w, 3), value, dtype=np.float32)


def make_mask(h: int, w: int, value: float) -> np.ndarray:
    """Uniform foreground mask of shape (H, W)."""
    return np.full((h, w), value, dtype=np.float32)


class MockClock:
    """Controllable monotonic clock for tests."""

    def __init__(self, t: float = 0.0):
        self._t = t

    def __call__(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


# ── TEST 1 — Simple mean (no mask) ────────────────────────────────────────────


def test_mean_no_mask():
    """Three unmasked frames → background = arithmetic mean."""
    bg = BackgroundModel(10, 10)
    for v in [10.0, 20.0, 30.0]:
        bg.update(make_frame(10, 10, v))
    bg.finalize()
    result = bg.get()
    assert result is not None
    assert np.allclose(result, 20.0, atol=0.5), f"Expected ≈20, got {result.mean():.2f}"


# ── TEST 2 — Reset ─────────────────────────────────────────────────────────────


def test_reset_clears_all_state():
    """reset() must return model to empty state with no accumulated data."""
    bg = BackgroundModel(5, 5)
    bg.update(make_frame(5, 5, 50.0))
    bg.finalize()
    assert bg.ready is True

    bg.reset()
    assert bg.ready is False
    assert bg._pixel_sum is None
    assert bg._pixel_count is None
    assert bg._global_count == 0
    assert bg.get() is None
    assert bg.is_valid() is False


# ── TEST 3 — Foreground exclusion ─────────────────────────────────────────────


def test_foreground_exclusion_binary_mask():
    """Foreground pixels must NOT influence the background mean.

    Setup: 1×2 synthetic image.
      pixel[0] = background (mask=0) with values 10, 20, 30 → expected mean 20
      pixel[1] = foreground (mask=1) with values 100, 100, 100 → must not appear as 100
    """
    H, W = 1, 2
    # calibration_min_frames=0 so masking starts immediately from frame 1
    bg = BackgroundModel(H, W, calibration_min_frames=0)

    mask = np.array([[0.0, 1.0]], dtype=np.float32)  # pixel[1] = foreground

    for v in [10.0, 20.0, 30.0]:
        frame = np.array([[[v, v, v], [100.0, 100.0, 100.0]]], dtype=np.float32)
        bg.update(frame, foreground_mask=mask, allow_update=False)

    bg.finalize()
    result = bg.get()
    assert result is not None

    # Background pixel: expect ~20
    assert np.allclose(result[0, 0], 20.0, atol=0.5), (
        f"Background pixel expected ≈20, got {result[0, 0]}"
    )

    # Foreground pixel: must NOT be 100 (the contaminated value)
    # It should fall back to the global (unmasked) mean or previous valid value
    assert not np.allclose(result[0, 1], 100.0, atol=1.0), (
        "Foreground pixel should not be 100 — it was not observed as background"
    )
    # Must also not be 0 (black / zero-filled)
    assert not np.allclose(result[0, 1], 0.0, atol=1.0), (
        "Foreground pixel should not be 0 — fallback must be non-black"
    )


# ── TEST 4 — Per-pixel counts ──────────────────────────────────────────────────


def test_per_pixel_counts_independent():
    """Different pixels can have different effective observation counts.

    Frame 1: pixel A=10 (background), pixel B=100 (foreground)
    Frame 2: pixel A=20 (background), pixel B=200 (background)

    Expected:
        A = (10 + 20) / 2 = 15
        B = 200 / 1 = 200
    """
    H, W = 1, 2
    # min_frames=0 → masking from frame 1
    bg = BackgroundModel(H, W, calibration_min_frames=0)

    frame1 = np.array([[[10.0, 10.0, 10.0], [100.0, 100.0, 100.0]]], dtype=np.float32)
    mask1 = np.array([[0.0, 1.0]], dtype=np.float32)  # B is foreground

    frame2 = np.array([[[20.0, 20.0, 20.0], [200.0, 200.0, 200.0]]], dtype=np.float32)
    mask2 = np.array([[0.0, 0.0]], dtype=np.float32)  # both background

    bg.update(frame1, foreground_mask=mask1, allow_update=False)
    bg.update(frame2, foreground_mask=mask2, allow_update=False)
    bg.finalize()

    result = bg.get()
    assert result is not None
    assert np.allclose(result[0, 0], 15.0, atol=0.5), (
        f"Pixel A: expected 15, got {result[0, 0, 0]:.2f}"
    )
    assert np.allclose(result[0, 1], 200.0, atol=0.5), (
        f"Pixel B: expected 200, got {result[0, 1, 0]:.2f}"
    )


# ── TEST 5 — EMA update ────────────────────────────────────────────────────────


def test_ema_update_single_step():
    """One EMA step: bg=100, frame=200, alpha=0.02 → ~102."""
    bg = BackgroundModel(5, 5, update_rate=0.02, foreground_protection=False)

    # Calibrate to value 100
    for _ in range(3):
        bg.update(make_frame(5, 5, 100.0))
    bg.finalize()
    assert np.allclose(bg.get(), 100.0, atol=0.5)

    # One EMA step with new frame value 200, no mask
    bg.update(make_frame(5, 5, 200.0), foreground_mask=None, allow_update=True)
    expected = (1 - 0.02) * 100.0 + 0.02 * 200.0  # = 102.0
    assert np.allclose(bg.get(), expected, atol=0.5), (
        f"Expected ≈{expected}, got {bg.get().mean():.2f}"
    )


# ── TEST 6 — Foreground protection ────────────────────────────────────────────


def test_foreground_protection_prevents_contamination():
    """Person pixels (mask=1) must not be absorbed into the background.

    Apply 50 EMA updates with frame=255, mask=1.
    Background must remain near its calibrated value of 100.
    """
    bg = BackgroundModel(5, 5, update_rate=0.02, foreground_protection=True)

    for _ in range(3):
        bg.update(make_frame(5, 5, 100.0))
    bg.finalize()
    initial = bg.get().mean()

    full_fg_mask = make_mask(5, 5, 1.0)
    for _ in range(50):
        bg.update(make_frame(5, 5, 255.0), foreground_mask=full_fg_mask, allow_update=True)

    final = bg.get().mean()
    assert abs(final - initial) < 1.0, (
        f"Background drifted too much: start={initial:.2f}, end={final:.2f}"
    )


# ── TEST 7 — Partial mask (0.5) ───────────────────────────────────────────────


def test_partial_mask_ema_scaling():
    """With mask=0.5 the effective alpha is halved compared to mask=0.

    One step:
        eff_alpha = 0.02 * (1 - 0.5) = 0.01
        new_bg = 0.99 * 100 + 0.01 * 200 = 101
    """
    bg = BackgroundModel(5, 5, update_rate=0.02, foreground_protection=True)
    for _ in range(3):
        bg.update(make_frame(5, 5, 100.0))
    bg.finalize()

    half_mask = make_mask(5, 5, 0.5)
    bg.update(make_frame(5, 5, 200.0), foreground_mask=half_mask, allow_update=True)

    eff_alpha = 0.02 * 0.5  # = 0.01
    expected = (1 - eff_alpha) * 100.0 + eff_alpha * 200.0  # = 101.0
    assert np.allclose(bg.get(), expected, atol=0.5), (
        f"Expected ≈{expected}, got {bg.get().mean():.2f}"
    )


# ── TEST 8 — No-mask EMA compatibility ────────────────────────────────────────


def test_no_mask_ema_compatibility():
    """update(frame, foreground_mask=None, allow_update=True) must not crash."""
    bg = BackgroundModel(5, 5, update_rate=0.02, foreground_protection=True)
    for _ in range(3):
        bg.update(make_frame(5, 5, 100.0))
    bg.finalize()

    # Should work without a mask
    bg.update(make_frame(5, 5, 200.0), foreground_mask=None, allow_update=True)
    result = bg.get()
    assert result is not None
    # Background should have shifted slightly toward 200
    assert result.mean() > 100.0, "EMA should move background slightly toward new frame"


# ── TEST 9 — Lighting adaptation ──────────────────────────────────────────────


def test_lighting_adaptation_gradual():
    """Background adapts gradually when lighting changes.

    After calibrating to 50, provide 200 frames at value 150.
    Background should trend toward 150 but not jump there immediately.
    """
    bg = BackgroundModel(5, 5, update_rate=0.02, foreground_protection=False)
    for _ in range(5):
        bg.update(make_frame(5, 5, 50.0))
    bg.finalize()

    # After 1 step it should barely move
    bg.update(make_frame(5, 5, 150.0), allow_update=True)
    after_1 = bg.get().mean()
    assert after_1 < 55.0, f"After 1 step, background jumped too fast: {after_1:.2f}"

    # After 200 more steps it should be significantly closer to 150
    for _ in range(200):
        bg.update(make_frame(5, 5, 150.0), allow_update=True)
    after_many = bg.get().mean()
    assert after_many > 80.0, f"Background did not adapt enough after 200 steps: {after_many:.2f}"
    assert after_many < 150.0, f"Background jumped instantly to target: {after_many:.2f}"


# ── TEST 10 — Stale detection (mock clock) ────────────────────────────────────


def test_stale_detection_mock_clock():
    """is_valid() returns False when elapsed > stale_threshold_sec.

    Uses injectable clock so no real sleep is needed.
    """
    clock = MockClock(t=0.0)
    bg = BackgroundModel(5, 5, stale_threshold_sec=5.0, clock=clock)

    for _ in range(3):
        bg.update(make_frame(5, 5, 100.0))
    bg.finalize()

    # At t=0 model should be valid
    assert bg.is_valid() is True

    # Advance past the stale threshold
    clock.advance(6.0)
    assert bg.is_valid() is False
    assert bg.stale is True

    # Background data is NOT destroyed — just flagged stale
    assert bg.get() is None  # get() returns None when not valid
    assert bg.bg is not None  # but underlying data is still there
    # The effect may continue using a stable calibrated reference.
    assert bg.get_reference() is bg.bg


# ── TEST 11 — Recovery after stale ────────────────────────────────────────────


def test_recovery_after_stale():
    """reset() clears stale state; recalibration restores is_valid()."""
    clock = MockClock(t=0.0)
    bg = BackgroundModel(5, 5, stale_threshold_sec=5.0, clock=clock)

    for _ in range(3):
        bg.update(make_frame(5, 5, 100.0))
    bg.finalize()

    clock.advance(10.0)
    assert bg.is_valid() is False

    bg.reset()
    assert bg.ready is False
    assert bg.stale is False

    # Re-calibrate
    for _ in range(3):
        bg.update(make_frame(5, 5, 80.0))
    bg.finalize()
    assert bg.is_valid() is True
    assert np.allclose(bg.get(), 80.0, atol=0.5)


# ── TEST 12 — Memory regression ───────────────────────────────────────────────


def test_no_frame_buffer_in_memory():
    """Model must never hold a list/deque of full frames.

    Checks that no attribute with a name suggesting frame storage exists.
    """
    bg = BackgroundModel(100, 100)
    for _ in range(30):
        bg.update(make_frame(100, 100, 50.0))

    # Known forbidden attribute names
    forbidden = ("buf", "frames", "frame_buffer", "history", "frame_list", "deque", "buffer")
    for name in forbidden:
        assert not hasattr(bg, name), f"BackgroundModel has forbidden attribute: {name!r}"

    # The only ndarray attributes should be O(frame_size), not O(frames × frame_size)
    frame_size = 100 * 100 * 3 * 4  # bytes for float32 (H, W, 3)
    max_acceptable = frame_size * 5  # generous: at most 5× frame size total
    arr_bytes = sum(a.nbytes for a in vars(bg).values() if isinstance(a, np.ndarray))
    assert arr_bytes <= max_acceptable, (
        f"Total ndarray memory {arr_bytes} bytes exceeds {max_acceptable} bytes "
        f"(O(frame_size) × 5). Frame buffers may have been introduced."
    )
