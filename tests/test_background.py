"""tests/test_background.py — backwards-compatible tests for BackgroundModel 2.0.

These tests preserve the behaviour verified in Phase 1.  The API has changed:
  - update() now accepts foreground_mask= and allow_update= keyword arguments
  - The internal attribute _sum is gone (replaced by _pixel_sum and _pixel_count)
  - BackgroundModel(h, w) still works with all defaults
"""

import numpy as np
import pytest
from vision.background import BackgroundModel


def test_initial_state():
    bg = BackgroundModel(720, 1280)
    assert bg.get() is None
    assert bg.ready is False
    assert bg._global_count == 0


def test_single_frame():
    bg = BackgroundModel(100, 100)
    frame = np.ones((100, 100, 3), dtype=np.float32) * 50
    bg.update(frame)

    assert bg._global_count == 1
    assert bg._pixel_sum is not None
    assert bg.get() is None
    assert bg.ready is False


def test_mean_calculation():
    bg = BackgroundModel(10, 10)
    frame1 = np.ones((10, 10, 3), dtype=np.float32) * 10
    frame2 = np.ones((10, 10, 3), dtype=np.float32) * 20
    frame3 = np.ones((10, 10, 3), dtype=np.float32) * 30

    bg.update(frame1)
    bg.update(frame2)
    bg.update(frame3)

    bg.finalize()

    assert bg.ready is True
    result = bg.get()
    assert result is not None
    assert np.allclose(result, np.ones((10, 10, 3), dtype=np.float32) * 20)


def test_different_pixel_values():
    bg = BackgroundModel(2, 2)

    frame1 = np.array([
        [[10, 20, 30], [40, 50, 60]],
        [[70, 80, 90], [100, 110, 120]]
    ], dtype=np.float32)

    frame2 = np.array([
        [[30, 40, 50], [60, 70, 80]],
        [[90, 100, 110], [120, 130, 140]]
    ], dtype=np.float32)

    bg.update(frame1)
    bg.update(frame2)
    bg.finalize()

    expected = np.array([
        [[20, 30, 40], [50, 60, 70]],
        [[80, 90, 100], [110, 120, 130]]
    ], dtype=np.float32)

    assert np.allclose(bg.get(), expected)


def test_reset():
    bg = BackgroundModel(10, 10)
    frame = np.ones((10, 10, 3), dtype=np.float32) * 50
    bg.update(frame)
    bg.finalize()

    assert bg.ready is True
    assert bg._global_count == 1

    bg.reset()

    assert bg.ready is False
    assert bg._global_count == 0
    assert bg._pixel_sum is None
    assert bg.get() is None


def test_additional_frames():
    bg = BackgroundModel(10, 10)
    frame1 = np.ones((10, 10, 3), dtype=np.float32) * 10
    bg.update(frame1)
    bg.finalize()

    assert np.allclose(bg.get(), frame1)

    frame2 = np.ones((10, 10, 3), dtype=np.float32) * 30
    bg.update(frame2)
    bg.finalize()

    expected = np.ones((10, 10, 3), dtype=np.float32) * 20
    assert np.allclose(bg.get(), expected)


def test_no_frame_retention():
    bg = BackgroundModel(10, 10)
    # Ensure there are no deque or list attributes tracking full frames
    assert not hasattr(bg, 'buf')
    assert not hasattr(bg, 'frames')
