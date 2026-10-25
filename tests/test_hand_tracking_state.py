"""Regression tests for hand-tracking state reset behavior."""

import numpy as np

from core.state import FrameContext
from vision.hand_tracking import HandTracker


class _NoHands:
    multi_hand_landmarks = None


class _HandsProcessor:
    def process(self, _image):
        return _NoHands()


def test_lost_hands_clear_stale_pinch_ratios(monkeypatch):
    """A lost hand must release the gesture so a later pinch can toggle again."""
    tracker = HandTracker.__new__(HandTracker)
    tracker.config = type("Config", (), {"hand_width": 10})()
    tracker.hands = _HandsProcessor()
    context = FrameContext(frame=np.zeros((10, 10, 3), dtype=np.uint8), pinch_ratios=[0.2])

    monkeypatch.setattr("vision.hand_tracking.cv2.resize", lambda frame, *_args, **_kwargs: frame)
    monkeypatch.setattr("vision.hand_tracking.cv2.cvtColor", lambda frame, _code: frame)

    tracker.process(context.frame, context)

    assert context.hand_count == 0
    assert context.pinch_ratios == []
