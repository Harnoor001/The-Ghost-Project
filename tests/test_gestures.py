import time
import pytest
import numpy as np
from core.config import GestureConfig
from core.state import FrameContext
from interaction.gestures import GestureDetector, GestureState, GestureEvent

@pytest.fixture
def config():
    return GestureConfig(
        pinch_enter_threshold=0.40,
        pinch_exit_threshold=0.50,
        pinch_confirmation_sec=0.15,
        gesture_cooldown_sec=0.50
    )

@pytest.fixture
def detector(config):
    return GestureDetector(config)

def create_ctx(ratios=None):
    if ratios is None:
        ratios = []
    ctx = FrameContext(frame=np.zeros((10, 10, 3)))
    ctx.pinch_ratios = ratios
    return ctx

def test_no_pinch(detector):
    ctx = create_ctx([0.60])
    events = detector.update(ctx)
    assert not events
    assert detector.state == GestureState.IDLE

def test_pinch_confirmation(detector, monkeypatch):
    ctx = create_ctx([0.30])
    
    # Start pinch
    start_time = 0.0
    monkeypatch.setattr(time, 'monotonic', lambda: start_time)
    events = detector.update(ctx)
    assert not events
    assert detector.state == GestureState.PINCH_DETECTED
    
    # Advance time beyond confirmation duration
    monkeypatch.setattr(time, 'monotonic', lambda: start_time + 0.20)
    events = detector.update(ctx)
    assert GestureEvent.PINCH_CONFIRMED in events
    assert detector.state == GestureState.PINCH_CONFIRMED

def test_continuous_pinch(detector, monkeypatch):
    ctx = create_ctx([0.30])
    
    start_time = 0.0
    monkeypatch.setattr(time, 'monotonic', lambda: start_time)
    detector.update(ctx)
    
    # Confirm pinch
    monkeypatch.setattr(time, 'monotonic', lambda: start_time + 0.20)
    events = detector.update(ctx)
    assert GestureEvent.PINCH_CONFIRMED in events
    
    # Keep holding pinch for a long time
    monkeypatch.setattr(time, 'monotonic', lambda: start_time + 5.0)
    events = detector.update(ctx)
    assert not events # No extra toggle events!
    assert detector.state == GestureState.PINCH_CONFIRMED

def test_pinch_release(detector, monkeypatch):
    ctx_pinch = create_ctx([0.30])
    ctx_release = create_ctx([0.60])
    
    start_time = 0.0
    monkeypatch.setattr(time, 'monotonic', lambda: start_time)
    detector.update(ctx_pinch)
    
    monkeypatch.setattr(time, 'monotonic', lambda: start_time + 0.20)
    detector.update(ctx_pinch) # Confirmed
    
    # Release pinch
    monkeypatch.setattr(time, 'monotonic', lambda: start_time + 0.30)
    events = detector.update(ctx_release)
    assert GestureEvent.PINCH_RELEASED in events
    assert detector.state == GestureState.COOLDOWN

def test_second_pinch(detector, monkeypatch):
    ctx_pinch = create_ctx([0.30])
    ctx_release = create_ctx([0.60])
    
    t = 0.0
    monkeypatch.setattr(time, 'monotonic', lambda: t)
    detector.update(ctx_pinch) # Detected
    
    t = 0.2
    monkeypatch.setattr(time, 'monotonic', lambda: t)
    events = detector.update(ctx_pinch) # Confirmed
    assert GestureEvent.PINCH_CONFIRMED in events
    
    t = 0.3
    monkeypatch.setattr(time, 'monotonic', lambda: t)
    detector.update(ctx_release) # Released -> Cooldown
    
    t = 0.9 # Past 0.5s cooldown
    monkeypatch.setattr(time, 'monotonic', lambda: t)
    detector.update(ctx_release) # Cooldown -> Idle
    
    t = 1.0
    monkeypatch.setattr(time, 'monotonic', lambda: t)
    detector.update(ctx_pinch) # Detected 2
    
    t = 1.2
    monkeypatch.setattr(time, 'monotonic', lambda: t)
    events = detector.update(ctx_pinch) # Confirmed 2
    assert GestureEvent.PINCH_CONFIRMED in events

def test_hysteresis(detector, monkeypatch):
    start_time = 0.0
    monkeypatch.setattr(time, 'monotonic', lambda: start_time)
    
    # Enter pinch
    detector.update(create_ctx([0.35]))
    assert detector.state == GestureState.PINCH_DETECTED
    
    # Ratio goes up to 0.45 (between 0.40 and 0.50)
    detector.update(create_ctx([0.45]))
    assert detector.state == GestureState.PINCH_DETECTED # Remains detected
    
    # Confirm
    monkeypatch.setattr(time, 'monotonic', lambda: start_time + 0.20)
    detector.update(create_ctx([0.45]))
    assert detector.state == GestureState.PINCH_CONFIRMED
    
    # Ratio goes up to 0.48 (still between 0.40 and 0.50)
    detector.update(create_ctx([0.48]))
    assert detector.state == GestureState.PINCH_CONFIRMED
    
    # Ratio crosses exit threshold
    detector.update(create_ctx([0.55]))
    assert detector.state == GestureState.COOLDOWN

def test_cooldown(detector, monkeypatch):
    t = 0.0
    monkeypatch.setattr(time, 'monotonic', lambda: t)
    
    detector.update(create_ctx([0.30])) # Detected
    
    t = 0.2
    monkeypatch.setattr(time, 'monotonic', lambda: t)
    detector.update(create_ctx([0.30])) # Confirmed
    
    t = 0.3
    monkeypatch.setattr(time, 'monotonic', lambda: t)
    detector.update(create_ctx([0.60])) # Released -> Cooldown
    
    # Try to pinch again immediately
    t = 0.5 # Wait, cooldown is 0.5s, so it expires at 0.3 + 0.5 = 0.8
    monkeypatch.setattr(time, 'monotonic', lambda: t)
    detector.update(create_ctx([0.30]))
    # State should still be COOLDOWN!
    assert detector.state == GestureState.COOLDOWN

def test_fps_independence(detector, monkeypatch):
    ctx = create_ctx([0.30])
    
    # High FPS simulation: lots of small time increments
    t = 0.0
    monkeypatch.setattr(time, 'monotonic', lambda: t)
    detector.update(ctx)
    
    for _ in range(10):
        t += 0.01 # 10 ms steps (100 FPS)
        monkeypatch.setattr(time, 'monotonic', lambda: t)
        events = detector.update(ctx)
        assert not events
        
    # Jump to confirmation
    t = 0.20
    monkeypatch.setattr(time, 'monotonic', lambda: t)
    events = detector.update(ctx)
    assert GestureEvent.PINCH_CONFIRMED in events

def test_two_hands(detector, monkeypatch):
    # Both hands pinch simultaneously (one at 0.35, one at 0.25)
    ctx = create_ctx([0.35, 0.25])
    
    t = 0.0
    monkeypatch.setattr(time, 'monotonic', lambda: t)
    detector.update(ctx)
    assert detector.state == GestureState.PINCH_DETECTED
    
    t = 0.2
    monkeypatch.setattr(time, 'monotonic', lambda: t)
    events = detector.update(ctx)
    
    # Still only ONE event, even though both are pinching
    assert len(events) == 1
    assert events[0] == GestureEvent.PINCH_CONFIRMED
