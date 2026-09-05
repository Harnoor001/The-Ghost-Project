import time
from enum import Enum, auto
from core.config import GestureConfig
from core.state import FrameContext

class GestureState(Enum):
    IDLE = auto()
    PINCH_DETECTED = auto()
    PINCH_CONFIRMED = auto()
    COOLDOWN = auto()

class GestureEvent(Enum):
    PINCH_CONFIRMED = auto()
    PINCH_RELEASED = auto()

class InteractionZone:
    def __init__(self):
        self.is_active = False
        self.tl = None
        self.br = None

    def calculate(self, ctx: FrameContext):
        if ctx.tips is not None:
            p1, p2 = ctx.tips
            self.tl = (min(p1[0], p2[0]), min(p1[1], p2[1]))
            self.br = (max(p1[0], p2[0]), max(p1[1], p2[1]))
            w_box = self.br[0] - self.tl[0]
            h_box = self.br[1] - self.tl[1]
            self.is_active = w_box > 80 and h_box > 80
        else:
            self.is_active = False
            self.tl = None
            self.br = None

class GestureDetector:
    def __init__(self, config: GestureConfig):
        self.config = config
        self.state = GestureState.IDLE
        self.state_entered_at = time.monotonic()
        self.zone = InteractionZone()

    def update(self, ctx: FrameContext) -> list[GestureEvent]:
        self.zone.calculate(ctx)
        
        events = []
        now = time.monotonic()
        
        if ctx.pinch_ratios:
            min_ratio = min(ctx.pinch_ratios)
        else:
            min_ratio = 999.0

        if self.state == GestureState.IDLE:
            if min_ratio < self.config.pinch_enter_threshold:
                self.state = GestureState.PINCH_DETECTED
                self.state_entered_at = now
                
        elif self.state == GestureState.PINCH_DETECTED:
            if min_ratio > self.config.pinch_exit_threshold:
                self.state = GestureState.IDLE
            elif (now - self.state_entered_at) >= self.config.pinch_confirmation_sec:
                self.state = GestureState.PINCH_CONFIRMED
                self.state_entered_at = now
                events.append(GestureEvent.PINCH_CONFIRMED)
                
        elif self.state == GestureState.PINCH_CONFIRMED:
            if min_ratio > self.config.pinch_exit_threshold:
                self.state = GestureState.COOLDOWN
                self.state_entered_at = now
                events.append(GestureEvent.PINCH_RELEASED)
                
        elif self.state == GestureState.COOLDOWN:
            if (now - self.state_entered_at) >= self.config.gesture_cooldown_sec:
                self.state = GestureState.IDLE

        # Maintain legacy fingers_touching logic for HUD rendering
        ctx.fingers_touching = (self.state in (GestureState.PINCH_DETECTED, GestureState.PINCH_CONFIRMED))

        return events

    def reset(self):
        self.state = GestureState.IDLE
        self.state_entered_at = time.monotonic()
        self.zone.is_active = False
