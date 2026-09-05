from core.config import GestureConfig
from core.state import FrameContext

class GestureDetector:
    def __init__(self, config: GestureConfig):
        self.config = config
        self._touch_cd = 0
        self.invisible = False
        self.is_active = False
        self.tl = None
        self.br = None

    def update(self, ctx: FrameContext):
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

        if self._touch_cd > 0:
            self._touch_cd -= 1

        if ctx.fingers_touching and ctx.hand_count >= 1 and self._touch_cd == 0:
            self.invisible = not self.invisible
            self._touch_cd = self.config.touch_cooldown_frames

    def reset(self):
        self.invisible = False
        self._touch_cd = 0
