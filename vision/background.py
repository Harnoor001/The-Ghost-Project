import numpy as np

class BackgroundModel:
    def __init__(self, h, w):
        self.h, self.w = h, w
        self.reset()

    def reset(self):
        self._sum = None
        self._count = 0
        self.bg = None
        self.ready = False

    def update(self, frame_f32):
        if self._sum is None:
            self._sum = frame_f32.copy()
        else:
            self._sum += frame_f32
        
        self._count += 1
        
        if self._count >= 15 and self._count % 6 == 0:
            self.finalize()

    def finalize(self):
        if self._count > 0:
            self.bg = (self._sum / self._count).astype(np.float32)
            self.ready = True

    def get(self):
        return self.bg if self.ready else None
