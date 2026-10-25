import sys

import cv2

from core.config import CameraConfig


class Camera:
    def __init__(self, index: int, config: CameraConfig):
        self.config = config
        self.cap = None
        self.device_name = "CPU"
        self._detect_device()

        backends = (
            [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "MSMF"), (cv2.CAP_ANY, "Auto")]
            if sys.platform == "win32"
            else [(cv2.CAP_ANY, "Auto")]
        )

        for backend, name in backends:
            print(f"[CAM] Trying {name} ...")
            # Pass the camera index and backend separately. Adding the backend
            # constant to the index attempts to open a different device.
            _c = cv2.VideoCapture(index, backend)
            if _c.isOpened():
                _c.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
                _c.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
                _c.set(cv2.CAP_PROP_FPS, self.config.fps)
                ret, probe = _c.read()
                if ret and probe is not None:
                    self.cap = _c
                    self.actual_h, self.actual_w = probe.shape[:2]
                    print(f"[CAM] OK with {name} (Resolution: {self.actual_w}x{self.actual_h})")
                    break
                _c.release()

        if self.cap is None:
            raise RuntimeError("Could not open webcam. Close Teams/Zoom first.")

    def _detect_device(self):
        try:
            import torch

            if torch.cuda.is_available():
                self.device_name = "GPU/" + torch.cuda.get_device_name(0).split()[0]
        except ImportError:
            pass

    def read(self):
        return self.cap.read()

    def release(self):
        if self.cap:
            self.cap.release()
