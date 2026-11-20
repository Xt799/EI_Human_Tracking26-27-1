"""Frame sources: a video file (played at its real speed) or the live Basler camera."""
import math
import time

import cv2

CAM_WIDTH, CAM_HEIGHT = 1900, 1200   # requested from the Basler camera


class VideoSource:
    """Reads a video file by frame index. Keeps the video clock: when processing is slower than the recording, due_index() skips frames."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width, self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.clock_start = None       # wall time of video time 0

    def due_index(self):
        """Index of the frame to read next at the recording's pace: the next one, or a later one if processing fell behind."""
        next_index = self.index() + 1
        now = time.time()
        if self.clock_start is None:
            self.clock_start = now - next_index / self.fps
        return max(next_index, math.ceil((now - self.clock_start) * self.fps) - 1)

    def read_at(self, index):
        """The frame with this index, skipping forward to it. None at the end of the video."""
        while self.index() < index - 1:
            if not self.cap.grab():
                return None
        ok, frame = self.cap.read()
        return frame if ok else None

    def index(self):
        """Index of the last frame returned."""
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

    def time(self):
        """Video time of the last frame returned, in seconds."""
        return self.index() / self.fps

    def resync(self):
        """Restart the clock from the current frame, e.g. after a pause."""
        self.clock_start = None

    def close(self):
        self.cap.release()


class BaslerSource:
    """Reads the first Basler camera found. Needs the pylon driver and the pypylon package."""

    def __init__(self):
        try:
            from pypylon import pylon
        except ImportError:
            raise SystemExit("Live camera requested but pypylon is not installed.")
        self.pylon = pylon
        factory = pylon.TlFactory.GetInstance()
        devices = factory.EnumerateDevices()
        if not devices:
            raise SystemExit("No Basler camera found.")
        self.camera = pylon.InstantCamera(factory.CreateDevice(devices[0]))
        self.camera.Open()
        try:
            self.camera.Width.SetValue(CAM_WIDTH)
            self.camera.Height.SetValue(CAM_HEIGHT)
        except Exception as e:
            print(f"Could not set the camera resolution ({e}); using its default.")
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        self.t0 = time.time()

    def read(self):
        """Next frame; a failed grab is retried, so None is only returned if the camera stops."""
        while self.camera.IsGrabbing():
            grab = self.camera.RetrieveResult(5000, self.pylon.TimeoutHandling_Return)
            try:
                if grab.GrabSucceeded():
                    return self.converter.Convert(grab).GetArray()
            finally:
                grab.Release()
        return None

    def time(self):
        return time.time() - self.t0

    def resync(self):
        pass

    def close(self):
        self.camera.StopGrabbing()
        self.camera.Close()


def open_source(video):
    """The word 'camera' opens the Basler camera, anything else is a video file."""
    return BaslerSource() if video == "camera" else VideoSource(video)
