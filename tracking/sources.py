"""Frame sources: a video file (played at its real speed) or the live Basler camera."""
import time

import cv2

CAM_WIDTH, CAM_HEIGHT = 1900, 1200   # requested from the Basler camera


class VideoSource:
    """Reads a video file. Keeps the video clock: when processing is slower than the recording, frames are dropped."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.clock_start = None       # wall time of video time 0

    def read(self):
        """Next frame to process, or None at the end of the video."""
        pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        if self.clock_start is None:
            self.clock_start = time.time() - pos / self.fps
        target = (time.time() - self.clock_start) * self.fps
        while pos < target - 1:
            if not self.cap.grab():
                return None
            pos += 1
        ok, frame = self.cap.read()
        return frame if ok else None

    def time(self):
        """Video time of the last frame returned, in seconds."""
        return (self.cap.get(cv2.CAP_PROP_POS_FRAMES) - 1) / self.fps

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
