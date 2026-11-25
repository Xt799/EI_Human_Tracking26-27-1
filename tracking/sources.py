"""Frame sources: a video file (played at its real speed) or the live Basler camera."""
import math
import time
from dataclasses import dataclass
from threading import Event, Lock, Thread

import cv2

CAM_WIDTH, CAM_HEIGHT = 1900, 1200   # requested from the Basler camera


class VideoSource:
    """Reads a video file by frame index. Keeps the video clock: when processing is slower than the recording, due_index() skips frames."""

    def __init__(self, path, shift=0):
        self.shift = shift  # frame offset relative to the primary recording
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.cap.release()
            raise SystemExit(f"Could not open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width, self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.clock_start = None       # wall time of video time 0

    def due_index(self):
        """Index of the frame to read next at the recording's pace: the next one, or a later one if processing fell behind."""
        next_index = self.index() + 1
        now = time.monotonic()
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


LIVE_TIMEOUT_S = 5.0
GRAB_TIMEOUT_MS = 200


def pylon_devices():
    try:
        from pypylon import genicam, pylon
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"Could not load pypylon. Install the project environment and camera drivers: {exc}") from exc
    try:
        factory = pylon.TlFactory.GetInstance()
        devices = {device.GetSerialNumber(): device for device in factory.EnumerateDevices()}
    except genicam.GenericException as exc:
        raise RuntimeError(f"Could not enumerate Basler cameras. Check pylon Viewer. {exc}") from exc
    return pylon, genicam, factory, devices


def list_baslers():
    _, _, _, devices = pylon_devices()
    return [(serial, device.GetModelName(), device.GetDeviceClass()) for serial, device in sorted(devices.items())]


@dataclass(frozen=True)
class CapturedFrame:
    image: object
    received_at: float


class BaslerSource:
    """One serial-selected camera. After start(), only its worker grabs frames."""

    def __init__(self, serial, devices=None):
        self.serial = serial
        self.pylon, self.genicam, factory, available = pylon_devices() if devices is None else devices
        if serial not in available:
            found = ", ".join(sorted(available)) or "none"
            raise RuntimeError(f"Basler {serial} unavailable (found: {found}). Check power, connection and pylon Viewer.")
        self.camera = None
        self.lock = Lock()
        self.stop = Event()
        self.worker = None
        self.latest = None
        self.last_received = None
        self.error = None
        self.overwritten = self.skipped = self.received = 0
        try:
            self.camera = self.pylon.InstantCamera(factory.CreateDevice(available[serial]))
            self.camera.Open()
            for name, size in (("Width", CAM_WIDTH), ("Height", CAM_HEIGHT)):
                node = getattr(self.camera, name, None)
                if node is not None and self.genicam.IsWritable(node) and node.Min <= size <= node.Max and (size - node.Min) % node.Inc == 0:
                    node.Value = size
            self.geometry = self.read_geometry()
            if not all(self.geometry.get(axis, 0) > 0 for axis in ("Width", "Height")):
                raise RuntimeError(f"Basler {serial} did not report a readable frame size.")
            self.converter = self.pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = self.pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment.Value = "MsbAligned"
        except BaseException as exc:
            try:
                self.close()
            except self.genicam.GenericException as cleanup_error:
                exc.add_note(f"Error closing Basler {serial}: {cleanup_error}")
            if isinstance(exc, self.genicam.GenericException):
                raise RuntimeError(f"Could not open Basler {serial}. Check the connection and close pylon Viewer. {exc}") from exc
            raise

    def read_geometry(self):
        names = ("Width", "Height", "OffsetX", "OffsetY", "BinningHorizontal", "BinningVertical",
                 "DecimationHorizontal", "DecimationVertical", "ReverseX", "ReverseY")
        nodes = ((name, getattr(self.camera, name, None)) for name in names)
        return {name: node.Value for name, node in nodes if node is not None and self.genicam.IsReadable(node)}

    def start(self):
        try:
            self.camera.StartGrabbing(self.pylon.GrabStrategy_LatestImageOnly)
        except self.genicam.GenericException as exc:
            raise RuntimeError(f"Could not start capture on Basler {self.serial}: {exc}") from exc
        self.worker = Thread(target=self.capture, name=f"Basler {self.serial}")
        self.worker.start()

    def capture(self):
        try:
            while not self.stop.is_set() and self.camera.IsGrabbing():
                grab = self.camera.RetrieveResult(GRAB_TIMEOUT_MS, self.pylon.TimeoutHandling_Return)
                received_at = time.monotonic()
                if not grab or not grab.IsValid():
                    continue
                try:
                    skipped = grab.GetNumberOfSkippedImages()
                    with self.lock:
                        self.skipped += skipped
                    if not grab.GrabSucceeded():
                        if self.camera.IsCameraDeviceRemoved():
                            raise RuntimeError(grab.GetErrorDescription())
                        continue
                    image = self.converter.Convert(grab).GetArray().copy()
                finally:
                    grab.Release()
                if image.shape[:2] != (self.geometry["Height"], self.geometry["Width"]):
                    raise RuntimeError(f"Capture dimensions changed to {image.shape[1]}x{image.shape[0]}. Recalibrate.")
                with self.lock:
                    self.received += 1
                    self.overwritten += self.latest is not None
                    self.latest = CapturedFrame(image, received_at)
                    self.last_received = received_at
            if not self.stop.is_set():
                raise RuntimeError("Camera stopped grabbing.")
        except Exception as exc:
            with self.lock:
                self.error = str(exc)

    def take_latest(self):
        with self.lock:
            frame, self.latest = self.latest, None
        return frame

    def close(self):
        self.stop.set()
        if self.camera is None:
            return
        if self.worker is not None and self.worker.ident is not None:
            self.worker.join()
        try:
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()
        finally:
            self.camera.Close()


class LiveFrames:
    """Consume one synchronized batch of recent observations per tracking update."""

    def __init__(self, sources, window_s, clock=time.monotonic, timeout_s=LIVE_TIMEOUT_S):
        self.sources = sources
        self.window_s = window_s
        self.clock = clock
        self.timeout_s = timeout_s
        self.started_at = clock()
        self.last_update = float("-inf")
        self.discarded = dict.fromkeys(sources, 0)

    def check_liveness(self, now):
        received = (source.last_received for source in self.sources.values())
        last_received = max((stamp for stamp in received if stamp is not None), default=self.started_at)
        if now - last_received >= self.timeout_s:
            status = "; ".join(f"{name}: {source.error or 'no frames'}" for name, source in self.sources.items())
            raise RuntimeError(f"No live frames for {self.timeout_s:g} seconds ({status}). Check the cameras and restart.")

    def next(self, after=float("-inf")):
        frames = {name: source.take_latest() for name, source in self.sources.items()}
        frames = {name: frame for name, frame in frames.items() if frame is not None}
        now = self.clock()
        self.check_liveness(now)
        newest = max((frame.received_at for frame in frames.values()), default=float("-inf"))
        batch = {}
        for name, frame in frames.items():
            stamp = frame.received_at
            if stamp > max(self.last_update, after) and stamp >= now - self.window_s and stamp >= newest - self.window_s:
                batch[name] = frame
            else:
                self.discarded[name] += 1
        if batch:
            self.last_update = newest
        return batch
