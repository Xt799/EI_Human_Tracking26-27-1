"""Paths, lens correction and people detection shared by the tracking scripts."""
import os

import cv2
import numpy as np
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = os.path.join(SCRIPT_DIR, "calibration")
POSE_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "yolo11s-pose.pt")
OBJECT_MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "training", "runs", "yolo26n_e30", "weights", "best.pt")

POSE_CONF = 0.25      # minimum confidence for a person box
TRACKER = "bytetrack.yaml"


def lens_correction(width, height):
    """Returns a function frame -> lens-corrected frame, for frames of the given size."""
    K = np.load(os.path.join(CALIB_DIR, "camera_matrix.npy"))
    D = np.load(os.path.join(CALIB_DIR, "distortion_coefficients.npy"))
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (width, height), 0, (width, height))
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (width, height), cv2.CV_16SC2)
    return lambda frame: cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)


def camera_positions():
    """Names of the floor maps in calibration/: the camera positions we can track from."""
    return sorted(f[len("floor_map_"):-len(".npy")] for f in os.listdir(CALIB_DIR)
                  if f.startswith("floor_map_") and f.endswith(".npy"))


def load_floor_map(camera):
    return np.load(os.path.join(CALIB_DIR, f"floor_map_{camera}.npy"))


class PeopleDetector:
    """YOLO pose model in tracking mode: every person gets a persistent id."""

    def __init__(self):
        self.model = YOLO(POSE_MODEL_PATH)

    def detect(self, frame):
        """Returns (ids, boxes xyxy, keypoints xy, keypoint confidences), empty arrays if nobody is in view."""
        r = self.model.track(frame, conf=POSE_CONF, persist=True, tracker=TRACKER, verbose=False)[0]
        if len(r.boxes) == 0 or r.boxes.id is None:
            return np.empty(0, int), np.empty((0, 4)), np.empty((0, 17, 2)), np.empty((0, 17))
        return (r.boxes.id.cpu().numpy().astype(int), r.boxes.xyxy.cpu().numpy(),
                r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy())


def put_label(img, text, org, bg=(255, 255, 255), fg=(0, 0, 0), scale=0.6):
    """Text with a filled background box, bottom-left corner at org."""
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = org
    cv2.rectangle(img, (x, y - th - baseline - 4), (x + tw + 6, y + 4), bg, -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 2)
