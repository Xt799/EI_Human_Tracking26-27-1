"""Shared by the tracking scripts: paths, lens correction, people detection, floor maps, keypoint names, room constants, the label drawer."""
import json
import os

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = os.path.join(SCRIPT_DIR, "calibration")
POSE_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "yolo11s-pose.pt")
OBJECT_MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "training", "runs", "yolo26n_e100", "weights", "best.pt")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")   # one position CSV per run, named by the start time

POSE_CONF = 0.25      # minimum confidence for a person box
TRACKER = "bytetrack.yaml"
KPT_CONF = 0.5        # a keypoint below this confidence is treated as not visible
TRACK_TIMEOUT_S = 1.0 # a person or object not seen for this long is forgotten
SMOOTH_HALF_S = 0.065 # smoothed track positions: a measurement counts half after this long (= weight 0.3 per frame at 30 fps)
CAMERA_STEP_PX = 20  # a step this far straight down in the image points towards the camera on the floor

# COCO keypoint indices of the pose model
NOSE, LEFT_EYE, RIGHT_EYE = 0, 1, 2
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE = 11, 12, 15, 16

# Colours (BGR) of the floor origin and axes, in the calibration overlay and on the room map
ORIGIN_COLOUR, X_AXIS_COLOUR, Y_AXIS_COLOUR = (0, 0, 255), (255, 128, 0), (0, 255, 255)


def lens_matrices(width, height):
    """(camera matrix, distortion coefficients, camera matrix of the lens-corrected frame) for frames of the given size."""
    K = np.load(os.path.join(CALIB_DIR, "camera_matrix.npy"))
    D = np.load(os.path.join(CALIB_DIR, "distortion_coefficients.npy"))
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (width, height), 0, (width, height))
    return K, D, new_K


def lens_correction(width, height):
    """Returns a function frame -> lens-corrected frame, for frames of the given size."""
    K, D, new_K = lens_matrices(width, height)
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (width, height), cv2.CV_16SC2)
    return lambda frame: cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)


def camera_pose(H, width, height):
    """Where the camera hangs, from its floor map: ((x, y) on the floor below it, height above the floor), in metres."""
    M = np.linalg.inv(lens_matrices(width, height)[2]) @ np.linalg.inv(H)   # floor point (x, y, 1) -> camera coordinates, up to scale
    M /= (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1])) / 2              # the two floor axes are unit vectors
    if M[2, 2] < 0:                                                           # the floor lies in front of the camera
        M = -M
    R = np.column_stack([M[:, 0], M[:, 1], np.cross(M[:, 0], M[:, 1])])
    centre = -R.T @ M[:, 2]
    return centre[:2], abs(float(centre[2]))


def camera_positions():
    """Names of the floor maps in calibration/: the camera positions we can track from."""
    return sorted(f[len("floor_map_"):-len(".npy")] for f in os.listdir(CALIB_DIR)
                  if f.startswith("floor_map_") and f.endswith(".npy"))


def load_floor_map(camera):
    """The 3x3 matrix of a camera position: pixel on the floor -> metres from the origin."""
    return np.load(os.path.join(CALIB_DIR, f"floor_map_{camera}.npy"))


def load_room(cameras):
    """Room bounds ((x0, x1), (y0, y1)) in metres and the grid pitch for the room map, from the cameras' floor map files.
    Cameras calibrated together share the origin; the bounds are the union over the cameras."""
    infos = []
    for camera in cameras:
        path = os.path.join(CALIB_DIR, f"floor_map_{camera}.json")
        with open(path) as f:
            info = json.load(f)
        if "room_m" not in info:
            raise SystemExit(f"{path} has no room bounds: calibrate the camera again (calibrate.py).")
        infos.append(info)
    rooms = np.array([info["room_m"] for info in infos], float)          # camera, axis, (lo, hi)
    room = tuple((float(rooms[:, k, 0].min()), float(rooms[:, k, 1].max())) for k in range(2))
    return room, float(infos[0].get("grid_m", 0.5))


def towards_camera(H, point_px):
    """Unit vector on the floor (metres) pointing from this point towards the camera: straight down in the image."""
    a = cv2.perspectiveTransform(np.asarray(point_px, np.float32).reshape(1, 1, 2), H)[0, 0]
    b = cv2.perspectiveTransform((np.asarray(point_px, np.float32) + (0, CAMERA_STEP_PX)).reshape(1, 1, 2), H)[0, 0]
    return (b - a) / np.linalg.norm(b - a)


class PeopleDetector:
    """YOLO pose model in tracking mode: every person gets a persistent id."""

    def __init__(self):
        from ultralytics import YOLO      # here, so that calibrate.py runs without the detection packages
        self.model = YOLO(POSE_MODEL_PATH)

    def detect(self, frame):
        """Returns (ids, boxes xyxy, keypoints xy, keypoint confidences), empty arrays if nobody is in view."""
        r = self.model.track(frame, conf=POSE_CONF, persist=True, tracker=TRACKER, verbose=False)[0]
        if len(r.boxes) == 0 or r.boxes.id is None:
            return np.empty(0, int), np.empty((0, 4)), np.empty((0, 17, 2)), np.empty((0, 17))
        return (r.boxes.id.cpu().numpy().astype(int), r.boxes.xyxy.cpu().numpy(),
                r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy())


def put_label(img, text, org, bg=(255, 255, 255)):
    """Black text with a filled background box, bottom-left corner at org."""
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x, y = org
    cv2.rectangle(img, (x, y - th - baseline - 4), (x + tw + 6, y + 4), bg, -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
