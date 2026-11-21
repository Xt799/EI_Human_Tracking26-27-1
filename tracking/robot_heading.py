"""Which way the robot faces: a keypoint model finds a front and a rear point on a crop around the robot's box.

The crop is a square of CROP_SCALE x the box's longer side, centred on the box (the training crops of
dataset/robot_heading/ are cut the same way). The two points are at the same height on the robot, so mapping both
to the floor with the camera's floor map keeps the direction rear -> front exact: a plane parallel to the floor maps
onto it only scaled about the camera's foot point.
"""
import os

import cv2
import numpy as np

from common import SCRIPT_DIR, require_trained_weight

HEADING_MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "training", "runs", "heading", "weights", "last.pt")
CROP_SCALE = 2.0          # crop side: this many times the robot box's longer side
HEADING_IMGSZ = 320
HEADING_CONF = 0.25       # minimum confidence of the heading model


def crop_square(img, box, scale, size=None):
    """Square crop around a box (grey outside the image): side = scale x the box's longer side, resized to size px
    (None: native pixels). Returns the crop and (ox, oy, f): crop px = (frame px - (ox, oy)) * f."""
    x1, y1, x2, y2 = box
    side = scale * max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ox, oy = cx - side / 2, cy - side / 2
    if size is None:
        ox, oy, side = round(ox), round(oy), round(side)
        size = side
    f = size / side
    M = np.array([[f, 0, -ox * f], [0, f, -oy * f]], np.float32)
    out = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(114, 114, 114))
    return out, (ox, oy, f)


def floor_heading(H, front_px, rear_px):
    """Unit vector on the floor (metres) from the rear point to the front point, or None if they coincide."""
    fl = cv2.perspectiveTransform(np.float32([front_px, rear_px]).reshape(-1, 1, 2), H)[:, 0]
    v = fl[0] - fl[1]
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else None


class RobotHeading:
    """The heading model. measure() takes a lens-corrected frame and the robot's box in it."""

    def __init__(self, path=HEADING_MODEL_PATH, conf=HEADING_CONF):
        from ultralytics import YOLO
        self.model = YOLO(require_trained_weight(path))
        self.conf = conf

    def measure(self, frame, box):
        """(front point, rear point) in frame pixels and the model's confidence, or None if it finds no robot."""
        crop, (ox, oy, f) = crop_square(frame, box, CROP_SCALE)
        r = self.model.predict(crop, imgsz=HEADING_IMGSZ, conf=self.conf, verbose=False)[0]
        if len(r.boxes) == 0:
            return None
        k = int(r.boxes.conf.argmax())
        xy = r.keypoints.xy[k].cpu().numpy() / f + (ox, oy)
        return xy[0], xy[1], float(r.boxes.conf[k])
