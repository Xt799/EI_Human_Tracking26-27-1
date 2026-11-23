"""Robot crops of the dataset: cutting them out, and their file names (which carry where the crop is in the frame)."""
import re
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HEADING_DIR = ROOT / "dataset" / "robot_heading"   # the crops: train/ and val/
HEADING_MODEL = ROOT / "training" / "runs" / "heading" / "weights" / "last.pt"   # front/rear model, trained on all crops
CROP_SCALE = 2.0          # stored crop: square of this many times the robot box's longer side, centred on the box
NAME = re.compile(r"(?P<rec>.+)_(?P<cam>camera_\d+)_t(?P<frame>\d+)(_x(?P<x>-?\d+)_y(?P<y>-?\d+))?$")


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


def frame_name(recording, camera, frame):
    """File name (no extension) of a full frame: <recording>_<camera>_t<frame>."""
    return f"{recording}_{camera}_t{frame:06d}"


def crop_name(recording, camera, frame, ox, oy):
    """File name of a crop: the frame's name, then the crop's top-left corner in the lens-corrected frame."""
    return f"{frame_name(recording, camera, frame)}_x{int(ox)}_y{int(oy)}"


def parse_name(stem):
    """(recording, camera, frame, ox, oy) from a frame or crop file name (ox, oy are 0 for a full frame)."""
    m = NAME.match(stem)
    return m["rec"], m["cam"], int(m["frame"]), int(m["x"] or 0), int(m["y"] or 0)


def heading_deg(front, rear):
    """Image angle of rear -> front in degrees (0 = right, 90 = down)."""
    return float(np.degrees(np.arctan2(front[1] - rear[1], front[0] - rear[0])))
