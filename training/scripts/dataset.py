"""Shared by the dataset and training scripts: where things are, the train / val split of dataset/splits.yaml and the
file names of our images (which carry the recording, camera, frame and, for a crop, where it is in the frame).

The robot crop itself (CROP_SCALE, crop_square) and the heading model path come from tracking/robot_heading.py, so
training and tracking cut the crop the same way.
"""
import re
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tracking"))
from robot_heading import CROP_SCALE, HEADING_MODEL_PATH, crop_square  # noqa: E402,F401  (re-exported)

HEADING_DIR = ROOT / "dataset" / "robot_heading"   # the crops: train/ and val/
DETECTION_DIR = ROOT / "dataset" / "detection"     # full frames: train/ and val/, one folder per source
HEADING_MODEL = Path(HEADING_MODEL_PATH).resolve()
NAME = re.compile(r"(?P<rec>.+)_(?P<cam>camera_\d+)_t(?P<frame>\d+)(_x(?P<x>-?\d+)_y(?P<y>-?\d+))?$")


# ---- Recordings and splits ----
def find_video(listed):
    """Path of a video listed in dataset/recordings.yaml (e.g. videos/room_1/camera_1/<name>.mp4): there if it exists,
    otherwise the file of the same name anywhere under videos/, so the recordings may be sorted any way."""
    if (ROOT / listed).exists():
        return str(ROOT / listed)
    name = Path(listed).name
    for path in (ROOT / "videos").rglob(name):
        return str(path)
    raise SystemExit(f"Video {name} not found under {ROOT / 'videos'} (any sub-folder): copy it there.")


def recordings():
    """The recordings of dataset/recordings.yaml: {name: {room, cameras: {camera: {video, floor_map}}, offset_frames, ...}}."""
    return yaml.safe_load((ROOT / "dataset" / "recordings.yaml").read_text())["recordings"]


def load_splits():
    return yaml.safe_load((ROOT / "dataset" / "splits.yaml").read_text())


def ranges(splits, recording, camera="camera_1"):
    """{split: [(first, last), ...]} of one camera of a recording. A recording's ranges apply to both cameras (paired:
    the camera_1 timeline), unless the recording lists them per camera (unpaired cameras: each camera's own frame
    numbers). A split is one [first, last] range or a list of them."""
    parts = splits.get(recording) or {}
    if camera in parts:
        parts = parts[camera]
    return {k: [tuple(r) for r in v] if isinstance(v[0], list) else [tuple(v)] for k, v in parts.items() if v}


def split_of(splits, recording, camera, frame):
    """Split name of an image, or None."""
    for name, rs in ranges(splits, recording, camera).items():
        if any(lo <= frame <= hi for lo, hi in rs):
            return name
    return None


# ---- File names ----
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


# ---- Labels ----
def heading_deg(front, rear):
    """Image angle of rear -> front in degrees (0 = right, 90 = down)."""
    return float(np.degrees(np.arctan2(front[1] - rear[1], front[0] - rear[0])))


def rect(label, b):
    """An X-AnyLabeling rectangle shape from a box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = (float(v) for v in b)
    return {"label": label, "score": None, "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], "group_id": None,
            "description": "", "difficult": False, "shape_type": "rectangle", "flags": {}, "attributes": {},
            "kie_linking": []}
