"""Which split (train / val / test) an image of dataset/ belongs to, from dataset/splits.yaml."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load():
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
