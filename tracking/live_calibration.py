"""Local calibration profiles for serial-selected live cameras."""
import hashlib
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import numpy as np

from common import CALIB_DIR

LIVE_CALIB_DIR = Path(CALIB_DIR) / "live"
INTRINSICS = ("camera_matrix.npy", "distortion_coefficients.npy")


def validate_pairs(pairs):
    serials, views = zip(*pairs)
    if len(set(serials)) != len(serials) or len(set(map(str.casefold, views))) != len(views):
        raise ValueError("Use each serial and view only once.")
    for view in views:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", view) or view.casefold() == "room":
            raise ValueError(f"Invalid view {view!r}. Use letters, digits, underscores or dashes, and reserve 'Room' for the map.")


def intrinsics(view, geometry):
    directory = LIVE_CALIB_DIR / "intrinsics" / view
    if not directory.exists():
        if (geometry["Width"], geometry["Height"]) != (1900, 1200):
            raise ValueError(f"{view}: supply intrinsics for this geometry in {directory}.")
        directory = Path(CALIB_DIR)
    try:
        fingerprints = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in INTRINSICS}
    except OSError as exc:
        raise ValueError(f"{view}: supply both intrinsics files in {directory}: {exc}") from exc
    return directory, fingerprints


def read_profiles():
    path = LIVE_CALIB_DIR / "profiles.json"
    try:
        profiles = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read live calibration profiles at {path}: {exc}") from exc
    if not isinstance(profiles, dict):
        raise ValueError(f"Invalid live calibration profiles at {path}. Expected a map of view names to profiles.")
    return profiles


def save_profiles(profiles):
    LIVE_CALIB_DIR.mkdir(parents=True, exist_ok=True)
    path = LIVE_CALIB_DIR / "profiles.json"
    data = json.dumps(read_profiles() | profiles, indent=2, allow_nan=False)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=LIVE_CALIB_DIR, suffix=".tmp", delete=False) as temp:
        temp_path = Path(temp.name)
        try:
            temp.write(data)
            temp.close()
            temp_path.replace(path)
        finally:
            temp.close()
            temp_path.unlink(missing_ok=True)
    return path


def load_live_profiles(sources):
    saved = read_profiles()
    profiles = {}
    for view, source in sources.items():
        try:
            if view not in saved:
                raise ValueError("no saved live profile")
            data = saved[view]
            if data["serial"] != source.serial or data["geometry"] != source.geometry:
                raise ValueError("serial or image geometry differs from calibration")
            directory, fingerprints = intrinsics(view, source.geometry)
            if data["intrinsics"] != fingerprints:
                raise ValueError("lens intrinsics have changed")
            matrix = np.asarray(data["homography"], dtype=np.float64)
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
                raise ValueError("invalid homography")
            room = np.asarray(data["room_m"], dtype=float)
            grid = float(data["grid_m"])
            if (room.shape != (2, 2) or not np.isfinite(room).all() or np.any(room[:, 0] >= room[:, 1])
                    or not np.isfinite(grid) or grid <= 0 or not data["room_id"]):
                raise ValueError("invalid room calibration")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{view}: {exc}. Calibrate the live cameras together with tracking/calibrate.py.") from exc
        profiles[view] = (matrix, directory)
        if len(profiles) == 1:
            room_id, room_m, grid_m = data["room_id"], room, grid
        elif data["room_id"] != room_id or not np.array_equal(room, room_m) or grid != grid_m:
            raise ValueError("Live views do not share a calibration session. Calibrate the requested cameras together.")
    return room_m, grid_m, profiles
