"""Render a check video of the robot heading model on a paired recording: both cameras and a top-down floor view.

    uv run python training/scripts/render_heading_paired.py room_1_20261012_1800 [--weights ...] [--seconds 0 60]

Per camera (lens-corrected frame): robot model box -> square crop -> heading model -> front and rear point, drawn on
the frame and on a zoomed crop. Both points are mapped to the floor with the camera's floor map. They are at the same
height on the robot, so the floor direction rear -> front is exact (a plane parallel to the floor maps onto it scaled
about the camera's foot point); the position is shifted a little away from the camera by that height.
Floor view: each camera's heading (camera_1 green, camera_2 purple) at its own position and both together (black,
weighted by the model's confidence). A strip plots the three floor angles over the last seconds, raw per frame.
Printed at the end: how far the two cameras disagree, and how much the heading changes from frame to frame.
Output: results/heading_<recording>.mp4
"""
import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tracking"))
sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402
from render_heading_video import arrow, predict  # noqa: E402
from robot_crops import CROP_SCALE, HEADING_MODEL, crop_square  # noqa: E402

CAMS = ("camera_1", "camera_2")
COLOURS = {"camera_1": (0, 170, 0), "camera_2": (200, 0, 200), "both": (0, 0, 0)}
VIEW_W, ZOOM, PLOT_H, PLOT_S = 950, 300, 200, 10.0


def floor_angle(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


def adiff(a, b):
    return abs((a - b + 180) % 360 - 180)


def map_panel(width, height, room, est):
    """est: {name: (position (2,), unit direction (2,))} in metres."""
    (x0, x1), (y0, y1) = room
    s = min((width - 40) / (x1 - x0), (height - 60) / (y1 - y0))
    mx, my = (width - (x1 - x0) * s) / 2, height - 20 - (height - 40 - (y1 - y0) * s) / 2
    to_px = lambda q: (int(mx + (q[0] - x0) * s), int(my - (q[1] - y0) * s))
    p = np.full((height, width, 3), 255, np.uint8)
    cv2.rectangle(p, to_px((x0, y0)), to_px((x1, y1)), (180, 180, 180), 1)
    for name, (pos, u) in est.items():
        length = 0.8 if name == "both" else 0.5
        cv2.arrowedLine(p, to_px(pos), to_px(pos + length * u), COLOURS[name], 3 if name == "both" else 2, cv2.LINE_AA, tipLength=0.3)
    if "camera_1" in est and "camera_2" in est:
        d = adiff(floor_angle(est["camera_1"][1]), floor_angle(est["camera_2"][1]))
        common.put_label(p, f"cameras disagree by {d:.0f} deg", (8, height - 10))
    common.put_label(p, "floor: camera_1 green, camera_2 purple, both black", (8, 22))
    return p


def plot(hist, width, fps):
    p = np.full((PLOT_H, width, 3), 255, np.uint8)
    y_of = lambda a: int(10 + (180 - a) / 360 * (PLOT_H - 20))
    for a in (-180, -90, 0, 90, 180):
        cv2.line(p, (40, y_of(a)), (width - 5, y_of(a)), (220, 220, 220), 1)
        cv2.putText(p, f"{a}", (2, y_of(a) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1, cv2.LINE_AA)
    n = int(PLOT_S * fps)
    for name in ("camera_1", "camera_2", "both"):
        h = hist[name]
        for i, a in enumerate(h):
            if a is not None:
                x = int(40 + (i + n - len(h)) / n * (width - 45))
                cv2.circle(p, (x, y_of(a)), 2 if name == "both" else 1, COLOURS[name], -1)
    cv2.putText(p, f"heading on the floor (deg), last {PLOT_S:.0f} s, raw per frame", (45, PLOT_H - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("--weights", default=str(HEADING_MODEL.relative_to(ROOT)))
    ap.add_argument("--seconds", type=float, nargs=2)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out")
    a = ap.parse_args()
    from ultralytics import YOLO
    detector, model = YOLO(common.OBJECT_MODEL_PATH), YOLO(str(ROOT / a.weights))
    rec = yaml.safe_load((ROOT / "dataset" / "recordings.yaml").read_text())["recordings"][a.recording]
    lo, hi = rec["paired_frames"]
    fps = 30.0
    if a.seconds:
        lo, hi = max(lo, int(a.seconds[0] * fps)), min(hi, int(a.seconds[1] * fps))
    off = rec["offset_frames"]
    caps, fixes, floors = {}, {}, {}
    for c in CAMS:
        info = rec["cameras"][c]
        caps[c] = cv2.VideoCapture(str(ROOT / info["video"]))
        caps[c].set(cv2.CAP_PROP_POS_FRAMES, lo - (off if c == "camera_2" else 0))
        fixes[c] = common.lens_correction(int(caps[c].get(cv2.CAP_PROP_FRAME_WIDTH)), int(caps[c].get(cv2.CAP_PROP_FRAME_HEIGHT)))
        floors[c] = common.load_floor_map(info["floor_map"])
    room, _ = common.load_room([rec["cameras"][c]["floor_map"] for c in CAMS])
    out_path = Path(a.out) if a.out else ROOT / "results" / f"heading_{a.recording}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hist = {k: deque(maxlen=int(PLOT_S * fps)) for k in ("camera_1", "camera_2", "both")}
    disagree, steps, seen = [], {k: [] for k in hist}, {k: 0 for k in hist}
    writer = None
    for frame in range(lo, hi + 1):
        views, zooms, est, weights = [], [], {}, {}
        for c in CAMS:
            ok, img = caps[c].read()
            if not ok:
                break
            img = fixes[c](img)
            res = predict(detector, model, img, a.conf)
            s = VIEW_W / img.shape[1]
            view = cv2.resize(img, (VIEW_W, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
            zoom = np.full((ZOOM, ZOOM, 3), 40, np.uint8)
            if res is not None:
                front, rear, box, conf = res
                arrow(view, front, rear, s, 2)
                zoom, (ox, oy, zf) = crop_square(img, box, CROP_SCALE, ZOOM)
                arrow(zoom, (front - (ox, oy)) * zf, (rear - (ox, oy)) * zf, 1.0, 2)
                fl = cv2.perspectiveTransform(np.float32([front, rear]).reshape(-1, 1, 2), floors[c])[:, 0]
                v = fl[0] - fl[1]
                if np.linalg.norm(v) > 1e-6:
                    est[c] = (fl.mean(0), v / np.linalg.norm(v))
                    weights[c] = conf
            common.put_label(view, c, (8, 22))
            common.put_label(zoom, c + ("" if res is not None else ": no robot"), (8, ZOOM - 10))
            views.append(view)
            zooms.append(zoom)
        if len(views) < 2:
            break
        if est:
            v = sum(est[c][1] * weights[c] for c in est)
            if np.linalg.norm(v) > 1e-6:
                est["both"] = (np.mean([e[0] for e in est.values()], 0), v / np.linalg.norm(v))
        if "camera_1" in est and "camera_2" in est:
            disagree.append(adiff(floor_angle(est["camera_1"][1]), floor_angle(est["camera_2"][1])))
        for k in hist:
            ang = floor_angle(est[k][1]) if k in est else None
            if ang is not None:
                seen[k] += 1
                if hist[k] and hist[k][-1] is not None:
                    steps[k].append(adiff(ang, hist[k][-1]))
            hist[k].append(ang)
        top = np.hstack(views)
        W = top.shape[1]
        bottom = np.hstack(zooms + [map_panel(W - 2 * ZOOM, ZOOM, room, est)])
        canvas = np.vstack([top, bottom, plot(hist, W, fps)])
        common.put_label(canvas, f"frame {frame}  t={frame / fps:6.1f}s", (W - 260, 22))
        if writer is None:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas.shape[1], canvas.shape[0]))
        writer.write(canvas)
    writer.release()
    total = hi - lo + 1
    print(f"-> {out_path}")
    print("robot heading found: " + ", ".join(f"{k} {seen[k]}/{total}" for k in hist))
    d = np.array(disagree)
    if len(d):
        print(f"camera_1 vs camera_2 on the floor ({len(d)} frames with both): median {np.median(d):.1f} deg, "
              f"90% {np.percentile(d, 90):.1f} deg, within 10 deg {np.mean(d <= 10):.0%}, within 20 deg {np.mean(d <= 20):.0%}, "
              f"opposite (> 90 deg) {np.mean(d > 90):.1%}")
    for k, st in steps.items():
        st = np.array(st)
        if len(st):
            print(f"{k} frame-to-frame change: median {np.median(st):.1f} deg, 90% {np.percentile(st, 90):.1f} deg, "
                  f"jumps > 30 deg {np.mean(st > 30):.1%}")


if __name__ == "__main__":
    main()
