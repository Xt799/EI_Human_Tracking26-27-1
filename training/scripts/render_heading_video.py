"""Render a check video of the robot heading model: the raw per-frame heading, no smoothing.

    uv run python training/scripts/render_heading_video.py videos/room_2/camera_1/basler_record_20261110_170937.mp4 --split val

Each frame (lens-corrected): the robot model's box -> square crop (CROP_SCALE x the box) -> heading model -> front and
rear point -> arrow rear -> front on the frame and on a zoomed crop. A strip below plots the heading angle in the image
over the last seconds, frame by frame, so jitter and jumps show directly.
--split: only the frames of that split of dataset/splits.yaml (val = not trained on, except by the final all-data model). Output: results/heading_<name>.mp4,
and the frame-to-frame change of the heading (how smooth it is) printed at the end.
"""
import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from dataset import CROP_SCALE, HEADING_MODEL, ROOT, crop_square, heading_deg, load_splits, ranges, recordings
import common

VIEW_W, ZOOM, PLOT_H, PLOT_S = 1200, 400, 220, 10.0       # frame view width, zoom side, plot height, plotted seconds


def predict(detector, model, img, conf):
    """(front, rear, box, keypoint conf) in frame px, or None."""
    d = detector.predict(img, imgsz=1920, conf=0.3, verbose=False)[0]
    boxes = [(c, b) for b, c, k in zip(d.boxes.xyxy.cpu().numpy(), d.boxes.conf.cpu().numpy(), d.boxes.cls.cpu().numpy())
             if detector.names[int(k)] == "robot"]
    if not boxes:
        return None
    box = max(boxes, key=lambda x: x[0])[1]
    crop, (ox, oy, f) = crop_square(img, box, CROP_SCALE)
    r = model.predict(crop, imgsz=320, conf=conf, verbose=False)[0]
    if len(r.boxes) == 0:
        return None
    k = int(r.boxes.conf.argmax())
    xy = r.keypoints.xy[k].cpu().numpy() / f + (ox, oy)
    return xy[0], xy[1], box, float(r.boxes.conf[k])


def arrow(img, front, rear, s, thick):
    f, r = np.array(front) * s, np.array(rear) * s
    tip = f + 0.6 * (f - r)
    for colour, t in (((0, 0, 0), thick + 2), ((0, 255, 0), thick)):
        cv2.arrowedLine(img, tuple(map(int, r)), tuple(map(int, tip)), colour, t, cv2.LINE_AA, tipLength=0.3)
    cv2.circle(img, tuple(map(int, f)), thick + 2, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(img, tuple(map(int, r)), thick + 2, (255, 0, 0), -1, cv2.LINE_AA)


def plot(history, width, fps):
    """Heading angle over the last PLOT_S seconds: one dot per frame, gaps where nothing was found."""
    p = np.full((PLOT_H, width, 3), 255, np.uint8)
    y_of = lambda a: int(10 + (180 - a) / 360 * (PLOT_H - 20))
    for a in (-180, -90, 0, 90, 180):
        cv2.line(p, (40, y_of(a)), (width - 5, y_of(a)), (220, 220, 220), 1)
        cv2.putText(p, f"{a}", (2, y_of(a) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1, cv2.LINE_AA)
    n = int(PLOT_S * fps)
    for i, a in enumerate(history):
        if a is not None:
            x = int(40 + (i + n - len(history)) / n * (width - 45))
            cv2.circle(p, (x, y_of(a)), 2, (0, 140, 0), -1)
    cv2.putText(p, f"heading in the image (deg), last {PLOT_S:.0f} s, raw per frame", (45, PLOT_H - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--weights", default=str(HEADING_MODEL.relative_to(ROOT)))
    ap.add_argument("--split", help="only this split's frames (dataset/splits.yaml)")
    ap.add_argument("--seconds", type=float, nargs=2)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out")
    a = ap.parse_args()
    from ultralytics import YOLO
    detector, model = YOLO(common.OBJECT_MODEL_PATH), YOLO(str(ROOT / a.weights))
    cap = cv2.VideoCapture(str(ROOT / a.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fix = common.lens_correction(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    stretches = [(0, n - 1)]
    if a.split:
        recs = recordings()
        rec, cam = next((r, c) for r, v in recs.items() for c, cv in v["cameras"].items() if Path(cv["video"]).name == Path(a.video).name)
        stretches = ranges(load_splits(), rec, cam)[a.split]
    if a.seconds:
        stretches = [(max(lo, int(a.seconds[0] * fps)), min(hi, int(a.seconds[1] * fps))) for lo, hi in stretches]
    out_path = Path(a.out) if a.out else ROOT / "results" / f"heading_{Path(a.video).stem}{'_' + a.split if a.split else ''}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer, history, found, steps = None, deque(maxlen=int(PLOT_S * fps)), 0, []
    total = sum(hi - lo + 1 for lo, hi in stretches)
    for lo, hi in stretches:
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        history.clear()
        for f in range(lo, hi + 1):
            ok, img = cap.read()
            if not ok:
                break
            img = fix(img)
            res = predict(detector, model, img, a.conf)
            s = VIEW_W / img.shape[1]
            view = cv2.resize(img, (VIEW_W, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
            zoom = np.full((ZOOM, ZOOM, 3), 40, np.uint8)
            if res is not None:
                front, rear, box, c = res
                found += 1
                if history and history[-1] is not None:
                    steps.append(abs((heading_deg(front, rear) - history[-1] + 180) % 360 - 180))
                history.append(heading_deg(front, rear))
                arrow(view, front, rear, s, 2)
                zoom, (ox, oy, zf) = crop_square(img, box, CROP_SCALE, ZOOM)
                arrow(zoom, (front - (ox, oy)) * zf, (rear - (ox, oy)) * zf, 1.0, 2)
                common.put_label(zoom, f"{history[-1]:.0f} deg, conf {c:.2f}", (8, ZOOM - 10))
            else:
                history.append(None)
                common.put_label(zoom, "no robot", (8, ZOOM - 10))
            common.put_label(view, f"frame {f}  t={f / fps:6.1f}s" + (f"  split: {a.split}" if a.split else ""), (8, 22))
            right = np.vstack([zoom, np.full((view.shape[0] - ZOOM, ZOOM, 3), 30, np.uint8)])
            canvas = np.vstack([np.hstack([view, right]), plot(history, VIEW_W + ZOOM, fps)])
            if writer is None:
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas.shape[1], canvas.shape[0]))
            writer.write(canvas)
    writer.release()
    print(f"robot heading in {found}/{total} frames -> {out_path}")
    d = np.array(steps)
    if len(d):
        print(f"change between consecutive frames: median {np.median(d):.1f} deg, 90% {np.percentile(d, 90):.1f} deg, "
              f"jumps > 30 deg {np.mean(d > 30):.1%}, > 90 deg {np.mean(d > 90):.1%} ({len(d)} frame pairs)")


if __name__ == "__main__":
    main()
