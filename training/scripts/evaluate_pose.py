"""Evaluate the robot + wheel pose model on a split of the paired dataset: wheel point error and robot heading error.

    uv run python training/scripts/evaluate_pose.py training/runs/yolo26n_pose/weights/best.pt [--split test]

Heading = direction on the floor from the middle of the rear wheels (wheel_3, wheel_4) to the middle of the front wheels
(wheel_1, wheel_2); with only one side known, that side's rear -> front direction. Wheel points are mapped to the floor
with the camera's floor map (the labelled points agree across the two views as floor points, ~3 cm).
Reference heading of a moment: the labels of both views together on the floor.
Predicted heading: per camera from its 4 predicted points, and both cameras together (unit vectors weighted by the
points' confidence).
--crop: two-stage, as at run time: the robot model finds the box, the pose model runs on the square crop around it
(build_pose_crops.py), and the points are mapped back to the full image.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tracking"))
import common  # noqa: E402

WHEELS = ["wheel_1", "wheel_2", "wheel_3", "wheel_4"]
CAMS = ["camera_1", "camera_2"]


def heading(pts):
    """pts: {wheel index: floor xy}. Unit heading vector or None."""
    front = [pts[i] for i in (0, 1) if i in pts]
    rear = [pts[i] for i in (2, 3) if i in pts]
    if len(front) == 2 and len(rear) == 2:
        v = np.mean(front, 0) - np.mean(rear, 0)
    elif 0 in pts and 2 in pts:
        v = pts[0] - pts[2]
    elif 1 in pts and 3 in pts:
        v = pts[1] - pts[3]
    else:
        return None
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else None


def angle(u, v):
    return float(np.degrees(np.arccos(np.clip(u @ v, -1, 1))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--crop", action="store_true", help="two-stage: robot model box -> crop -> pose model")
    ap.add_argument("--crop-size", type=int, default=320)
    ap.add_argument("--crop-scale", type=float, default=1.5)
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(a.weights)
    if a.crop:
        sys.path.insert(0, str(Path(__file__).parent))
        from build_pose_crops import crop_square
        detector = YOLO(common.OBJECT_MODEL_PATH)

    def predict(img):
        """(keypoints xy in image px (4, 2), keypoint confidences (4,)) of the most confident robot, or None."""
        if not a.crop:
            r = model.predict(img, imgsz=a.imgsz, conf=a.conf, verbose=False)[0]
            if len(r.boxes) == 0:
                return None
            k = int(r.boxes.conf.argmax())
            return r.keypoints.xy[k].cpu().numpy(), r.keypoints.conf[k].cpu().numpy()
        d = detector.predict(img, imgsz=1920, conf=0.3, verbose=False)[0]
        boxes = [(c, b) for b, c, k in zip(d.boxes.xyxy.cpu().numpy(), d.boxes.conf.cpu().numpy(), d.boxes.cls.cpu().numpy())
                 if detector.names[int(k)] == "robot"]
        if not boxes:
            return None
        crop, (ox, oy, f) = crop_square(img, max(boxes, key=lambda x: x[0])[1], a.crop_scale, a.crop_size)
        r = model.predict(crop, imgsz=a.crop_size, conf=0.05, verbose=False)[0]
        if len(r.boxes) == 0:
            return None
        k = int(r.boxes.conf.argmax())
        return r.keypoints.xy[k].cpu().numpy() / f + (ox, oy), r.keypoints.conf[k].cpu().numpy()
    recs = yaml.safe_load((ROOT / "dataset" / "recordings.yaml").read_text())["recordings"]
    splits = yaml.safe_load((ROOT / "dataset" / "splits.yaml").read_text())

    px_err = {2: [], 1: []}          # visibility -> pixel errors
    errs = {"camera_1": [], "camera_2": [], "both": []}
    found, total = 0, 0
    for rec, parts in splits.items():
        lo, hi = parts[a.split]
        floor = {c: common.load_floor_map(recs[rec]["cameras"][c]["floor_map"]) for c in CAMS}
        stems = sorted({p.stem for p in (ROOT / "dataset" / rec).glob("camera_*/*.json") if lo <= int(p.stem[1:]) <= hi})
        for st in stems:
            ref_floor, pred = {}, {}
            for c in CAMS:
                js = ROOT / "dataset" / rec / c / f"{st}.json"
                labels = {}
                if js.exists():
                    for s in json.loads(js.read_text(encoding="utf-8"))["shapes"]:
                        if s["label"] in WHEELS:
                            labels[WHEELS.index(s["label"])] = (np.array(s["points"][0]), 1 if s.get("difficult") else 2)
                for i, (p, _) in labels.items():
                    ref_floor.setdefault(i, []).append(cv2.perspectiveTransform(p.reshape(1, 1, 2).astype(np.float32), floor[c])[0, 0])
                img = cv2.imread(str(ROOT / "dataset" / rec / c / f"{st}.jpg"))
                res = predict(img)
                if labels:
                    total += 1
                if res is None:
                    continue
                kxy, kc = res
                if labels:
                    found += 1
                    for i, (p, v) in labels.items():
                        px_err[v].append(float(np.linalg.norm(kxy[i] - p)))
                fl = {i: cv2.perspectiveTransform(kxy[i].reshape(1, 1, 2).astype(np.float32), floor[c])[0, 0] for i in range(4)}
                pred[c] = (heading(fl), float(kc.mean()))
            ref = heading({i: np.mean(v, 0) for i, v in ref_floor.items()})
            if ref is None:
                continue
            for c in CAMS:
                if c in pred and pred[c][0] is not None:
                    errs[c].append(angle(pred[c][0], ref))
            vs = [(h, w) for h, w in pred.values() if h is not None]
            if vs:
                s = sum(h * w for h, w in vs)
                if np.linalg.norm(s) > 1e-6:
                    errs["both"].append(angle(s / np.linalg.norm(s), ref))

    print(f"split {a.split}: robot found in {found}/{total} labelled images")
    for v, name in ((2, "seen"), (1, "hidden")):
        e = np.array(px_err[v])
        if len(e):
            print(f"wheel points {name}: n={len(e)}, median {np.median(e):.1f} px, 90% {np.percentile(e, 90):.1f} px")
    for k, e in errs.items():
        e = np.array(e)
        if len(e):
            print(f"heading {k:9s}: n={len(e)}, median {np.median(e):.1f} deg, mean {e.mean():.1f} deg, "
                  f"within 15 deg {np.mean(e <= 15):.0%}, flipped (>90 deg) {np.mean(e > 90):.0%}")


if __name__ == "__main__":
    main()
