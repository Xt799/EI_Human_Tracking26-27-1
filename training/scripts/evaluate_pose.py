"""Evaluate the robot + wheel pose model on a split of the paired dataset: wheel point error and robot heading error.

    uv run python training/scripts/evaluate_pose.py training/runs/yolo26n_pose/weights/best.pt [--split test]

Heading = direction on the floor the robot faces: with all four wheels, perpendicular to the line between the middles of
the right side (wheel_1, wheel_2) and the left side (wheel_3, wheel_4), the sides being ~33 cm apart, pointing to the
front pair (wheel_1, wheel_3) rather than the rear pair (wheel_2, wheel_4), which are only ~13 cm behind; with only one
side known, that side's rear -> front direction (wheel_2 -> wheel_1, wheel_4 -> wheel_3). Wheel points are mapped to the floor
with the camera's floor map (the labelled points agree across the two views as floor points, ~3 cm).
Reference heading of a moment: the labels of both views together on the floor (paired recording with floor maps);
otherwise each image's own labels, on its floor map or, without one, in the image (perspective-distorted, but it still
shows whether the direction is right). Results are reported per room and camera.
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
sys.path.insert(0, str(Path(__file__).parent))
from dataset_splits import load as load_splits, split_of  # noqa: E402

WHEELS = ["wheel_1", "wheel_2", "wheel_3", "wheel_4"]
FRONT, REAR = (0, 2), (1, 3)       # wheel indices: front pair wheel_1 + wheel_3, rear pair wheel_2 + wheel_4


def heading(pts):
    """pts: {wheel index: floor xy}. Unit heading vector or None."""
    if len(pts) == 4:
        # front and rear axles are only ~13 cm apart, the two sides ~33 cm: the axis comes from the sides (perpendicular
        # to the right side -> left side line), only its sign from front vs rear
        side = (pts[2] + pts[3] - pts[0] - pts[1]) / 2
        v = np.array([side[1], -side[0]])
        fr = (pts[0] + pts[2] - pts[1] - pts[3]) / 2
        if v @ fr < 0:
            v = -v
    elif 0 in pts and 1 in pts:
        v = pts[0] - pts[1]
    elif 2 in pts and 3 in pts:
        v = pts[2] - pts[3]
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
    splits = load_splits()

    px_err = {2: [], 1: []}          # visibility -> pixel errors
    errs = {}                        # "<room> <camera>" / "<room> both" -> heading errors in degrees
    found, total = 0, 0
    for rec in splits:
        cams = recs[rec]["cameras"]
        # wheel points to the plane the heading is measured in: the floor map, or the image itself without one
        # (an image angle is distorted by the perspective, but still shows whether the model gets the direction)
        floor = {c: common.load_floor_map(v["floor_map"]) if v.get("floor_map") else None for c, v in cams.items()}
        paired = recs[rec].get("offset_frames") is not None and all(m is not None for m in floor.values())
        stems = sorted({p.stem for p in (ROOT / "dataset" / rec).glob("camera_*/*.json")})
        room = recs[rec]["room"]
        for st in stems:
            ref_pts, pred, own_ref = {}, {}, {}
            for c in cams:
                js = ROOT / "dataset" / rec / c / f"{st}.json"
                if not js.exists() or split_of(splits, rec, c, int(st[1:])) != a.split:
                    continue
                labels = {}
                for s in json.loads(js.read_text(encoding="utf-8"))["shapes"]:
                    if s["label"] in WHEELS:
                        labels[WHEELS.index(s["label"])] = (np.array(s["points"][0]), 1 if s.get("difficult") else 2)
                if not labels:
                    continue
                to_plane = (lambda q, H=floor[c]: cv2.perspectiveTransform(np.float32(q).reshape(1, 1, 2), H)[0, 0]) if floor[c] is not None else (lambda q: np.float32(q))
                for i, (p, _) in labels.items():
                    ref_pts.setdefault(i, []).append(to_plane(p))
                own_ref[c] = heading({i: to_plane(p) for i, (p, _) in labels.items()})
                total += 1
                res = predict(cv2.imread(str(js.with_suffix(".jpg"))))
                if res is None:
                    continue
                kxy, kc = res
                found += 1
                for i, (p, v) in labels.items():
                    px_err[v].append(float(np.linalg.norm(kxy[i] - p)))
                pred[c] = (heading({i: to_plane(kxy[i]) for i in range(4)}), float(kc.mean()))
            # paired: both views together are the reference on the shared floor; otherwise each view's own labels
            ref = heading({i: np.mean(v, 0) for i, v in ref_pts.items()}) if paired else None
            for c, (h, _) in pred.items():
                r = ref if paired else own_ref.get(c)
                if h is not None and r is not None:
                    errs.setdefault(f"{room} {c}", []).append(angle(h, r))
            vs = [(h, w) for h, w in pred.values() if h is not None]
            if paired and ref is not None and vs:
                v = sum(h * w for h, w in vs)
                if np.linalg.norm(v) > 1e-6:
                    errs.setdefault(f"{room} both", []).append(angle(v / np.linalg.norm(v), ref))

    print(f"split {a.split}: robot found in {found}/{total} labelled images")
    for v, name in ((2, "seen"), (1, "hidden")):
        e = np.array(px_err[v])
        if len(e):
            print(f"wheel points {name}: n={len(e)}, median {np.median(e):.1f} px, 90% {np.percentile(e, 90):.1f} px")
    for k, e in sorted(errs.items()):
        e = np.array(e)
        if len(e):
            print(f"heading {k:16s}: n={len(e)}, median {np.median(e):.1f} deg, mean {e.mean():.1f} deg, "
                  f"within 15 deg {np.mean(e <= 15):.0%}, flipped (>90 deg) {np.mean(e > 90):.0%}")


if __name__ == "__main__":
    main()
