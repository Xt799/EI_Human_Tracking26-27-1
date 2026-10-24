"""Evaluate detectors on a hand-corrected X-AnyLabeling set (dataset/test by default).

uv run python training/scripts/evaluate.py WEIGHTS [WEIGHTS ...] [--set test] [--imgsz 1920] [--conf 0.5]

Per weights it prints
  * mAP50 / mAP50-95 per class (ultralytics val, low confidence threshold),
  * frame-level numbers at the operating threshold --conf (what live_view.py shows): TP/FP/FN, precision, recall (IoU >= 0.5),
  * frames where more than one robot / ball is predicted (there is always at most one of each).
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent.parent
CLASSES = ["robot", "ball"]


def load(folder: Path):
    """-> list of (image path, {cls: [box xyxy]}, w, h)"""
    items = []
    for js in sorted(folder.glob("frame_*.json")):
        d = json.loads(js.read_text())
        gt = {c: [] for c in CLASSES}
        for s in d["shapes"]:
            xs, ys = [p[0] for p in s["points"]], [p[1] for p in s["points"]]
            gt[s["label"]].append([min(xs), min(ys), max(xs), max(ys)])
        items.append((folder / d["imagePath"], gt, d["imageWidth"], d["imageHeight"]))
    return items


def iou(a, b):
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def yolo_dataset(items, name):
    out = ROOT / "training" / "ds_eval" / name
    shutil.rmtree(out, ignore_errors=True)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)
    for img, gt, w, h in items:
        shutil.copy(img, out / "images" / img.name)
        lines = [f"{CLASSES.index(c)} {(b[0] + b[2]) / 2 / w:.6f} {(b[1] + b[3]) / 2 / h:.6f} {(b[2] - b[0]) / w:.6f} {(b[3] - b[1]) / h:.6f}"
                 for c in CLASSES for b in gt[c]]
        (out / "labels" / (img.stem + ".txt")).write_text("\n".join(lines))
    (out / "data.yaml").write_text(f"path: {out.as_posix()}\ntrain: images\nval: images\nnames:\n  0: robot\n  1: ball\n")
    return out / "data.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="+", type=Path)
    ap.add_argument("--set", default="test", help="folder under dataset/ (test or val)")
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.45)
    a = ap.parse_args()

    items = load(ROOT / "dataset" / a.set)
    data = yolo_dataset(items, a.set)
    n_obj = {c: sum(len(gt[c]) for _, gt, _, _ in items) for c in CLASSES}
    print(f"set: dataset/{a.set}, {len(items)} frames, {n_obj['robot']} robots, {n_obj['ball']} balls; imgsz {a.imgsz}, conf {a.conf}\n")
    for wpath in a.weights:
        model = YOLO(str(wpath))
        m = model.val(data=str(data), imgsz=a.imgsz, batch=4, device=0, plots=False, verbose=False, project=str(ROOT / "training" / "ds_eval"),
                      name="val_tmp", exist_ok=True, save=False)
        stats = {c: dict(tp=0, fp=0, fn=0) for c in CLASSES}
        multi = {c: 0 for c in CLASSES}
        for img, gt, _, _ in items:
            r = model(cv2.imread(str(img)), imgsz=a.imgsz, conf=a.conf, iou=a.iou, agnostic_nms=True, device=0, verbose=False)[0]
            preds = [(CLASSES[k], float(c), [float(v) for v in b]) for b, c, k in
                     zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int))]
            for cls in CLASSES:
                p = sorted([x for x in preds if x[0] == cls], key=lambda x: -x[1])
                left = list(gt[cls])
                multi[cls] += len(p) > 1
                for _, _, box in p:
                    best = max(left, key=lambda g: iou(box, g), default=None)
                    if best is not None and iou(box, best) >= 0.5:
                        stats[cls]["tp"] += 1
                        left.remove(best)
                    else:
                        stats[cls]["fp"] += 1
                stats[cls]["fn"] += len(left)
        print(f"=== {wpath.parent.parent.name} ===")
        print(f"mAP50 {m.box.map50:.3f}  mAP50-95 {m.box.map:.3f}   (robot {m.box.maps[0]:.3f}, ball {m.box.maps[1]:.3f} mAP50-95)")
        for cls in CLASSES:
            s = stats[cls]
            prec = s["tp"] / max(s["tp"] + s["fp"], 1)
            rec = s["tp"] / max(s["tp"] + s["fn"], 1)
            print(f"{cls:6s} TP {s['tp']:3d}  FP {s['fp']:3d}  FN {s['fn']:3d}   precision {prec:.3f}  recall {rec:.3f}   frames with >1 {cls}: {multi[cls]}")
        print()


if __name__ == "__main__":
    main()
