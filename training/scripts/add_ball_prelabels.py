"""Add ball boxes found by a detector to our detection frames, for review by hand. Robot boxes are not touched.

    uv run python training/scripts/add_ball_prelabels.py training/runs/detect_v2/weights/best.pt [--conf 0.25]

Goes through dataset/detection/{train,val}/recordings/*.json; every ball the detector finds that does not overlap a
ball box already there (IoU < 0.3) is added as a "ball" rectangle. Prints how many were added per folder.
"""
import argparse
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
DET = ROOT / "dataset" / "detection"


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def box_of(shape):
    xs, ys = [p[0] for p in shape["points"]], [p[1] for p in shape["points"]]
    return min(xs), min(ys), max(xs), max(ys)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1600)
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(str(ROOT / a.weights))
    for split in ("train", "val"):
        added, frames = 0, 0
        for js in sorted((DET / split / "recordings").glob("*.json")):
            d = json.loads(js.read_text(encoding="utf-8"))
            have = [box_of(s) for s in d["shapes"] if s["label"] == "ball"]
            r = model.predict(cv2.imread(str(js.with_suffix(".jpg"))), imgsz=a.imgsz, conf=a.conf, verbose=False)[0]
            new = []
            for b, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                if model.names[int(k)] != "ball" or any(iou(b, h) >= 0.3 for h in have + new):
                    continue
                new.append(tuple(float(v) for v in b))
            for x1, y1, x2, y2 in new:
                d["shapes"].append({"label": "ball", "score": None, "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                                    "group_id": None, "description": "", "difficult": False, "shape_type": "rectangle",
                                    "flags": {}, "attributes": {}, "kie_linking": []})
            if new:
                js.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                added += len(new)
                frames += 1
        print(f"{split}: {added} ball boxes added in {frames} frames")


if __name__ == "__main__":
    main()
