"""Score the trained models on their val sets.

    uv run python training/scripts/evaluate.py detector training/runs/<name>/weights/best.pt [more weights] [--imgsz 1920]
    uv run python training/scripts/evaluate.py heading training/runs/<name>/weights/best.pt [--list training/ds_heading/val.txt]

detector: precision, recall and mAP per class on training/ds_detect (build_detection.py), and on how many of the
  false-detection crops (dataset/robot_heading/ crops whose points were deleted: no robot in them, e.g. a printed
  shirt) a robot is found.
heading: on a split file of build_heading.py, the angle in the image between the predicted and the labelled
  rear -> front direction (the crops are not on a floor map; the perspective distorts the angle, but the same for
  both), with the share within 10 / 20 degrees and flipped (more than 90 degrees off), per room; on the negative crops
  (no robot), how often a robot is found.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from dataset import HEADING_DIR, ROOT, heading_deg

NEGATIVE_CONF = 0.25      # a robot found on a negative heading crop at this confidence counts as a false detection


def detector(weights, imgsz):
    from ultralytics import YOLO
    negs = [j for j in HEADING_DIR.glob("*/*.json") if not json.loads(j.read_text(encoding="utf-8"))["shapes"]]
    for w in weights:
        m = YOLO(str(ROOT / w))
        r = m.val(data=str(ROOT / "training" / "ds_detect" / "data.yaml"), imgsz=imgsz, batch=8, workers=0,
                  verbose=False, plots=False, project=str(ROOT / "training" / "runs"), name="val", exist_ok=True)
        for i, c in r.names.items():
            p, rc, ap50, ap = r.box.class_result(i)
            print(f"{w}  {c}: precision {p:.2f}, recall {rc:.2f}, mAP50 {ap50:.2f}, mAP50-95 {ap:.2f}")
        fp = sum(any(m.names[int(k)] == "robot" for k in m.predict(cv2.imread(str(j.with_suffix(".jpg"))), imgsz=640,
                                                                   conf=0.3, verbose=False)[0].boxes.cls) for j in negs)
        print(f"{w}  robot found on {fp}/{len(negs)} false-detection crops")


def angle_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


def report(name, errs, n):
    e = np.array(errs)
    if not len(e):
        print(f"{name}: nothing found in {n} crops")
        return
    print(f"{name}: found {len(e)}/{n}, median {np.median(e):.1f} deg, mean {e.mean():.1f} deg, "
          f"within 10 deg {np.mean(e <= 10):.0%}, within 20 deg {np.mean(e <= 20):.0%}, flipped {np.mean(e > 90):.0%}")


def heading(weights, lists, imgsz=320):
    from ultralytics import YOLO
    model = YOLO(str(ROOT / weights))
    errs, n = {}, {}                       # per room (the crop name starts with the recording)
    neg, false_pos = 0, 0
    for lst in lists:
        for line in (ROOT / lst).read_text().split():
            img_path = Path(line)
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            v = (img_path.parent.parent / "labels" / f"{img_path.stem}.txt").read_text().split()
            if not v:                      # negative crop: no robot in it
                neg += 1
                r = model.predict(img, imgsz=imgsz, conf=NEGATIVE_CONF, verbose=False)[0]
                false_pos += int(len(r.boxes) > 0)
                continue
            room = img_path.stem[:6]
            front, rear = (float(v[5]) * w, float(v[6]) * h), (float(v[8]) * w, float(v[9]) * h)
            n[room] = n.get(room, 0) + 1
            r = model.predict(img, imgsz=imgsz, conf=0.05, verbose=False)[0]
            if len(r.boxes):
                k = r.keypoints.xy[int(r.boxes.conf.argmax())].cpu().numpy()
                errs.setdefault(room, []).append(angle_diff(heading_deg(k[0], k[1]), heading_deg(front, rear)))
    for room in sorted(n):
        report(room, errs.get(room, []), n[room])
    if neg:
        print(f"negatives (no robot): the model found a robot (conf >= {NEGATIVE_CONF}) in {false_pos}/{neg}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=["detector", "heading"])
    ap.add_argument("weights", nargs="+")
    ap.add_argument("--imgsz", type=int, default=1920, help="detector input size")
    ap.add_argument("--list", nargs="+", default=["training/ds_heading/val.txt"], help="heading: split file(s)")
    a = ap.parse_args()
    if a.model == "detector":
        detector(a.weights, a.imgsz)
    else:
        for w in a.weights:
            heading(w, a.list)


if __name__ == "__main__":
    main()
