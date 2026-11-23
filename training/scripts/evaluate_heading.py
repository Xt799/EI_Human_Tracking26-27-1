"""Heading error of the front/rear model on a list of labelled crops (a split file of build_heading.py).

    uv run python training/scripts/evaluate_heading.py training/runs/<name>/weights/best.pt training/ds_heading/val.txt

Error: angle in the image between the predicted and the labelled rear -> front direction (the crops are not on a floor
map; in the image the angle is distorted by the perspective, but the same for both). Also the share within 10 / 20
degrees and flipped (more than 90 degrees off), per room; on negative crops (no robot), how often a robot is found.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from robot_crops import heading_deg  # noqa: E402

CONF = 0.25               # a robot found on a negative crop at this confidence counts as a false detection


def angle_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


def report(name, errs, n):
    e = np.array(errs)
    if not len(e):
        print(f"{name}: nothing found in {n} crops")
        return
    print(f"{name}: found {len(e)}/{n}, median {np.median(e):.1f} deg, mean {e.mean():.1f} deg, "
          f"within 10 deg {np.mean(e <= 10):.0%}, within 20 deg {np.mean(e <= 20):.0%}, flipped {np.mean(e > 90):.0%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights")
    ap.add_argument("list", nargs="+", help="split file(s) with image paths; labels are next to them in labels/")
    ap.add_argument("--imgsz", type=int, default=320)
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(str(ROOT / a.weights))
    errs, n = {}, {}             # per room (the crop name starts with the recording)
    neg, false_pos = 0, 0
    for lst in a.list:
        for line in Path(lst).read_text().split():
            img_path = Path(line)
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            v = (img_path.parent.parent / "labels" / f"{img_path.stem}.txt").read_text().split()
            if not v:                      # negative crop: no robot in it
                neg += 1
                r = model.predict(img, imgsz=a.imgsz, conf=CONF, verbose=False)[0]
                false_pos += int(len(r.boxes) > 0)
                continue
            room = img_path.stem[:6]
            front, rear = (float(v[5]) * w, float(v[6]) * h), (float(v[8]) * w, float(v[9]) * h)
            truth = heading_deg(front, rear)
            n[room] = n.get(room, 0) + 1
            r = model.predict(img, imgsz=a.imgsz, conf=0.05, verbose=False)[0]
            if len(r.boxes):
                k = r.keypoints.xy[int(r.boxes.conf.argmax())].cpu().numpy()
                errs.setdefault(room, []).append(angle_diff(heading_deg(k[0], k[1]), truth))
    for room in sorted(n):
        report(room, errs.get(room, []), n[room])
    if neg:
        print(f"negatives (no robot): the model found a robot (conf >= {CONF}) in {false_pos}/{neg}")
    return errs


if __name__ == "__main__":
    main()
