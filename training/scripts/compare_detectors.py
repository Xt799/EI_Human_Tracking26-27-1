"""Compare robot + ball detectors on the detector val set and on the false-detection crops.

    uv run python training/scripts/compare_detectors.py OLD.pt NEW.pt [--imgsz 1600]

Val: training/ds_detect (build_detection.py) - precision, recall, mAP per class. False detections: the crops of
dataset/robot_heading/ whose points were deleted (no robot in them, e.g. a printed shirt): on how many a robot is found.
"""
import argparse
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", nargs="+")
    ap.add_argument("--imgsz", type=int, default=1600)
    a = ap.parse_args()
    from ultralytics import YOLO
    negs = [j for j in (ROOT / "dataset" / "robot_heading").glob("*/*.json")
            if not json.loads(j.read_text(encoding="utf-8"))["shapes"]]
    for w in a.weights:
        m = YOLO(str(ROOT / w))
        r = m.val(data=str(ROOT / "training" / "ds_detect" / "data.yaml"), imgsz=a.imgsz, batch=8, workers=0,
                  verbose=False, plots=False)
        for i, c in r.names.items():
            p, rc, ap50, ap = r.box.class_result(i)
            print(f"{w}  {c}: precision {p:.2f}, recall {rc:.2f}, mAP50 {ap50:.2f}, mAP50-95 {ap:.2f}")
        fp = sum(any(m.names[int(k)] == "robot" for k in m.predict(cv2.imread(str(j.with_suffix(".jpg"))), imgsz=640,
                                                                   conf=0.3, verbose=False)[0].boxes.cls) for j in negs)
        print(f"{w}  robot found on {fp}/{len(negs)} false-detection crops")


if __name__ == "__main__":
    main()
