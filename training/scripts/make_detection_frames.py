"""Pick full frames for the robot + ball detector from our recordings and pre-label them with the current detector.

    uv run python training/scripts/make_detection_frames.py [--val-every 4] [--train-every 8] [--train-rooms room_2]

Frames are taken at fixed time steps, not chosen by the detector, so frames where it fails are in as well:
- val: every --val-every seconds of every camera's val stretch (dataset/splits.yaml), all rooms;
- train: every --train-every seconds of the train stretches of --train-rooms (room_1 already has many training frames,
  from the first, single-camera dataset).
Written lens-corrected (what tracking/ sees) to dataset/detection/<split>/recordings/<recording>_<camera>_t<frame>.jpg
with an X-AnyLabeling label: rectangles robot and ball from the current detector, to be corrected by hand. The pre-labels are
recorded in dataset/detection/prelabels.json; existing frames are not touched.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tracking"))
sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402
from dataset_splits import load as load_splits, ranges  # noqa: E402
from robot_crops import frame_name  # noqa: E402

OUT = ROOT / "dataset" / "detection"
CONF = 0.25


def rect(label, b):
    x1, y1, x2, y2 = (float(v) for v in b)
    return {"label": label, "score": None, "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], "group_id": None,
            "description": "", "difficult": False, "shape_type": "rectangle", "flags": {}, "attributes": {},
            "kie_linking": []}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-every", type=float, default=4.0)
    ap.add_argument("--train-every", type=float, default=8.0)
    ap.add_argument("--train-rooms", nargs="*", default=["room_2"])
    a = ap.parse_args()
    from ultralytics import YOLO
    detector = YOLO(common.OBJECT_MODEL_PATH)
    recs = yaml.safe_load((ROOT / "dataset" / "recordings.yaml").read_text())["recordings"]
    splits = load_splits()
    pre_path = OUT / "prelabels.json"
    prelabels = json.loads(pre_path.read_text(encoding="utf-8")) if pre_path.exists() else {}
    counts = {}
    for rec_name, rec in recs.items():
        for cam, info in rec["cameras"].items():
            cap = cv2.VideoCapture(str(ROOT / info["video"]))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fix = common.lens_correction(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            off = rec.get("offset_frames")
            shift = off if cam != "camera_1" and off is not None else 0      # paired camera_2: split timeline = frame + offset
            for split, rs in ranges(splits, rec_name, cam).items():
                every = a.val_every if split == "val" else a.train_every if rec["room"] in a.train_rooms else None
                if every is None:
                    continue
                for lo, hi in rs:
                    for t in range(lo, hi + 1, int(every * fps)):
                        f = t - shift
                        if not 0 <= f < n:
                            continue
                        name = frame_name(rec_name, cam, f)
                        out = OUT / split / "recordings"
                        if (out / f"{name}.json").exists():
                            continue
                        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                        ok, img = cap.read()
                        if not ok:
                            continue
                        img = fix(img)
                        r = detector.predict(img, imgsz=1920, conf=CONF, verbose=False)[0]
                        shapes = [rect(detector.names[int(k)], b) for b, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy())]
                        out.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(out / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                        d = {"version": "4.0.6", "flags": {}, "checked": False, "imagePath": f"{name}.jpg", "imageData": None,
                             "imageHeight": img.shape[0], "imageWidth": img.shape[1], "description": "", "shapes": shapes}
                        (out / f"{name}.json").write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
                        prelabels[name] = [[s["label"], [round(v, 2) for p in s["points"] for v in p]] for s in shapes]
                        counts[(rec_name, cam, split)] = counts.get((rec_name, cam, split), 0) + 1
    pre_path.parent.mkdir(parents=True, exist_ok=True)
    pre_path.write_text(json.dumps(prelabels, indent=1) + "\n", encoding="utf-8")
    for k, v in sorted(counts.items()):
        print(" ".join(k), v)
    print(f"wrote {sum(counts.values())} frames -> {OUT}")


if __name__ == "__main__":
    main()
