"""Pick full frames for the robot + ball detector from our recordings and pre-label them with the current detector.

    uv run python training/scripts/make_detection_frames.py [--val-every 4] [--train-every 8] [--train-rooms room_2]
    uv run python training/scripts/make_detection_frames.py --add-balls WEIGHTS [--conf 0.2] [--imgsz 1920]

New frames: taken at fixed time steps, not chosen by the detector, so frames where it fails are in as well:
- val: every --val-every seconds of every camera's val stretch (dataset/splits.yaml), all rooms;
- train: every --train-every seconds of the train stretches of --train-rooms (room_1 already has many training frames,
  from the first, single-camera dataset).
Written lens-corrected (what tracking/ sees) to dataset/detection/<split>/recordings/<recording>_<camera>_t<frame>.jpg
with an X-AnyLabeling label: rectangles robot and ball from the current detector, to be corrected by hand. The
pre-labels are recorded in dataset/detection/prelabels.json; existing frames are not touched.

--add-balls: no new frames; every ball the given detector finds in the frames already there that does not overlap a
ball box (IoU < 0.3) is added as a "ball" rectangle, for review. Robot boxes are not touched.
"""
import argparse
import json

import cv2

from dataset import DETECTION_DIR, ROOT, find_video, frame_name, load_splits, ranges, recordings, rect
import common

CONF = 0.25


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def box_of(shape):
    xs, ys = [p[0] for p in shape["points"]], [p[1] for p in shape["points"]]
    return min(xs), min(ys), max(xs), max(ys)


def add_balls(weights, conf, imgsz):
    from ultralytics import YOLO
    model = YOLO(str(ROOT / weights))
    for split in ("train", "val"):
        added, frames = 0, 0
        for js in sorted((DETECTION_DIR / split / "recordings").glob("*.json")):
            d = json.loads(js.read_text(encoding="utf-8"))
            have = [box_of(s) for s in d["shapes"] if s["label"] == "ball"]
            r = model.predict(cv2.imread(str(js.with_suffix(".jpg"))), imgsz=imgsz, conf=conf, verbose=False)[0]
            new = []
            for b, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                if model.names[int(k)] == "ball" and all(iou(b, h) < 0.3 for h in have + new):
                    new.append(tuple(float(v) for v in b))
            if new:
                d["shapes"] += [rect("ball", b) for b in new]
                js.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                added += len(new)
                frames += 1
        print(f"{split}: {added} ball boxes added in {frames} frames")


def new_frames(a):
    from ultralytics import YOLO
    detector = YOLO(common.OBJECT_MODEL_PATH)
    splits = load_splits()
    pre_path = DETECTION_DIR / "prelabels.json"
    prelabels = json.loads(pre_path.read_text(encoding="utf-8")) if pre_path.exists() else {}
    counts = {}
    for rec_name, rec in recordings().items():
        for cam, info in rec["cameras"].items():
            cap = cv2.VideoCapture(find_video(info["video"]))
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
                        out = DETECTION_DIR / split / "recordings"
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
    print(f"wrote {sum(counts.values())} frames -> {DETECTION_DIR}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-every", type=float, default=4.0)
    ap.add_argument("--train-every", type=float, default=8.0)
    ap.add_argument("--train-rooms", nargs="*", default=["room_2"])
    ap.add_argument("--add-balls", metavar="WEIGHTS", help="add the balls this detector finds to the frames already there")
    ap.add_argument("--conf", type=float, default=0.2, help="--add-balls: minimum ball confidence")
    ap.add_argument("--imgsz", type=int, default=1920, help="--add-balls: detector input size")
    a = ap.parse_args()
    if a.add_balls:
        add_balls(a.add_balls, a.conf, a.imgsz)
    else:
        new_frames(a)


if __name__ == "__main__":
    main()
