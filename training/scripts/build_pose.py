"""Build the YOLO pose dataset (robot box + 4 wheel points) from the labelled paired recordings in dataset/.

    uv run python training/scripts/build_pose.py

Splits come from dataset/splits.yaml (stretches of time, both cameras of a moment in the same split). Only images with
at least one wheel point are used. The robot box comes from the robot model; when it misses the robot (a person in
front), the box is drawn around the wheel points. Point visibility: 2 = seen, 1 = hidden but known ("difficult"),
0 = not labelled. Output: training/ds_pose/ (images, labels, data.yaml).
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tracking"))
import common  # noqa: E402

OUT = ROOT / "training" / "ds_pose"
WHEELS = ["wheel_1", "wheel_2", "wheel_3", "wheel_4"]
FLIP = [1, 0, 3, 2]          # a mirrored image swaps left and right wheels
MARGIN = 0.25                # box around the wheel points when the robot model misses: grow by this share of its size


def main():
    from ultralytics import YOLO
    model = YOLO(common.OBJECT_MODEL_PATH)
    splits = yaml.safe_load((ROOT / "dataset" / "splits.yaml").read_text())
    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "images").mkdir(parents=True)
    (OUT / "labels").mkdir(parents=True)
    lists = {"train": [], "val": [], "test": []}
    counts = {k: [0, 0, 0] for k in lists}          # images, points seen, points hidden
    for rec, parts in splits.items():
        for js in sorted((ROOT / "dataset" / rec).glob("camera_*/*.json")):
            d = json.loads(js.read_text(encoding="utf-8"))
            pts = {s["label"]: (s["points"][0], 1 if s.get("difficult") else 2) for s in d["shapes"] if s["label"] in WHEELS}
            if not pts:
                continue
            frame = int(js.stem[1:])
            split = next((k for k, (lo, hi) in parts.items() if lo <= frame <= hi), None)
            if split is None:
                continue
            img_path = js.with_suffix(".jpg")
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            r = model.predict(img, imgsz=1920, conf=0.3, verbose=False)[0]
            boxes = [(c, b) for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy())
                     if model.names[int(k)] == "robot"]
            xy = np.array([p for p, _ in pts.values()], float)
            if boxes:
                x1, y1, x2, y2 = max(boxes, key=lambda x: x[0])[1]
            else:
                (x1, y1), (x2, y2) = xy.min(0), xy.max(0)
                g = MARGIN * max(x2 - x1, y2 - y1, 40)
                x1, y1, x2, y2 = x1 - g, y1 - g, x2 + g, y2 + g
            x1, y1 = min(x1, xy[:, 0].min() - 3), min(y1, xy[:, 1].min() - 3)    # the box must hold every point
            x2, y2 = max(x2, xy[:, 0].max() + 3), max(y2, xy[:, 1].max() + 3)
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w - 1), min(y2, h - 1)
            kp = []
            for wl in WHEELS:
                if wl in pts:
                    (px, py), v = pts[wl]
                    kp += [px / w, py / h, v]
                    counts[split][1 if v == 2 else 2] += 1
                else:
                    kp += [0, 0, 0]
            line = f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f} " + " ".join(f"{v:.6f}" if i % 3 < 2 else str(int(v)) for i, v in enumerate(kp))
            stem = f"{rec}_{js.parent.name}_{js.stem}"
            shutil.copy(img_path, OUT / "images" / f"{stem}.jpg")
            (OUT / "labels" / f"{stem}.txt").write_text(line + "\n")
            lists[split].append((OUT / "images" / f"{stem}.jpg").as_posix())
            counts[split][0] += 1
    for k, v in lists.items():
        (OUT / f"{k}.txt").write_text("\n".join(v) + "\n")
    (OUT / "data.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: train.txt\nval: val.txt\ntest: test.txt\n"
        f"kpt_shape: [4, 3]\nflip_idx: {FLIP}\nkpt_names:\n  0: [{', '.join(WHEELS)}]\nnames:\n  0: robot\n")
    for k, (n, seen, hidden) in counts.items():
        print(f"{k}: {n} images, {seen} wheel points seen, {hidden} hidden")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
