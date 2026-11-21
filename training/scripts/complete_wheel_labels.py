"""Complete the hand-placed wheel points of a paired recording, without touching them.

    uv run python training/scripts/complete_wheel_labels.py room_1_20261012_1800 [--review]

1. Other view: a wheel marked in one camera but not in the other is carried over through the floor (pixel -> floor map ->
   pixel). The wheel was not marked there because it is hidden, so it is added as hidden ("difficult": true).
   The hand-placed points of the two views agree to about 3 cm on the floor when mapped as floor points, so no height
   correction is applied.
2. Still robot: a moment without labels gets the points of a labelled moment if the robot has not moved: its box and
   appearance are the same in at least one camera (in the other one a person may stand in front of it). Each copied point
   is checked against the labelled image: if the image around it looks the same, it is visible, otherwise hidden.

Added points have "description": "auto: ..." (where they come from); hand-placed ones are left exactly as they are.
A rerun replaces the earlier "auto" points. --review writes overlay pages to training/ds_review/ (hand-placed: green,
added visible: yellow, added hidden: red).
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
SAME_SHIFT_PX = 2.0       # robot box centre moved at most this much ...
SAME_CORR = 0.9           # ... and its appearance correlates at least this much: the robot has not moved
OTHER_SHIFT_PX = 15.0     # in the other camera the box may jitter this much (a person in front of the robot)
PATCH = 7                 # half size of the patch compared around a copied point
VISIBLE_NCC = 0.8         # patch correlation above this: the wheel looks the same, so it is visible


def robot_boxes(model, folder, stems):
    out = {}
    for st in stems:
        img = cv2.imread(str(folder / f"{st}.jpg"))
        r = model.predict(img, imgsz=1920, conf=0.3, verbose=False)[0]
        boxes = [(c, b) for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy())
                 if model.names[int(k)] == "robot"]
        if not boxes:
            out[st] = None
            continue
        b = max(boxes, key=lambda x: x[0])[1]
        x1, y1, x2, y2 = b.astype(int)
        g = cv2.resize(cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32)
        out[st] = (b, (g - g.mean()) / (g.std() + 1e-6))
    return out


def compare(a, b):
    """(box centre shift in px, appearance correlation) of two (box, thumbnail); (inf, -1) if one is missing."""
    if a is None or b is None:
        return np.inf, -1.0
    shift = float(np.abs((a[0][:2] + a[0][2:]) / 2 - (b[0][:2] + b[0][2:]) / 2).max())
    return shift, float((a[1] * b[1]).mean())


def ncc(img_a, img_b, p):
    x, y = int(round(p[0])), int(round(p[1]))
    if min(x, y) < PATCH or x + PATCH >= img_a.shape[1] or y + PATCH >= img_a.shape[0]:
        return -1.0
    a = img_a[y - PATCH:y + PATCH + 1, x - PATCH:x + PATCH + 1].astype(np.float32).ravel()
    b = img_b[y - PATCH:y + PATCH + 1, x - PATCH:x + PATCH + 1].astype(np.float32).ravel()
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


def shape(label, p, description, hidden):
    return {"label": label, "score": None, "points": [[float(p[0]), float(p[1])]], "group_id": None,
            "description": description, "difficult": bool(hidden), "shape_type": "point", "flags": {},
            "attributes": {}, "kie_linking": []}


def load(path, img_shape):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"version": "4.0.6", "flags": {}, "checked": False, "shapes": [], "imagePath": path.with_suffix(".jpg").name,
            "imageData": None, "imageHeight": img_shape[0], "imageWidth": img_shape[1], "description": ""}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("--review", action="store_true")
    a = ap.parse_args()

    rec = yaml.safe_load((ROOT / "dataset" / "recordings.yaml").read_text())["recordings"][a.recording]
    folder = ROOT / "dataset" / a.recording
    floor = {c: common.load_floor_map(rec["cameras"][c]["floor_map"]) for c in CAMS}
    to_other = {"camera_1": np.linalg.inv(floor["camera_2"]) @ floor["camera_1"],    # camera_1 px -> camera_2 px
                "camera_2": np.linalg.inv(floor["camera_1"]) @ floor["camera_2"]}
    stems = sorted(p.stem for p in (folder / "camera_1").glob("*.jpg"))
    img_shape = cv2.imread(str(folder / "camera_1" / f"{stems[0]}.jpg")).shape

    # hand-placed points per moment and camera; earlier auto points are dropped
    labels = {}
    for st in stems:
        for c in CAMS:
            d = load(folder / c / f"{st}.json", img_shape)
            d["shapes"] = [s for s in d["shapes"] if not str(s.get("description", "")).startswith("auto")]
            labels[(st, c)] = d
    manual = {k: {s["label"]: s["points"][0] for s in d["shapes"] if s["label"] in WHEELS} for k, d in labels.items()}
    labelled = [st for st in stems if manual[(st, "camera_1")] or manual[(st, "camera_2")]]

    # 1. carry wheels over to the other view
    n_projected = 0
    full = {}    # (stem, cam) -> {wheel: (point, hidden)}
    for st in labelled:
        for c in CAMS:
            full[(st, c)] = {w: (np.array(p), False) for w, p in manual[(st, c)].items()}
        for c in CAMS:
            o = CAMS[1 - CAMS.index(c)]
            for w, p in manual[(st, c)].items():
                if w in manual[(st, o)]:
                    continue
                q = cv2.perspectiveTransform(np.array([[p]], np.float32), to_other[c])[0, 0]
                full[(st, o)][w] = (q, True)
                labels[(st, o)]["shapes"].append(shape(w, q, f"auto: projected from {c}", True))
                n_projected += 1

    # 2. copy to moments where the robot has not moved
    from ultralytics import YOLO
    model = YOLO(common.OBJECT_MODEL_PATH)
    boxes = {c: robot_boxes(model, folder / c, stems) for c in CAMS}
    grey = {}

    def gray(st, c):
        if (st, c) not in grey:
            grey[(st, c)] = cv2.cvtColor(cv2.imread(str(folder / c / f"{st}.jpg")), cv2.COLOR_BGR2GRAY)
        return grey[(st, c)]

    n_copied, n_hidden, copied_moments = 0, 0, 0
    for st in stems:
        if st in labelled:
            continue
        best = None
        for ref in labelled:
            cmp = {c: compare(boxes[c][st], boxes[c][ref]) for c in CAMS}
            same = [c for c in CAMS if cmp[c][0] <= SAME_SHIFT_PX and cmp[c][1] >= SAME_CORR]
            if not same:
                continue
            other_ok = all(cmp[c][0] <= OTHER_SHIFT_PX or boxes[c][st] is None for c in CAMS if c not in same)
            if not other_ok:
                continue
            score = max(cmp[c][1] for c in same)
            if best is None or score > best[0]:
                best = (score, ref)
        if best is None:
            continue
        ref = best[1]
        copied_moments += 1
        for c in CAMS:
            for w, (p, ref_hidden) in full[(ref, c)].items():
                hidden = ref_hidden or ncc(gray(st, c), gray(ref, c), p) < VISIBLE_NCC
                labels[(st, c)]["shapes"].append(shape(w, p, f"auto: copied from {ref} (robot not moved)", hidden))
                n_copied += 1
                n_hidden += hidden

    written = 0
    for (st, c), d in labels.items():
        path = folder / c / f"{st}.json"
        if not d["shapes"] and not path.exists():
            continue
        path.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        written += 1
    print(f"{len(labelled)} labelled moments; {n_projected} points carried over to the other view")
    print(f"{copied_moments} more moments where the robot had not moved: {n_copied} points copied ({n_hidden} hidden)")
    print(f"{len(stems) - len(labelled) - copied_moments} moments still without labels; {written} label files written")

    if a.review:
        out = ROOT / "training" / "ds_review" / a.recording
        out.mkdir(parents=True, exist_ok=True)
        tiles = []
        for st in stems:
            row = []
            for c in CAMS:
                d = labels[(st, c)]
                pts = [s for s in d["shapes"] if s["label"] in WHEELS]
                if not pts:
                    break
                img = cv2.imread(str(folder / c / f"{st}.jpg"))
                b = boxes[c][st][0] if boxes[c][st] is not None else None
                xy = np.array([s["points"][0] for s in pts])
                lo = np.minimum(xy.min(0), b[:2] if b is not None else xy.min(0)) - 30
                hi = np.maximum(xy.max(0), b[2:] if b is not None else xy.max(0)) + 30
                x0, y0 = np.maximum(lo, 0).astype(int)
                x1, y1 = hi.astype(int)
                crop = cv2.resize(img[y0:y1, x0:x1], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                for s in pts:
                    p = (np.array(s["points"][0]) - (x0, y0)) * 3
                    col = (0, 255, 0) if not str(s["description"]).startswith("auto") else ((0, 0, 255) if s["difficult"] else (0, 255, 255))
                    cv2.circle(crop, tuple(p.astype(int)), 6, col, 2)
                    cv2.putText(crop, s["label"][-1], tuple((p + (7, -7)).astype(int)), 0, 0.6, col, 2)
                cv2.putText(crop, f"{st} {c}", (5, 20), 0, 0.6, (255, 255, 255), 2)
                row.append(cv2.resize(crop, (300, 300)))
            if len(row) == 2:
                tiles.append(np.hstack(row))
        for k in range(0, len(tiles), 8):
            page = tiles[k:k + 8]
            page += [np.zeros_like(tiles[0])] * ((-len(page)) % 2)
            cv2.imwrite(str(out / f"page_{k // 8:02d}.jpg"), np.vstack([np.hstack(page[i:i + 2]) for i in range(0, len(page), 2)]))
        print(f"review pages -> {out}")


if __name__ == "__main__":
    main()
