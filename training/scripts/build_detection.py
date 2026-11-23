"""Build the YOLO dataset of the robot + ball detector: training/ds_detect/.

    uv run python training/scripts/build_detection.py [--own-repeat 5]

train:
- dataset/detection/train/recordings/  our frames (reviewed boxes)
- dataset/detection/train/screenshots/ a few pasted screenshots of the first room_1 videos (source frame unknown)
- dataset/detection/train/coco/, openimages/   public ball images (no robot in them: they also teach the detector
                                       that people, clothes and furniture are not the robot)
- dataset/robot_heading/train/ crops whose points were deleted: false robot detections (e.g. a printed shirt), as
  images with no object
Of the public images only --public per source are used (half of them those with the smallest balls); our own
images are listed --own-repeat times, so they are not drowned by them.
val: dataset/detection/val/recordings/ and the false-detection crops of dataset/robot_heading/val/.
Classes: 0 robot, 1 ball. Images are hard-linked (no copy) into training/ds_detect/images/.
"""
import argparse
import json
import os
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DET = ROOT / "dataset" / "detection"
OUT = ROOT / "training" / "ds_detect"
CLASSES = ["robot", "ball"]


def yolo_lines(d):
    w, h = d["imageWidth"], d["imageHeight"]
    lines = []
    for s in d["shapes"]:
        if s["label"] not in CLASSES or s["shape_type"] != "rectangle":
            continue
        xs, ys = [p[0] for p in s["points"]], [p[1] for p in s["points"]]
        x1, x2, y1, y2 = max(min(xs), 0), min(max(xs), w), max(min(ys), 0), min(max(ys), h)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        lines.append(f"{CLASSES.index(s['label'])} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines


def negatives(split):
    """Heading crops whose points were all deleted: no robot in them."""
    out = []
    for js in (ROOT / "dataset" / "robot_heading" / split).glob("*.json"):
        if not json.loads(js.read_text(encoding="utf-8"))["shapes"]:
            out.append(js)
    return out


def pick_public(folder, n):
    """n label files of a public source: half the ones with the smallest ball (ours are small, seen from above),
    half random from the rest. All of them if n is 0."""
    files = sorted(folder.glob("*.json"))
    if not n or n >= len(files):
        return files

    def smallest(js):
        d = json.loads(js.read_text(encoding="utf-8"))
        areas = [(max(p[0] for p in s["points"]) - min(p[0] for p in s["points"])) *
                 (max(p[1] for p in s["points"]) - min(p[1] for p in s["points"])) for s in d["shapes"]]
        return min(areas) / (d["imageWidth"] * d["imageHeight"]) if areas else 1.0
    by_size = sorted(files, key=smallest)
    small, rest = by_size[:n // 2], by_size[n // 2:]
    return small + random.Random(0).sample(rest, n - len(small))


def add(js, prefix, empty=False):
    """Link the image of a label file into the dataset and write its YOLO labels; returns the image path."""
    d = json.loads(js.read_text(encoding="utf-8"))
    src = js.parent / Path(d.get("imagePath") or js.with_suffix(".jpg").name).name
    if not src.exists():
        src = next((p for p in (js.with_suffix(".jpg"), js.with_suffix(".png")) if p.exists()), None)
    if src is None:
        return None
    name = f"{prefix}{js.stem}"
    img = OUT / "images" / f"{name}{src.suffix.lower()}"
    if not img.exists():
        try:
            os.link(src, img)
        except OSError:
            shutil.copy(src, img)
    (OUT / "labels" / f"{name}.txt").write_text("" if empty else "\n".join(yolo_lines(d)) + "\n")
    return img.as_posix()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--own-repeat", type=int, default=5)
    ap.add_argument("--public", type=int, default=1000, help="public images per source (coco, openimages); 0 = all")
    a = ap.parse_args()
    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "images").mkdir(parents=True)
    (OUT / "labels").mkdir(parents=True)
    own = [add(j, "") for j in sorted((DET / "train" / "recordings").glob("*.json"))]
    shots = [add(j, "") for j in sorted((DET / "train" / "screenshots").glob("*.json"))]
    neg = [add(j, "neg_", empty=True) for j in negatives("train")]
    public = [add(j, "") for src in ("coco", "openimages") for j in pick_public(DET / "train" / src, a.public)]
    own, shots, neg, public = ([p for p in x if p] for x in (own, shots, neg, public))
    train = (own + shots + neg) * a.own_repeat + public
    val = [add(j, "") for j in sorted((DET / "val" / "recordings").glob("*.json"))] + \
          [add(j, "neg_", empty=True) for j in negatives("val")]
    val = [p for p in val if p]
    (OUT / "train.txt").write_text("\n".join(train) + "\n")
    (OUT / "val.txt").write_text("\n".join(val) + "\n")
    (OUT / "data.yaml").write_text(f"path: {OUT.as_posix()}\ntrain: train.txt\nval: val.txt\nnames:\n"
                                   + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    print(f"train: {len(own)} own frames, {len(shots)} screenshots, {len(neg)} false-detection crops "
          f"(each x{a.own_repeat}), {len(public)} public ball images -> {len(train)} entries")
    print(f"val: {len(val)} images -> {OUT}")


if __name__ == "__main__":
    main()
