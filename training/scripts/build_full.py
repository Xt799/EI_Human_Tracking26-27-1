"""Build the YOLO dataset (full frames, no crops) from the X-AnyLabeling labels in dataset/{train,test,val}.

Run from the repository root:  uv run python training/scripts/build_full.py [--with-test]
  train  = dataset/train (+ dataset/test ONLY with --with-test; normally the test set must never be trained on)
  val    = dataset/val
Every image is used once per epoch.
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT.parent / "dataset"
OUT = ROOT / "ds_full"
CLASSES = ["robot", "ball"]


def convert(src: Path, tag: str):
    """Copy the labelled images of one dataset folder into ds_full and write YOLO labels. Returns the image paths."""
    paths = []
    for js in sorted(src.glob("*.json")):
        d = json.loads(js.read_text())
        if "shapes" not in d:      # not an X-AnyLabeling file (e.g. the old labels_real.json)
            continue
        w, h = d["imageWidth"], d["imageHeight"]
        lines = []
        for s in d["shapes"]:
            xs = [p[0] for p in s["points"]]
            ys = [p[1] for p in s["points"]]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            lines.append(f"{CLASSES.index(s['label'])} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                         f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
        stem = f"{tag}_{Path(d['imagePath']).stem}"
        img = OUT / "images" / (stem + Path(d["imagePath"]).suffix)
        shutil.copy(src / d["imagePath"], img)
        (OUT / "labels" / (stem + ".txt")).write_text("\n".join(lines))   # an empty file = background image
        paths.append(img.as_posix())
    return paths


ap = argparse.ArgumentParser()
ap.add_argument("--with-test", action="store_true", help="ALSO train on dataset/test (cheating: only for pseudo-labelling other videos)")
a = ap.parse_args()

shutil.rmtree(OUT, ignore_errors=True)
(OUT / "images").mkdir(parents=True)
(OUT / "labels").mkdir(parents=True)

train = convert(DATASET / "train", "train")
test = convert(DATASET / "test", "test") if a.with_test else []
val = convert(DATASET / "val", "val")
assert val, "dataset/val is empty"
(OUT / "train.txt").write_text("\n".join(train + test) + "\n")
(OUT / "val.txt").write_text("\n".join(val) + "\n")
(OUT / "data.yaml").write_text(f"path: {OUT.as_posix()}\ntrain: train.txt\nval: val.txt\n"
                               + "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
print(f"train: {len(train)} images" + (f" + test {len(test)}" if test else "") + f", val: {len(val)} images -> {OUT}")
