"""Download public ball images with boxes (COCO 'sports ball', Open Images ball classes) into the detector's training data.

    uv run python training/scripts/download_balls.py [--per-class 1500] [--workers 16]

COCO 2017: every image with a 'sports ball' box (train and val; crowd boxes skipped). Open Images V7: up to --per-class
images of each ball class (Ball, Football, Tennis ball, Volleyball, Rugby ball, Cricket ball, Golf ball), from the
train and validation box annotations (streamed line by line, not stored). Images are shrunk to at most MAX_SIDE px.
Output: dataset/detection/train/<source>/<source>_<id>.jpg + .json (X-AnyLabeling, rectangles "ball"). The source
folders are git-ignored (about 1.3 GB, more than the free Git LFS quota); this script brings them back. A rerun skips
images already there.
"""
import argparse
import csv
import io
import json
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "dataset" / "detection" / "train"
CACHE = ROOT / "training" / "cache"            # downloaded annotation files (git-ignored)
MAX_SIDE = 1280
HEADERS = {"User-Agent": "dataset-downloader"}
COCO_ANN = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMG = "http://images.cocodataset.org/{split}/{file}"
OI_BOX = {"train": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
          "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"}
OI_IMG = "https://s3.amazonaws.com/open-images-dataset/{split}/{id}.jpg"
OI_CLASSES = {"/m/018xm": "Ball", "/m/01226z": "Football", "/m/05ctyq": "Tennis ball", "/m/02rgn06": "Volleyball",
              "/m/0wdt60w": "Rugby ball", "/m/02ctlc": "Cricket ball", "/m/044r5d": "Golf ball"}


def fetch(url, path, boxes):
    """Download one image, shrink it, write it with its labels. boxes: [(cx, cy, w, h)] normalised."""
    name = f"{path.name}_{url_id(url)}"
    img_path, lbl_path = path / f"{name}.jpg", path / f"{name}.json"
    if img_path.exists() and lbl_path.exists():
        return True
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return False
        s = MAX_SIDE / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        h, w = img.shape[:2]
        shapes = []
        for cx, cy, bw, bh in boxes:
            x1, y1, x2, y2 = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
            shapes.append({"label": "ball", "score": None, "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                           "group_id": None, "description": "", "difficult": False, "shape_type": "rectangle",
                           "flags": {}, "attributes": {}, "kie_linking": []})
        d = {"version": "4.0.6", "flags": {}, "checked": False, "imagePath": img_path.name, "imageData": None,
             "imageHeight": h, "imageWidth": w, "description": "", "shapes": shapes}
        lbl_path.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
        return True
    except (requests.RequestException, OSError):
        return False


def url_id(url):
    return url.rsplit("/", 1)[1].rsplit(".", 1)[0]


def run(jobs, path, workers):
    path.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as ex:
        ok = list(ex.map(lambda j: fetch(j[0], path, j[1]), jobs))
    print(f"{path.name}: {sum(ok)}/{len(jobs)} images")


def coco(workers):
    cache = CACHE / "coco_annotations_trainval2017.zip"
    if not cache.exists():
        print("downloading COCO annotations (241 MB)...", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(COCO_ANN, headers=HEADERS, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(cache, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    jobs = []
    with zipfile.ZipFile(cache) as z:
        for split in ("train2017", "val2017"):
            d = json.loads(z.read(f"annotations/instances_{split}.json"))
            ball = next(c["id"] for c in d["categories"] if c["name"] == "sports ball")
            imgs = {i["id"]: i for i in d["images"]}
            boxes = {}
            for a in d["annotations"]:
                if a["category_id"] == ball and not a["iscrowd"]:
                    im = imgs[a["image_id"]]
                    x, y, w, h = a["bbox"]
                    boxes.setdefault(a["image_id"], []).append(((x + w / 2) / im["width"], (y + h / 2) / im["height"],
                                                                w / im["width"], h / im["height"]))
            jobs += [(COCO_IMG.format(split=split, file=imgs[i]["file_name"]), b) for i, b in boxes.items()]
    print(f"COCO: {len(jobs)} images with a sports ball", flush=True)
    run(jobs, OUT / "coco", workers)


def openimages(per_class, workers):
    per_image = {}                       # (split, image id) -> [boxes], [classes]
    for split, url in OI_BOX.items():
        print(f"reading Open Images {split} boxes (streamed)...", flush=True)
        with requests.get(url, headers=HEADERS, stream=True, timeout=60) as r:
            r.raise_for_status()
            lines = (line.decode() for line in r.iter_lines())
            header = next(csv.reader([next(lines)]))
            col = {k: i for i, k in enumerate(header)}
            for line in lines:
                if not any(c in line for c in OI_CLASSES):
                    continue
                row = next(csv.reader(io.StringIO(line)))
                if row[col["LabelName"]] not in OI_CLASSES:
                    continue
                x0, x1, y0, y1 = (float(row[col[k]]) for k in ("XMin", "XMax", "YMin", "YMax"))
                e = per_image.setdefault((split, row[col["ImageID"]]), ([], set()))
                e[0].append(((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
                e[1].add(OI_CLASSES[row[col["LabelName"]]])
    # up to per_class images per class (an image with several ball classes counts for each)
    rng = random.Random(0)
    keys = list(per_image)
    rng.shuffle(keys)
    count, chosen = {c: 0 for c in OI_CLASSES.values()}, []
    for k in keys:
        cls = per_image[k][1]
        if any(count[c] < per_class for c in cls):
            chosen.append(k)
            for c in cls:
                count[c] += 1
    print("Open Images: " + ", ".join(f"{c} {n}" for c, n in count.items()), flush=True)
    jobs = [(OI_IMG.format(split=s, id=i), per_image[(s, i)][0]) for s, i in chosen]
    run(jobs, OUT / "openimages", workers)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", choices=["coco", "openimages"])
    a = ap.parse_args()
    if a.only != "openimages":
        coco(a.workers)
    if a.only != "coco":
        openimages(a.per_class, a.workers)


if __name__ == "__main__":
    main()
