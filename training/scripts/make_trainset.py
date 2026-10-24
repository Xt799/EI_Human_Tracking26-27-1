"""Pseudo-label many diverse frames from the NON-test videos into dataset/train for MANUAL REVIEW in X-AnyLabeling.

uv run python training/scripts/make_trainset.py [--n 400] [--hard 120] [--scan-step 5]

1. Scan every --scan-step-th frame of every video except the largest (= test video) with three model configurations, and cluster their boxes.
2. Choose --n frames: up to --hard "hard" ones (second robot/ball candidate, uncertain, seen by one configuration only), the rest by
   farthest-point sampling on thumbnails, so near-identical frames (robot standing still) are not chosen again and again.
3. Write dataset/train/auto_<video>_<frame>.jpg + .json (X-AnyLabeling format: at most 1 robot + 1 ball per frame), review pages in
   dataset/train/review/page_*.jpg (flagged frames first) and dataset/train/frames_to_check.txt.
The labels are model output, NOT ground truth, until you have reviewed them.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "dataset" / "train"
CLASSES = ["robot", "ball"]
COLORS = {"robot": (0, 200, 255), "ball": (0, 255, 0)}   # BGR
CONFIGS = [("yolo26n_e30", 1920), ("yolo26n_e30", 2560), ("yolo26n_e30", 3200)]   # (run name, imgsz): their agreement is the confidence
CONF = 0.25


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def cluster(dets):
    """dets: list of dict(cfg, conf, box) of one class -> clusters sorted by support (sum of conf)."""
    left, out = sorted(dets, key=lambda d: -d["conf"]), []
    while left:
        seed = left[0]
        members = [d for d in left if iou(seed["box"], d["box"]) >= 0.5]
        left = [d for d in left if d not in members]
        w = np.array([d["conf"] for d in members])
        box = (np.array([d["box"] for d in members]) * w[:, None]).sum(0) / w.sum()
        out.append(dict(box=box, support=float(w.sum()), n_cfg=len({d["cfg"] for d in members}), mean=float(w.mean())))
    return sorted(out, key=lambda c: -c["support"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--hard", type=int, default=120)
    ap.add_argument("--scan-step", type=int, default=5)
    ap.add_argument("--force", action="store_true", help="overwrite the auto_* frames (and your corrections!) already in dataset/train")
    a = ap.parse_args()
    if list(OUT.glob("auto_*.json")) and not a.force:
        raise SystemExit(f"{OUT} already holds auto_* labels (probably hand-corrected). Use --force to overwrite them.")

    models = {n: YOLO(str(ROOT / "training" / "runs" / n / "weights" / "best.pt")) for n in {c[0] for c in CONFIGS}}
    videos = sorted((ROOT / "videos").glob("*.mp4"), key=lambda p: p.stat().st_size)[:-1]   # all but the test video

    def detect(f):
        res = []
        for i, (name, sz) in enumerate(CONFIGS):
            r = models[name](f, imgsz=sz, conf=CONF, iou=0.45, agnostic_nms=True, device=0, verbose=False)[0]
            for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
                res.append(dict(cfg=i, cls=CLASSES[k], conf=float(c), box=[float(v) for v in b]))
        return res

    def label(dets):
        """-> (shapes as (cls, box, note), flags, extra candidate boxes)"""
        shapes, flags, extra = [], [], []
        for cls in CLASSES:
            cl = cluster([d for d in dets if d["cls"] == cls])
            if not cl:
                continue
            best = cl[0]
            if best["mean"] < 0.5 or best["n_cfg"] < 2:
                flags.append(f"{cls} uncertain ({best['mean']:.2f}, {best['n_cfg']} cfg)")
            if len(cl) > 1 and cl[1]["support"] >= 0.5:
                flags.append(f"second {cls} candidate")
                extra.append(cl[1]["box"])
            shapes.append((cls, best["box"], f"auto: {cls} support={best['support']:.2f} cfgs={best['n_cfg']}/{len(CONFIGS)} meanconf={best['mean']:.2f}"))
        return shapes, flags, extra

    # ---- 1. scan
    scan = []   # (video index, frame idx, thumb, flags, centers)
    for vi, v in enumerate(videos):
        cap = cv2.VideoCapture(str(v))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"scanning {v.name}: {n} frames", flush=True)
        for idx in range(0, n, a.scan_step):
            ok, f = cap.read()
            for _ in range(a.scan_step - 1):
                cap.grab()
            if not ok:
                break
            shapes, flags, _ = label(detect(f))
            thumb = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (48, 30), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255
            centers = np.full(4, -1.0, np.float32)
            for cls, box, _ in shapes:
                k = CLASSES.index(cls) * 2
                centers[k:k + 2] = [(box[0] + box[2]) / 2 / 1900, (box[1] + box[3]) / 2 / 1200]
            scan.append((vi, idx, thumb, flags, centers))
    print(f"{len(scan)} frames scanned, {sum(bool(s[3]) for s in scan)} with a flag", flush=True)

    # ---- 2. choose
    hard = [i for i, s in enumerate(scan) if s[3]]
    hard = [hard[int((k + 0.5) * len(hard) / a.hard)] for k in range(min(a.hard, len(hard)))] if hard else []
    feats = np.stack([np.concatenate([s[2], s[4] * 6]) for s in scan])
    chosen = list(dict.fromkeys(hard))
    dist = np.full(len(scan), np.inf, np.float32)
    for i in chosen:
        dist = np.minimum(dist, np.linalg.norm(feats - feats[i], axis=1))
    if not chosen:
        chosen.append(0)
        dist = np.linalg.norm(feats - feats[0], axis=1)
    while len(chosen) < min(a.n, len(scan)):   # farthest-point sampling
        i = int(dist.argmax())
        chosen.append(i)
        dist = np.minimum(dist, np.linalg.norm(feats - feats[i], axis=1))
    chosen.sort(key=lambda i: (scan[i][0], scan[i][1]))

    # ---- 3. write
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "review").mkdir(exist_ok=True)
    for old in (OUT / "review").glob("page_*"):
        old.unlink()
    items, caps = [], {}
    for i in chosen:
        vi, idx = scan[i][0], scan[i][1]
        cap = caps.setdefault(vi, cv2.VideoCapture(str(videos[vi])))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok:
            continue
        H, W = f.shape[:2]
        name = f"auto_{videos[vi].stem[-6:]}_{idx:06d}.jpg"
        cv2.imwrite(str(OUT / name), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
        shapes, flags, extra = label(detect(f))
        doc = {"version": "2.5.4", "flags": {}, "imagePath": name, "imageData": None, "imageHeight": H, "imageWidth": W, "shapes": [
            {"label": c, "score": None, "points": [[round(b[0], 1), round(b[1], 1)], [round(b[2], 1), round(b[3], 1)]], "group_id": None,
             "description": note, "difficult": False, "shape_type": "rectangle", "flags": {}, "attributes": {}, "kie_linking": []}
            for c, b, note in shapes]}
        (OUT / (Path(name).stem + ".json")).write_text(json.dumps(doc, indent=2))
        vis = f.copy()
        for c, b, _ in shapes:
            cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), COLORS[c], 3)
            cv2.putText(vis, c, (int(b[0]), max(22, int(b[1]) - 8)), 0, 0.9, COLORS[c], 2)
        for b in extra:
            cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
        tag = name + (("  CHECK: " + "; ".join(flags)) if flags else "")
        cv2.putText(vis, tag, (10, 36), 0, 1.0, (0, 0, 0), 5)
        cv2.putText(vis, tag, (10, 36), 0, 1.0, (0, 0, 255) if flags else (255, 255, 255), 2)
        items.append((bool(flags), name, cv2.resize(vis, (W // 2, H // 2)), flags))
    if not (OUT / "classes.txt").exists():
        (OUT / "classes.txt").write_text("\n".join(CLASSES) + "\n")
    items.sort(key=lambda t: (not t[0], t[1]))
    for p in range(0, len(items), 4):
        tiles = [t[2] for t in items[p:p + 4]]
        tiles += [np.zeros_like(tiles[0])] * (4 - len(tiles))
        cv2.imwrite(str(OUT / "review" / f"page_{p // 4:03d}.jpg"), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    (OUT / "frames_to_check.txt").write_text("".join(f"{t[1]}: {'; '.join(t[3])}\n" for t in items if t[0]))
    print(f"{len(items)} frames written to {OUT}, {sum(t[0] for t in items)} flagged (frames_to_check.txt, review/page_*.jpg)")


if __name__ == "__main__":
    main()
