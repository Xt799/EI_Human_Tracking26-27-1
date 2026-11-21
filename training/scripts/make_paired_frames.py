"""Pick diverse moments of a paired recording and write both camera views as images into dataset/<recording>/.

    uv run python training/scripts/make_paired_frames.py room_1_20261012_1800 [--n 150] [--step 15]

1. Candidate moments: every --step-th frame of the common timeline (camera_1 frames) inside the recording's paired range.
2. The robot model runs on both views; a moment is kept only if the robot is found in both.
3. Diversity: the robot crops of both views are shrunk to small grey thumbnails and moments are picked by farthest-point
   sampling, per split (splits.yaml) in proportion to its length, so the robot's pose, occlusions and surroundings vary.
4. Written as lens-corrected images (what tracking/ feeds the models): dataset/<recording>/camera_<k>/t<frame>.jpg,
   the same file name in both camera folders = the same moment.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tracking"))
import common  # noqa: E402

THUMB = 24


def robot_box(model, img):
    r = model.predict(img, imgsz=1920, conf=0.3, verbose=False)[0]
    boxes = [(c, b) for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy())
             if model.names[int(k)] == "robot"]
    return max(boxes, key=lambda x: x[0])[1] if boxes else None


def thumb(img, box):
    x1, y1, x2, y2 = box
    m = 0.1 * max(x2 - x1, y2 - y1)
    c = img[int(max(0, y1 - m)):int(y2 + m), int(max(0, x1 - m)):int(x2 + m)]
    t = cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), (THUMB, THUMB)).astype(np.float32).ravel()
    return (t - t.mean()) / (t.std() + 1e-6)


def farthest(feats, k, rng):
    if len(feats) <= k:
        return list(range(len(feats)))
    chosen = [int(rng.integers(len(feats)))]
    d = np.linalg.norm(feats - feats[chosen[0]], axis=1)
    while len(chosen) < k:
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(feats - feats[i], axis=1))
    return sorted(chosen)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("--n", type=int, default=150, help="number of moments (each gives one image per camera)")
    ap.add_argument("--step", type=int, default=15, help="candidate every n-th frame of the common timeline")
    a = ap.parse_args()

    rec = yaml.safe_load((ROOT / "dataset" / "recordings.yaml").read_text())["recordings"][a.recording]
    splits = yaml.safe_load((ROOT / "dataset" / "splits.yaml").read_text())[a.recording]
    off = rec["offset_frames"]
    first, last = rec["paired_frames"]
    from ultralytics import YOLO
    model = YOLO(common.OBJECT_MODEL_PATH)
    caps = {k: cv2.VideoCapture(str(ROOT / v["video"])) for k, v in rec["cameras"].items()}
    size = (int(caps["camera_1"].get(cv2.CAP_PROP_FRAME_WIDTH)), int(caps["camera_1"].get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fix = common.lens_correction(*size)

    def read(cam, frame):
        cap = caps[cam]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        return fix(img) if ok else None

    cands = []   # (frame, feature)
    for f in range(first, last + 1, a.step):
        f2 = f - off
        imgs = {"camera_1": read("camera_1", f), "camera_2": read("camera_2", f2)}
        if any(v is None for v in imgs.values()):
            continue
        boxes = {k: robot_box(model, v) for k, v in imgs.items()}
        if any(b is None for b in boxes.values()):
            continue
        cands.append((f, np.concatenate([thumb(imgs[k], boxes[k]) for k in ("camera_1", "camera_2")])))
        if len(cands) % 50 == 0:
            print(f"  {len(cands)} candidate moments (frame {f}/{last})", flush=True)
    print(f"{len(cands)} moments with the robot in both views")

    rng = np.random.default_rng(0)
    total = sum(hi - lo + 1 for lo, hi in splits.values())
    picked = []
    for name, (lo, hi) in splits.items():
        idx = [i for i, (f, _) in enumerate(cands) if lo <= f <= hi]
        k = max(1, round(a.n * (hi - lo + 1) / total))
        sel = farthest(np.array([cands[i][1] for i in idx]), k, rng) if idx else []
        picked += [cands[idx[s]][0] for s in sel]
        print(f"  {name}: {len(sel)} of {len(idx)} moments")

    out = ROOT / "dataset" / a.recording
    for cam in caps:
        (out / cam).mkdir(parents=True, exist_ok=True)
    for f in sorted(picked):
        for cam, frame in (("camera_1", f), ("camera_2", f - off)):
            cv2.imwrite(str(out / cam / f"t{f:06d}.jpg"), read(cam, frame), [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"wrote {len(picked)} moments x 2 cameras -> {out}")


if __name__ == "__main__":
    main()
