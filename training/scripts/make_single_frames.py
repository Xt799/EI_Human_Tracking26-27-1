"""Pick robot views of one camera that the dataset does not have yet, and pre-label their wheels with the pose model.

    uv run python training/scripts/make_single_frames.py room_2_20261110_1709 camera_1 [--n 50] [--step 10]

For cameras without a partner (or whose partner is not in sync): every image stands alone, named t<own frame>.jpg.
1. Candidates: every --step-th frame where the robot model finds the robot.
   In a paired recording the paired stretch is skipped (make_paired_frames.py covers it).
2. Diversity: farthest-point sampling on a small grey thumbnail of the robot plus its place in the image, per split
   (splits.yaml), in proportion to the split's number of candidates; images already in the folder count as picked, so a rerun adds
   only new views. Near-identical views (same place, same look, nothing in front) are not picked twice.
3. Pre-labels: the two-stage pose model (robot box -> crop -> yolo26n_pose_crop) puts wheel_1..wheel_4 on each image as
   ordinary points, to be corrected by hand in X-AnyLabeling.
Images are lens-corrected (what tracking/ feeds the models): dataset/<recording>/<camera>/t<frame>.jpg + .json.
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
sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402
from complete_wheel_labels import WHEELS, load, shape  # noqa: E402
from dataset_splits import load as load_splits, split_of  # noqa: E402
from make_paired_frames import farthest, robot_box, thumb  # noqa: E402
from render_pose_video import predict  # noqa: E402

PLACE_WEIGHT = 2.0       # weight of the robot's place in the image (0..1 per axis) next to the look (thumbnail scaled to length 1)


def feature(img, box):
    h, w = img.shape[:2]
    c = np.array([(box[0] + box[2]) / 2 / w, (box[1] + box[3]) / 2 / h])
    return np.concatenate([thumb(img, box) / np.sqrt(thumb(img, box).size), PLACE_WEIGHT * c])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("camera")
    ap.add_argument("--n", type=int, default=50, help="number of new images")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--weights", default="training/runs/yolo26n_pose_crop/weights/best.pt")
    a = ap.parse_args()

    rec = yaml.safe_load((ROOT / "dataset" / "recordings.yaml").read_text())["recordings"][a.recording]
    splits = load_splits()
    from ultralytics import YOLO
    detector = YOLO(common.OBJECT_MODEL_PATH)
    pose = YOLO(str(ROOT / a.weights))
    cap = cv2.VideoCapture(str(ROOT / rec["cameras"][a.camera]["video"]))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fix = common.lens_correction(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = ROOT / "dataset" / a.recording / a.camera
    out.mkdir(parents=True, exist_ok=True)

    def read(frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        return fix(img) if ok else None

    existing = {int(p.stem[1:]) for p in out.glob("t*.jpg")}
    skip = range(rec["paired_frames"][0], rec["paired_frames"][1] + 1) if rec.get("paired_frames") else range(0)
    existing = {f for f in existing if f not in skip}
    cands = []                                   # (frame, feature, already in the dataset)
    for f in sorted({f for f in range(0, n_frames, a.step) if f not in skip} | existing):
        img = read(f)
        if img is None:
            continue
        box = robot_box(detector, img)
        if box is None:
            continue
        cands.append((f, feature(img, box), f in existing))
        if len(cands) % 200 == 0:
            print(f"  {len(cands)} candidates (frame {f}/{n_frames})", flush=True)
    print(f"{len(cands)} frames with the robot, {len(existing)} already in the dataset")

    by_split = {}
    for i, (f, _, _) in enumerate(cands):
        by_split.setdefault(split_of(splits, a.recording, a.camera, f), []).append(i)
    by_split.pop(None, None)
    total = sum(len(v) for v in by_split.values())
    picked = []
    for name, idx in by_split.items():
        feats = np.array([cands[i][1] for i in idx])
        have = [j for j, i in enumerate(idx) if cands[i][2]]
        k = max(1, round(a.n * len(idx) / total))
        if have:                                 # continue the farthest-point sampling from what is there
            d = np.min([np.linalg.norm(feats - feats[j], axis=1) for j in have], axis=0)
            new = []
            while len(new) < k and d.max() > 0:
                j = int(np.argmax(d))
                new.append(j)
                d = np.minimum(d, np.linalg.norm(feats - feats[j], axis=1))
        else:
            new = farthest(feats, k, np.random.default_rng(0))
        picked += [cands[idx[j]][0] for j in new]
        print(f"  {name}: {len(new)} new of {len(idx)} candidates")

    for f in sorted(picked):
        img = read(f)
        name = f"t{f:06d}"
        cv2.imwrite(str(out / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        d = load(out / f"{name}.json", img.shape)
        res = predict(pose, img, 1920, detector)
        if res is not None:
            d["shapes"] += [shape(w, p, "", False) for w, p in zip(WHEELS, res[0])]
        (out / f"{name}.json").write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(picked)} images with pre-labels -> {out}")


if __name__ == "__main__":
    main()
