"""Build the YOLO keypoint dataset of the robot heading model (front + rear point) from the reviewed crops in
dataset/robot_heading/.

    uv run python training/scripts/build_heading.py                       # train / val as the folders
    uv run python training/scripts/build_heading.py --train-room room_2   # train on one room, validate on the others
    uv run python training/scripts/build_heading.py --folds 5             # cross-validation: training/ds_heading_cv<k>/
    uv run python training/scripts/build_heading.py --all                 # final model: train on everything

--all: once the recipe has been measured with the train / val folders, the final weights are trained on all reviewed
crops (training/ds_heading_all/). Its val list is the val folder again, only so that training runs: those images are
also trained on, so its scores mean nothing; take last.pt, trained as many epochs as the measured run.

Only reviewed crops are used: both points present and not exactly the pre-labels of
dataset/robot_heading/prelabels.json. A crop whose points were all deleted is a false robot detection (something else the robot model took for the robot): it goes
in as a negative, an image with no robot, so the model learns not to give a heading there.
The stored crop is the model input as it is (the square of CROP_SCALE x the robot box around the robot, the same crop as
at run time); the robot box is its middle part. Keypoints: front, rear (visibility 2); a mirrored image keeps them
(flip_idx [0, 1]). --folds: the reviewed crops in time order (recording, camera, frame) are cut into 4 x that many
contiguous stretches, dealt out to the folds in turn (so every fold has every room); fold k validates on its stretches
and trains on the others.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from robot_crops import CROP_SCALE, HEADING_DIR, parse_name  # noqa: E402

KPTS = ("front", "rear")


def reviewed_crops():
    """[(name, image path, {label: (x, y)}, split, (width, height))] of the reviewed crops, in time order
    (recording, camera, frame). The split is the folder the crop is in (dataset/robot_heading/train or val)."""
    pre_path = HEADING_DIR / "prelabels.json"
    pre = json.loads(pre_path.read_text(encoding="utf-8")) if pre_path.exists() else {}
    out = []
    for split in ("train", "val"):
        for js in (HEADING_DIR / split).glob("*.json"):
            d = json.loads(js.read_text(encoding="utf-8"))
            pts = {s["label"]: [round(v, 2) for v in s["points"][0]] for s in d["shapes"] if s["label"] in KPTS}
            if (pts and len(pts) < 2) or pre.get(js.stem) == pts:  # one point only, or not reviewed (empty = negative)
                continue
            out.append((js.stem, js.with_suffix(".jpg"), pts, split, (d["imageWidth"], d["imageHeight"])))
    return sorted(out, key=lambda c: parse_name(c[0])[:3])


def write(out_dir, parts):
    """parts: {split: [crop, ...]} -> YOLO pose dataset in out_dir."""
    shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / "images").mkdir(parents=True)
    (out_dir / "labels").mkdir(parents=True)
    b = 1 / CROP_SCALE                     # robot box: the middle 1/CROP_SCALE of the crop
    for split, crops in parts.items():
        paths = []
        for stem, img, pts, _, (w, h) in crops:
            shutil.copy(img, out_dir / "images" / f"{stem}.jpg")
            if pts:
                kp = " ".join(f"{pts[k][0] / w:.6f} {pts[k][1] / h:.6f} 2" for k in KPTS)
                (out_dir / "labels" / f"{stem}.txt").write_text(f"0 0.5 0.5 {b:.6f} {b:.6f} {kp}\n")
            else:
                (out_dir / "labels" / f"{stem}.txt").write_text("")              # negative: no robot
            paths.append((out_dir / "images" / f"{stem}.jpg").as_posix())
        (out_dir / f"{split}.txt").write_text("\n".join(paths) + "\n")
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.as_posix()}\n" + "".join(f"{s}: {s}.txt\n" for s in parts) +
        f"kpt_shape: [2, 3]\nflip_idx: [0, 1]\nkpt_names:\n  0: [{', '.join(KPTS)}]\nnames:\n  0: robot\n")
    print(f"{out_dir.name}: " + ", ".join(f"{s} {len(c)} ({sum(1 for x in c if not x[2])} negative)" for s, c in parts.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--train-room", help="train on this room only (all its crops) and validate on the other rooms: "
                                         "how well the heading carries over to a room never seen")
    ap.add_argument("--all", action="store_true", help="train on all reviewed crops (final weights)")
    a = ap.parse_args()
    crops = reviewed_crops()
    print(f"{len(crops)} reviewed crops")
    if a.folds:
        chunks = 4 * a.folds                       # short stretches of time, dealt out in turn: every fold gets all rooms
        fold_of = [i * chunks // len(crops) % a.folds for i in range(len(crops))]
        for k in range(a.folds):
            write(ROOT / "training" / f"ds_heading_cv{k}", {"train": [c for c, f in zip(crops, fold_of) if f != k],
                                                             "val": [c for c, f in zip(crops, fold_of) if f == k]})
    elif a.train_room:
        in_room = lambda c: c[0].startswith(a.train_room + "_")
        write(ROOT / "training" / f"ds_heading_{a.train_room}", {"train": [c for c in crops if in_room(c)],
                                                                 "val": [c for c in crops if not in_room(c)]})
    elif a.all:
        write(ROOT / "training" / "ds_heading_all", {"train": crops, "val": [c for c in crops if c[3] == "val"]})
    else:
        write(ROOT / "training" / "ds_heading", {s: [c for c in crops if c[3] == s] for s in ("train", "val")})


if __name__ == "__main__":
    main()
