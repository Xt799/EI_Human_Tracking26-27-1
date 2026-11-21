"""Build the crop dataset for the two-stage wheel model from training/ds_pose (run build_pose.py first).

    uv run python training/scripts/build_pose_crops.py [--size 320] [--scale 1.5]

Each robot is cut out as a square around its box (side = --scale x the longer box side, centred on the box), resized to
--size px; the wheel points move with it. At run time the robot model's box gives the same crop, so the pose model sees
the robot large and only the robot, and training can rotate the crop freely (any heading). Output: training/ds_pose_crop/.
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "training" / "ds_pose"
OUT = ROOT / "training" / "ds_pose_crop"


def crop_square(img, box, scale, size):
    """Crop (padded with grey outside the image) and the transform: crop px = (img px - origin) * factor."""
    x1, y1, x2, y2 = box
    side = scale * max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ox, oy = cx - side / 2, cy - side / 2
    f = size / side
    M = np.array([[f, 0, -ox * f], [0, f, -oy * f]], np.float32)
    out = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
    return out, (ox, oy, f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--scale", type=float, default=1.5)
    a = ap.parse_args()
    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "images").mkdir(parents=True)
    (OUT / "labels").mkdir(parents=True)
    for split in ("train", "val", "test"):
        paths = []
        for line in (SRC / f"{split}.txt").read_text().split():
            img_path = Path(line)
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            v = (SRC / "labels" / f"{img_path.stem}.txt").read_text().split()
            cx, cy, bw, bh = float(v[1]) * w, float(v[2]) * h, float(v[3]) * w, float(v[4]) * h
            box = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
            crop, (ox, oy, f) = crop_square(img, box, a.scale, a.size)
            kp = np.array(v[5:], float).reshape(4, 3)
            out = []
            for x, y, vis in kp:
                if vis == 0:
                    out += ["0", "0", "0"]
                    continue
                px, py = (x * w - ox) * f / a.size, (y * h - oy) * f / a.size
                out += [f"{px:.6f}", f"{py:.6f}", str(int(vis))]
            bx = [(box[0] - ox) * f / a.size, (box[1] - oy) * f / a.size, (box[2] - ox) * f / a.size, (box[3] - oy) * f / a.size]
            label = f"0 {(bx[0] + bx[2]) / 2:.6f} {(bx[1] + bx[3]) / 2:.6f} {bx[2] - bx[0]:.6f} {bx[3] - bx[1]:.6f} " + " ".join(out)
            cv2.imwrite(str(OUT / "images" / f"{img_path.stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (OUT / "labels" / f"{img_path.stem}.txt").write_text(label + "\n")
            paths.append((OUT / "images" / f"{img_path.stem}.jpg").as_posix())
        (OUT / f"{split}.txt").write_text("\n".join(paths) + "\n")
        print(f"{split}: {len(paths)} crops")
    data = (SRC / "data.yaml").read_text().replace(SRC.as_posix(), OUT.as_posix())
    (OUT / "data.yaml").write_text(data)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
