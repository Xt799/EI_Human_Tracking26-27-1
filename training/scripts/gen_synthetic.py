"""Add synthetic images (cut-out robots/balls pasted on other backgrounds) to the dataset built by build_full.py.

Run from the repository root, after build_full.py:
  uv run python training/scripts/gen_synthetic.py [--n 300] [--preview]

Backgrounds: every image under dataset/backgrounds/ (put photos of other rooms there). If the folder is empty,
coco128 (128 generic photos) is downloaded into it. Objects are pasted with random scale/rotation/colour, may be cut off by the image
border or partly covered by background patches / other objects; the label is the box of the VISIBLE part (kept only if >= MIN_VISIBLE
of the object is visible). The whole image then gets random blur, noise and JPEG compression.
The images are appended to ds_full/train.txt once per epoch (the real images are repeated, the synthetic ones are not).
"""
import argparse
import random
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
CUTOUTS = ROOT / "dataset" / "train"
BG_DIR = ROOT / "dataset" / "backgrounds"
OUT = ROOT / "training" / "ds_full"
CLASSES = ["robot", "ball"]
W, H = 1900, 1200          # same size as the real frames
MIN_VISIBLE = 0.3          # fraction of an object that must stay visible to be labelled (and kept)
COCO128 = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"


def load_cutouts(cls):
    out = []
    for f in sorted((CUTOUTS / cls / "cutout").glob("*.png")):
        im = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        ys, xs = np.where(im[..., 3] > 10)
        out.append(im[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    return out


def list_backgrounds():
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [p for p in BG_DIR.rglob("*") if p.suffix.lower() in exts]
    if not files:
        from ultralytics.utils.downloads import download
        print(f"no backgrounds in {BG_DIR}, downloading coco128 ...")
        BG_DIR.mkdir(parents=True, exist_ok=True)
        download(COCO128, dir=BG_DIR)
        files = [p for p in BG_DIR.rglob("*") if p.suffix.lower() in exts]
    return files


def random_background(files, rng):
    im = cv2.imread(str(rng.choice(files)))
    h, w = im.shape[:2]
    s = max(W / w, H / h) * rng.uniform(1.0, 1.6)
    im = cv2.resize(im, (int(w * s) + 1, int(h * s) + 1))
    y, x = rng.randint(0, im.shape[0] - H), rng.randint(0, im.shape[1] - W)
    im = im[y:y + H, x:x + W]
    if rng.random() < 0.5:
        im = im[:, ::-1]
    return np.ascontiguousarray(im)


def jitter(bgra, rng):
    """Random brightness/contrast/colour on the cut-out, so the object does not always match the real lighting."""
    rgb = bgra[..., :3].astype(np.float32)
    rgb = rgb * rng.uniform(0.65, 1.35) + rng.uniform(-20, 20)
    rgb *= np.array([rng.uniform(0.9, 1.1) for _ in range(3)], np.float32)
    out = bgra.copy()
    out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return out


def transform(cutout, cls, rng):
    """Random scale (distance), rotation and flip. Returns BGRA."""
    im = jitter(cutout, rng)
    s = rng.uniform(0.7, 1.6) if cls == "robot" else rng.uniform(0.7, 1.8)
    if rng.random() < 0.5:
        im = im[:, ::-1]
    im = cv2.resize(np.ascontiguousarray(im), None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    ang = rng.uniform(-15, 15) if cls == "robot" else rng.uniform(-180, 180)
    h, w = im.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1)
    c, s_ = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s_ + w * c) + 2, int(h * c + w * s_) + 2
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(im, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


def place(canvas, owner, sprite, oid, rng):
    """Paste sprite at a random place (may stick out of the image). Returns the unclipped pixel count of the sprite."""
    h, w = sprite.shape[:2]
    x0 = rng.randint(int(-0.5 * w), int(W - 0.5 * w))
    y0 = rng.randint(int(-0.5 * h), int(H - 0.5 * h))
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    full = int((sprite[..., 3] > 127).sum())
    if cx1 <= cx0 or cy1 <= cy0:
        return full
    sp = sprite[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    a = cv2.GaussianBlur(sp[..., 3], (3, 3), 0).astype(np.float32)[..., None] / 255
    roi = canvas[cy0:cy1, cx0:cx1]
    canvas[cy0:cy1, cx0:cx1] = (sp[..., :3] * a + roi * (1 - a)).astype(np.uint8)
    owner[cy0:cy1, cx0:cx1][sp[..., 3] > 127] = oid
    return full


def occluder(files, rng):
    """A random patch cut from another background, with an elliptical/rectangular soft mask. Returns BGRA."""
    bg = random_background(files, rng)
    w, h = rng.randint(40, 300), rng.randint(40, 300)
    x, y = rng.randint(0, W - w), rng.randint(0, H - h)
    patch = bg[y:y + h, x:x + w].copy()
    mask = np.zeros((h, w), np.uint8)
    if rng.random() < 0.5:
        cv2.ellipse(mask, (w // 2, h // 2), (w // 2 - 1, h // 2 - 1), 0, 0, 360, 255, -1)
    else:
        mask[:] = 255
    return np.dstack([patch, mask])


def degrade(img, rng):
    if rng.random() < 0.7:
        k = rng.uniform(0.3, 1.4)
        img = cv2.GaussianBlur(img, (0, 0), k)
    if rng.random() < 0.25:   # motion blur
        n = rng.randint(3, 9)
        kern = np.zeros((n, n), np.float32)
        kern[n // 2, :] = 1 / n
        M = cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), rng.uniform(0, 180), 1)
        img = cv2.filter2D(img, -1, cv2.warpAffine(kern, M, (n, n)) / max(kern.sum(), 1e-6))
    sigma = rng.uniform(0, 6)
    if sigma > 0.5:
        noise = np.random.default_rng(rng.randint(0, 2 ** 31)).normal(0, sigma, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    q = rng.randint(50, 95)
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def make_image(robots, balls, files, rng):
    canvas = random_background(files, rng)
    owner = np.full((H, W), -2, np.int32)     # -2 background, -1 occluder patch, >=0 object id
    objs = []                                 # (cls_idx, full_pixel_count)
    n_robot = rng.choices([0, 1, 2], [0.1, 0.65, 0.25])[0]
    n_ball = rng.choices([0, 1, 2, 3], [0.2, 0.5, 0.2, 0.1])[0]
    for cls, n, pool in (("robot", n_robot, robots), ("ball", n_ball, balls)):
        for _ in range(n):
            sprite = transform(rng.choice(pool), cls, rng)
            objs.append((CLASSES.index(cls), place(canvas, owner, sprite, len(objs), rng)))
    for _ in range(rng.choice([0, 0, 1, 2, 3])):   # occluders, undone if they hide too much of an object
        before_c, before_o = canvas.copy(), owner.copy()
        place(canvas, owner, occluder(files, rng), -1, rng)
        if any((owner == i).sum() < MIN_VISIBLE * full for i, (_, full) in enumerate(objs)
               if (before_o == i).sum() >= MIN_VISIBLE * full):
            canvas, owner = before_c, before_o
    labels = []
    for i, (cls, full) in enumerate(objs):
        ys, xs = np.where(owner == i)
        if len(xs) < MIN_VISIBLE * full or xs.max() - xs.min() < 5 or ys.max() - ys.min() < 5:
            continue   # too little visible: do not label (and the rare ambiguous case is accepted)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        labels.append(f"{cls} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
    return degrade(canvas, rng), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", action="store_true", help="also write ds_full/synth_preview.jpg (a grid with the labels drawn)")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    robots, balls = load_cutouts("robot"), load_cutouts("ball")
    files = list_backgrounds()
    print(f"{len(robots)} robot and {len(balls)} ball cut-outs, {len(files)} backgrounds")
    assert (OUT / "train.txt").exists(), "run build_full.py first"

    for old in (OUT / "images").glob("synth_*"):
        old.unlink()
    for old in (OUT / "labels").glob("synth_*"):
        old.unlink()
    paths, preview = [], []
    for i in range(a.n):
        img, labels = make_image(robots, balls, files, rng)
        p = OUT / "images" / f"synth_{i:05d}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (OUT / "labels" / f"synth_{i:05d}.txt").write_text("\n".join(labels))
        paths.append(p.as_posix())
        if a.preview and len(preview) < 12:
            vis = img.copy()
            for l in labels:
                c, cx, cy, bw, bh = l.split()
                x0, y0 = int((float(cx) - float(bw) / 2) * W), int((float(cy) - float(bh) / 2) * H)
                x1, y1 = int((float(cx) + float(bw) / 2) * W), int((float(cy) + float(bh) / 2) * H)
                col = (0, 200, 255) if c == "0" else (0, 255, 0)
                cv2.rectangle(vis, (x0, y0), (x1, y1), col, 3)
            preview.append(cv2.resize(vis, (W // 3, H // 3)))
    # drop the lines of an earlier run, then append
    lines = [l for l in (OUT / "train.txt").read_text().splitlines() if l and "/synth_" not in l]
    (OUT / "train.txt").write_text("\n".join(lines + paths) + "\n")
    if preview:
        rows = [np.hstack(preview[r * 4:r * 4 + 4]) for r in range(len(preview) // 4)]
        cv2.imwrite(str(OUT / "synth_preview.jpg"), np.vstack(rows))
    print(f"{len(paths)} synthetic images added to {OUT / 'train.txt'} ({len(lines)} real lines)")


if __name__ == "__main__":
    main()
