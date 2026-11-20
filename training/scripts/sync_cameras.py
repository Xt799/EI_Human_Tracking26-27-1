"""Find the time offset between two cameras that recorded the same scene, without a floor calibration.

    uv run python training/scripts/sync_cameras.py VIDEO_1 VIDEO_2 [--guess 0] [--search 30]

Result: time_1 = time_2 + offset, i.e. frame_1 = frame_2 + offset * fps, and how clear the match is.

1. Every --step-th frame of both videos: the people's ankle midpoints (pose model) and the bottom centre of the robot box
   (robot model), in pixels, lens-corrected when the video has the size of tracking/calibration/camera_matrix.npy.
   Cached in training/ds_sync/<video>_s<step>.npz, so a second run is fast.
2. Only MOVING points are used (nothing within MOVE_PX half a second before and after): a person or robot standing still
   fits every offset equally well and would hide the right one.
3. For each candidate offset, every moving point of view 2 is paired with every moving point of view 1 at the same moment,
   and a floor-to-floor homography is fitted with RANSAC. Wrong pairings scatter; at the right offset the true pairs agree
   on one floor mapping, so the number of pairs that fit it peaks.

A clear match has a peak several times higher than the best offset more than 2 s away; the script reports both. If there
is no clear peak, the two videos probably do not overlap in time (check the recording times and the scene content).

The guess can come from the file names: basler_record_<date>_<hh><mm><ss> is the recording start time; the two recording
PCs' clocks may differ by whole hours (time zone), so compare minutes and seconds.
"""
import argparse
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tracking"))
import common  # noqa: E402

CACHE = ROOT / "training" / "ds_sync"
KPT_CONF = 0.5
MOVE_PX = 15.0          # a point counts as moving if nothing was this close half a second before and after
RANSAC_PX = 30.0        # a pair fits the floor mapping if it lands this close (pixels in view 1)
MAX_PAIRS = 9           # skip moments with more combinations than this (crowds only add wrong pairs)


def extract(video, step, end_s):
    """Per sampled frame: robot point (or NaN) and people points, raw pixels. Cached per video and step."""
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{Path(video).stem}_s{step}.npz"
    if out.exists():
        d = np.load(out, allow_pickle=True)
        return d["frames"], d["robot"], list(d["people"]), float(d["fps"])
    from ultralytics import YOLO
    pose, obj = YOLO(common.POSE_MODEL_PATH), YOLO(common.OBJECT_MODEL_PATH)
    cap = cv2.VideoCapture(str(video))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last = n if end_s is None else min(n, int(end_s * fps))
    frames, robots, people = [], [], []
    for i in range(last):
        if not cap.grab():
            break
        if i % step:
            continue
        frame = cap.retrieve()[1]
        r = obj.predict(frame, imgsz=1920, conf=0.4, verbose=False)[0]
        rb = [b for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()) if obj.names[int(c)] == "robot"]
        robots.append(((rb[0][0] + rb[0][2]) / 2, rb[0][3]) if rb else (np.nan, np.nan))
        p = pose.predict(frame, conf=0.4, verbose=False)[0]
        feet = []
        if len(p.boxes):
            for xy, c in zip(p.keypoints.xy.cpu().numpy(), p.keypoints.conf.cpu().numpy()):
                if c[common.LEFT_ANKLE] > KPT_CONF and c[common.RIGHT_ANKLE] > KPT_CONF:
                    feet.append((xy[common.LEFT_ANKLE] + xy[common.RIGHT_ANKLE]) / 2)
        people.append(np.array(feet, float).reshape(-1, 2))
        frames.append(i)
        if len(frames) % 500 == 0:
            print(f"  {Path(video).name}: frame {i}/{last}", flush=True)
    np.savez(out, frames=np.array(frames), robot=np.array(robots, float), people=np.array(people, dtype=object), fps=fps)
    return np.array(frames), np.array(robots, float), people, fps


def undistorter(video):
    """Pixel -> lens-corrected pixel (as common.lens_correction does to frames), or identity for another frame size."""
    cap = cv2.VideoCapture(str(video))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if size != (1900, 1200):
        print(f"  {Path(video).name}: {size[0]}x{size[1]}, no lens calibration for this size: raw pixels")
        return lambda p: p
    K = np.load(common.CALIB_DIR + "/camera_matrix.npy")
    D = np.load(common.CALIB_DIR + "/distortion_coefficients.npy")
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, size, 0, size)
    return lambda p: p if len(p) == 0 else cv2.undistortPoints(p.reshape(-1, 1, 2).astype(np.float32), K, D, P=new_K).reshape(-1, 2)


def moving_points(video, step, end_s):
    """(times, per sampled frame: moving points in lens-corrected pixels)."""
    frames, robot, people, fps = extract(video, step, end_s)
    fix = undistorter(video)
    pts = [np.vstack([p, r[None]]) if not np.isnan(r[0]) else p.reshape(-1, 2) for p, r in zip(people, robot)]
    pts = [fix(np.asarray(p, np.float32)) for p in pts]
    half = max(1, int(round(0.5 * fps / step)))
    moving = []
    for k, p in enumerate(pts):
        keep = []
        for q in p:
            still = any(0 <= kk < len(pts) and len(pts[kk]) and np.min(np.linalg.norm(pts[kk] - q, axis=1)) < MOVE_PX
                        for kk in (k - half, k + half))
            if not still:
                keep.append(q)
        moving.append(np.array(keep, np.float32).reshape(-1, 2))
    return frames / fps, moving, fps


def score(offset_s, t1, m1, t2, m2):
    """Number of view-2 -> view-1 pairs at this offset that fit one floor homography, and the homography."""
    dt = t1[1] - t1[0]
    idx = np.round((t2 + offset_s - t1[0]) / dt).astype(int)
    a, b = [], []
    for j, i in enumerate(idx):
        if 0 <= i < len(t1) and len(m1[i]) and len(m2[j]) and len(m1[i]) * len(m2[j]) <= MAX_PAIRS:
            for p, q in itertools.product(m2[j], m1[i]):
                a.append(p)
                b.append(q)
    if len(a) < 8:
        return 0, None
    H, mask = cv2.findHomography(np.array(a), np.array(b), cv2.RANSAC, RANSAC_PX, maxIters=3000)
    return (int(mask.sum()) if mask is not None else 0), H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video1")
    ap.add_argument("video2")
    ap.add_argument("--guess", type=float, default=0.0, help="expected offset in seconds (time 1 = time 2 + offset)")
    ap.add_argument("--search", type=float, default=30.0, help="search guess +- this many seconds")
    ap.add_argument("--step", type=int, default=3, help="analyse every n-th frame")
    ap.add_argument("--end1", type=float, help="analyse video 1 only up to this many seconds")
    a = ap.parse_args()

    t1, m1, fps1 = moving_points(a.video1, a.step, a.end1)
    t2, m2, fps2 = moving_points(a.video2, a.step, None)
    if abs(fps1 - fps2) > 1e-3:
        print(f"WARNING: frame rates differ ({fps1} vs {fps2})")
    print(f"moving points per frame: view 1 {np.mean([len(m) for m in m1]):.2f}, view 2 {np.mean([len(m) for m in m2]):.2f}")

    dt = t1[1] - t1[0]
    offsets = np.arange(a.guess - a.search, a.guess + a.search + 1e-9, dt)
    scores = np.array([score(o, t1, m1, t2, m2)[0] for o in offsets])
    k = int(np.argmax(scores))
    far = scores[np.abs(offsets - offsets[k]) > 2.0]
    runner_up = int(far.max()) if len(far) else 0
    # sub-step estimate from the peak and its neighbours (parabola)
    best = offsets[k]
    if 0 < k < len(scores) - 1:
        y0, y1, y2 = scores[k - 1], scores[k], scores[k + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            best += 0.5 * (y0 - y2) / den * dt
    frames = int(round(best * fps1))
    ratio = scores[k] / max(runner_up, 1)
    print(f"offset {best:+.2f} s = {frames:+d} frames (+-{a.step} frames): {scores[k]} pairs fit one floor, "
          f"best offset more than 2 s away: {runner_up} ({ratio:.1f}x)")
    if ratio < 2:
        print("NO CLEAR MATCH: the videos probably do not overlap in time, or not enough moving people/robot in both views.")
    else:
        print(f"\nframe_1 = frame_2 {frames:+d}")


if __name__ == "__main__":
    main()
