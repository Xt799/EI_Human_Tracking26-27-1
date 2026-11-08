"""
Cut two simultaneous recordings into a pair of clips that start on the same frame, for track.py.

    uv run python tracking/tools/make_pair.py videos/basler_record_20261012_180049.mp4 low videos/AAA_basler_record_20261012_160049_FIXED.mp4 opposite 25

The last number is how many frames the second recording started earlier than the first (negative if later).
For our pair it is 25: the robot's first start, found frame by frame in both recordings, lands 25 frames later
in the opposite one. Writes videos/pair_<camera>.mp4 for both cameras, with equal frame counts.
"""
import argparse
import os

import cv2

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "videos")


def open_video(path, skip):
    """Opens the video and skips its first frames."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {path}")
    for _ in range(skip):
        cap.grab()
    return cap


def main(video_a, camera_a, video_b, camera_b, offset, out_dir=OUT_DIR):
    caps = [open_video(video_a, max(0, -offset)), open_video(video_b, max(0, offset))]
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 30.0
    os.makedirs(out_dir, exist_ok=True)
    paths = [os.path.join(out_dir, f"pair_{cam}.mp4") for cam in (camera_a, camera_b)]
    writers, n = [], 0
    while True:
        frames = [cap.read() for cap in caps]
        if not all(ok for ok, _ in frames):
            break
        if not writers:
            writers = [cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), fps, (f.shape[1], f.shape[0])) for p, (_, f) in zip(paths, frames)]
        for w, (_, f) in zip(writers, frames):
            w.write(f)
        n += 1
    for cap in caps:
        cap.release()
    for w in writers:
        w.release()
    print(f"{n} frames written to {paths[0]} and {paths[1]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("video_a", help="first recording")
    ap.add_argument("camera_a", help="camera position of the first recording")
    ap.add_argument("video_b", help="second recording, made at the same time")
    ap.add_argument("camera_b", help="camera position of the second recording")
    ap.add_argument("offset", type=int, help="frames the second recording started earlier than the first (negative: later)")
    a = ap.parse_args()
    main(a.video_a, a.camera_a, a.video_b, a.camera_b, a.offset)
