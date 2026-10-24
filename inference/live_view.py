"""Show the fine-tuned robot/ball detector live on a video.

uv run python live_view.py [--video VIDEO] [--weights best.pt] [--conf 0.5] [--iou 0.45] [--speed 1.0] [--imgsz 2560]

Defaults: the largest video in ../videos (the test set) and ../training/runs/yolo26n_e30/weights/best.pt.
Keys:  q/Esc quit | space pause | d or . step one frame (paused) | +/- change conf | [ / ] halve/double speed | s save screenshot
"""
import argparse
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
COLORS = {"robot": (0, 200, 255), "ball": (0, 255, 0)}   # BGR
WINDOW = "human_tracking - live"


def largest_video() -> Path:
    videos = sorted((ROOT / "videos").glob("*.mp4"), key=lambda p: p.stat().st_size)
    if not videos:
        raise SystemExit(f"No .mp4 files in {ROOT / 'videos'}")
    return videos[-1]


def draw(frame, result, names):
    for (x1, y1, x2, y2), c, k in zip(result.boxes.xyxy.cpu().numpy().astype(int),
                                      result.boxes.conf.cpu().numpy(),
                                      result.boxes.cls.cpu().numpy().astype(int)):
        color = COLORS.get(names[k], (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f"{names[k]} {c:.2f}", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=None, help="default: the largest video in ../videos")
    ap.add_argument("--weights", type=Path, default=ROOT / "training" / "runs" / "yolo26n_e30" / "weights" / "best.pt")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (lower = fewer duplicate boxes)")
    ap.add_argument("--no-agnostic", action="store_true", help="run NMS per class instead of across classes")
    ap.add_argument("--imgsz", type=int, default=2560, help="inference size; 2560 finds the small far-away ball much better than 1920")
    ap.add_argument("--start", type=float, default=0, help="start position in seconds")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback speed relative to the video clock (1 = real time, 4 = 4x); frames are skipped when the detector cannot keep up")
    ap.add_argument("--display-width", type=int, default=1280, help="width of the shown window in pixels")
    a = ap.parse_args()

    video = a.video or largest_video()
    if not a.weights.exists():
        raise SystemExit(f"Weights not found: {a.weights} (train first, or pass --weights)")
    device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = YOLO(str(a.weights))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"video: {video}\nweights: {a.weights}\ndevice: {device}")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    conf, paused, step, fps_ema, shot, speed = a.conf, False, False, 0.0, 0, a.speed
    anchor_t, anchor_f = time.perf_counter(), int(cap.get(cv2.CAP_PROP_POS_FRAMES))   # video clock: frame anchor_f at wall time anchor_t
    shown = None
    while True:
        ahead = False
        if not paused or step:
            if not step:   # real-time pacing; single-stepping ignores the clock
                behind = anchor_f + (time.perf_counter() - anchor_t) * video_fps * speed - cap.get(cv2.CAP_PROP_POS_FRAMES)
                for _ in range(int(behind)):   # detector too slow for this speed -> drop frames
                    cap.grab()
                if behind < 0:   # ahead of the clock -> wait, but keep handling keys
                    time.sleep(min(-behind / (video_fps * speed), 0.1))
                    ahead = True
        if (not paused or step) and not ahead:
            ok, frame = cap.read()
            if not ok:
                print("End of video.")
                break
            t0 = time.perf_counter()
            result = model(frame, imgsz=a.imgsz, conf=conf, iou=a.iou, agnostic_nms=not a.no_agnostic,
                           device=device, verbose=False)[0]
            dt = time.perf_counter() - t0
            fps_ema = 1 / dt if fps_ema == 0 else 0.9 * fps_ema + 0.1 / dt
            draw(frame, result, model.names)
            counts = {n: int((result.boxes.cls == i).sum()) for i, n in model.names.items()}
            h, w = frame.shape[:2]
            shown = cv2.resize(frame, (a.display_width, int(h * a.display_width / w)))
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            info = f"{pos}/{total}  {speed:g}x  {fps_ema:.1f} det/s  conf>={conf:.2f}  " + "  ".join(f"{n}:{c}" for n, c in counts.items())
            cv2.putText(shown, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(shown, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            step = False
        if shown is not None:
            cv2.imshow(WINDOW, shown)
        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        elif key == ord(" "):
            paused = not paused
            anchor_t, anchor_f = time.perf_counter(), int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        elif key in (ord("["), ord("]")):
            speed = min(64.0, speed * 2) if key == ord("]") else max(0.125, speed / 2)
            anchor_t, anchor_f = time.perf_counter(), int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        elif key in (ord("d"), ord(".")) and paused:
            step = True
        elif key in (ord("+"), ord("=")):
            conf = min(0.95, conf + 0.05)
        elif key in (ord("-"), ord("_")):
            conf = max(0.05, conf - 0.05)
        elif key == ord("s") and shown is not None:
            shot += 1
            cv2.imwrite(f"screenshot_{shot:03d}.png", shown)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
