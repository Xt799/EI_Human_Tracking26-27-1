"""Render a check video of the robot + wheel pose model: wheel points and heading drawn on every frame.

    uv run python training/scripts/render_pose_video.py --recording room_1_20261012_1800
    uv run python training/scripts/render_pose_video.py --video videos/room_2/camera_1/basler_record_20261110_170937.mp4 --seconds 0 180

Paired recording (--recording): both cameras side by side on the camera_1 timeline, a zoomed view of the robot per
camera, and a floor map (when both cameras have one) with the heading from each camera and from both together.
Single video (--video): one view and its zoom. Frames in a train/val/test stretch of dataset/splits.yaml are marked,
so the test part (never trained on) can be told apart.

Drawing: wheel_1..wheel_4 as numbered dots (filled = confident, ring = point confidence < 0.5); the arrow goes from
the back to the front of the robot, computed in the image as evaluate_pose.py does on the floor.
Default model: two-stage (robot model box -> crop -> yolo26n_pose_crop); --full with --weights for the one-stage model.
Output: results/pose_<name>.mp4 (results/ is not in git).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tracking"))
sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402
from dataset_splits import load as load_splits, split_of  # noqa: E402
from evaluate_pose import heading  # noqa: E402

WHEEL_COLOURS = [(0, 0, 255), (0, 165, 255), (255, 0, 0), (255, 255, 0)]     # BGR: wheel_1 red, 2 orange, 3 blue, 4 cyan
CAM_COLOURS = {"camera_1": (60, 200, 60), "camera_2": (200, 60, 200)}
KPT_SURE = 0.5
VIEW_W, ZOOM, MAP = 950, 400, 400      # view width, zoom panel side, map panel height


def predict(model, img, imgsz, detector=None, crop_size=320, crop_scale=1.5):
    """(xy (4, 2), conf (4,), box xyxy) of the most confident robot, or None.
    With a detector: two-stage, the robot model's box -> square crop -> pose model (as evaluate_pose.py --crop)."""
    if detector is None:
        r = model.predict(img, imgsz=imgsz, conf=0.25, verbose=False)[0]
        if len(r.boxes) == 0:
            return None
        k = int(r.boxes.conf.argmax())
        return r.keypoints.xy[k].cpu().numpy(), r.keypoints.conf[k].cpu().numpy(), r.boxes.xyxy[k].cpu().numpy()
    from build_pose_crops import crop_square
    d = detector.predict(img, imgsz=1920, conf=0.3, verbose=False)[0]
    boxes = [(c, b) for b, c, k in zip(d.boxes.xyxy.cpu().numpy(), d.boxes.conf.cpu().numpy(), d.boxes.cls.cpu().numpy())
             if detector.names[int(k)] == "robot"]
    if not boxes:
        return None
    box = max(boxes, key=lambda x: x[0])[1]
    crop, (ox, oy, f) = crop_square(img, box, crop_scale, crop_size)
    r = model.predict(crop, imgsz=crop_size, conf=0.05, verbose=False)[0]
    if len(r.boxes) == 0:
        return None
    k = int(r.boxes.conf.argmax())
    return r.keypoints.xy[k].cpu().numpy() / f + (ox, oy), r.keypoints.conf[k].cpu().numpy(), box


def draw_pose(img, res, scale=1.0, thick=2, ext=0.8):
    """Draw wheel dots and the image-space heading arrow onto img (coordinates of res are multiplied by scale)."""
    if res is None:
        return
    xy, conf, _ = res
    p = xy * scale
    L = np.linalg.norm(p[[2, 3]].mean(0) - p[[0, 1]].mean(0))          # arrow length ~ robot width in the image
    centre = p.mean(0)
    h = heading(dict(enumerate(p)))
    if h is None:
        return
    rear, tip = centre - 0.5 * h * L, centre + (0.5 + ext) * h * L
    cv2.arrowedLine(img, tuple(map(int, rear)), tuple(map(int, tip)), (255, 255, 255), thick + 2, cv2.LINE_AA, tipLength=0.3)
    cv2.arrowedLine(img, tuple(map(int, rear)), tuple(map(int, tip)), (0, 255, 0), thick, cv2.LINE_AA, tipLength=0.3)
    r = max(3, int(4 * thick))
    for i, (pt, c) in enumerate(zip(p, conf)):
        q = tuple(map(int, pt))
        cv2.circle(img, q, r, WHEEL_COLOURS[i], -1 if c >= KPT_SURE else thick, cv2.LINE_AA)
        cv2.putText(img, str(i + 1), (q[0] + r, q[1] - r), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * thick, (255, 255, 255), thick + 1, cv2.LINE_AA)
        cv2.putText(img, str(i + 1), (q[0] + r, q[1] - r), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * thick, WHEEL_COLOURS[i], thick - 1 or 1, cv2.LINE_AA)


def zoom_panel(img, res, title):
    panel = np.full((ZOOM, ZOOM, 3), 40, np.uint8)
    if res is not None:
        sys.path.insert(0, str(Path(__file__).parent))
        from build_pose_crops import crop_square
        crop, (ox, oy, f) = crop_square(img, res[2], 1.6, ZOOM)
        panel = crop
        draw_pose(panel, ((res[0] - (ox, oy)) * f, res[1], res[2]), 1.0, 2, 0.3)
        txt = "wheel conf " + " ".join(f"{c:.2f}" for c in res[1])
    else:
        txt = "no robot"
    common.put_label(panel, title, (8, 22))
    common.put_label(panel, txt, (8, ZOOM - 10))
    return panel


def map_panel(floor_res, room, width):
    """Top-down floor map: each camera's wheel points and heading, and the fused heading (black)."""
    (x0, x1), (y0, y1) = room
    s = min((width - 40) / (x1 - x0), (MAP - 70) / (y1 - y0))
    mx, my = (width - (x1 - x0) * s) / 2, MAP - 30 - (MAP - 70 - (y1 - y0) * s) / 2
    panel = np.full((MAP, width, 3), 255, np.uint8)
    to_px = lambda q: (int(mx + (q[0] - x0) * s), int(my - (q[1] - y0) * s))
    cv2.rectangle(panel, to_px((x0, y0)), to_px((x1, y1)), (180, 180, 180), 1)
    vs, centre = [], []
    for cam, (pts, conf) in floor_res.items():
        h = heading(dict(enumerate(pts)))
        c = pts.mean(0)
        centre.append(c)
        for i, q in enumerate(pts):
            cv2.circle(panel, to_px(q), 3, WHEEL_COLOURS[i], -1)
        if h is not None:
            vs.append((h, float(conf.mean())))
            cv2.arrowedLine(panel, to_px(c), to_px(c + 0.5 * h), CAM_COLOURS[cam], 2, cv2.LINE_AA, tipLength=0.3)
    if vs:
        v = sum(h * w for h, w in vs)
        if np.linalg.norm(v) > 1e-6:
            c = np.mean(centre, 0)
            cv2.arrowedLine(panel, to_px(c), to_px(c + 0.8 * v / np.linalg.norm(v)), (0, 0, 0), 2, cv2.LINE_AA, tipLength=0.3)
    if len(vs) == 2:
        common.put_label(panel, f"cameras disagree by {np.degrees(np.arccos(np.clip(vs[0][0] @ vs[1][0], -1, 1))):.0f} deg", (width - 310, MAP - 10))
    common.put_label(panel, "floor map: camera_1 green, camera_2 purple, both black", (8, 22))
    return panel


def open_video(path, start):
    cap = cv2.VideoCapture(str(ROOT / path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, common.lens_correction(w, h), cap.get(cv2.CAP_PROP_FPS) or 30.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--recording", help="paired recording name in dataset/recordings.yaml")
    g.add_argument("--video", help="single video path (relative to the repo)")
    ap.add_argument("--weights", default="training/runs/yolo26n_pose_crop/weights/best.pt")
    ap.add_argument("--full", action="store_true", help="one-stage pose model on the full frame (--weights must be such a model)")
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--seconds", type=float, nargs=2, help="only this stretch (camera_1 timeline)")
    ap.add_argument("--out")
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(str(ROOT / a.weights))
    detector = None if a.full else YOLO(common.OBJECT_MODEL_PATH)
    splits = load_splits()
    recordings = yaml.safe_load((ROOT / "dataset" / "recordings.yaml").read_text())["recordings"]

    if a.recording:
        rec = recordings[a.recording]
        cams = rec["cameras"]
        lo, hi = rec["paired_frames"]
        name = a.recording
    else:
        cams = {"camera_1": {"video": a.video, "floor_map": None}}
        cap = cv2.VideoCapture(str(ROOT / a.video))
        lo, hi = 0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        cap.release()
        name = Path(a.video).stem
        # the recording and camera this video is, for the split marks
        a.recording, split_cam = next(((r, c) for r, v in recordings.items() for c, cv in v["cameras"].items()
                                       if Path(cv["video"]) == Path(a.video)), (None, None))
    fps = 30.0
    if a.seconds:
        lo, hi = max(lo, int(a.seconds[0] * fps)), min(hi, int(a.seconds[1] * fps))
    off = rec["offset_frames"] if not a.video else 0
    caps = {c: open_video(v["video"], lo - (off if c == "camera_2" else 0)) for c, v in cams.items()}
    floor = {}
    if all(v.get("floor_map") for v in cams.values()):
        floor = {c: common.load_floor_map(v["floor_map"]) for c, v in cams.items()}
        room, _ = common.load_room([v["floor_map"] for v in cams.values()])

    out_path = Path(a.out) if a.out else ROOT / "results" / f"pose_{name}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    for frame_no in range(lo, hi + 1):
        views, zooms, floor_res = [], [], {}
        ok = True
        for c, (cap, fix, _) in caps.items():
            got, img = cap.read()
            if not got:
                ok = False
                break
            img = fix(img)
            res = predict(model, img, a.imgsz, detector)
            if res is not None and c in floor:
                floor_res[c] = (cv2.perspectiveTransform(res[0].reshape(-1, 1, 2).astype(np.float32), floor[c])[:, 0], res[1])
            zooms.append(zoom_panel(img, res, c))
            s = VIEW_W / img.shape[1]
            small = cv2.resize(img, (VIEW_W, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
            draw_pose(small, res, s, 1)
            common.put_label(small, c, (8, 22))
            views.append(small)
        if not ok:
            break
        top = np.hstack(views)
        bottom = zooms + ([map_panel(floor_res, room, top.shape[1] - ZOOM * len(zooms))] if floor else [])
        bottom = np.hstack(bottom)
        W = max(top.shape[1], bottom.shape[1])
        pad = lambda x: cv2.copyMakeBorder(x, 0, 0, 0, W - x.shape[1], cv2.BORDER_CONSTANT, value=(30, 30, 30))
        canvas = np.vstack([pad(top), pad(bottom)])
        sp = split_of(splits, a.recording, split_cam if a.video else "camera_1", frame_no) if a.recording else None
        tag = f"frame {frame_no}  t={frame_no / fps:6.1f}s" + (f"  split: {sp.upper()}" if sp else "")
        common.put_label(canvas, tag, (8 + ZOOM * len(zooms), top.shape[0] + ZOOM - 10) if floor else (W - 420, 22),
                         (120, 120, 255) if sp == "test" else (255, 255, 255))
        if writer is None:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas.shape[1], canvas.shape[0]))
        writer.write(canvas)
        if (frame_no - lo) % 300 == 0:
            print(f"{frame_no - lo}/{hi - lo + 1}", flush=True)
    writer.release()
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
