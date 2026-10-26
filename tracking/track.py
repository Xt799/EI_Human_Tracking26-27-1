"""
track.py  --  people, robot and ball: detection, tracking and floor position.

People detection, lens correction and the drawing helpers are based on the
previous group's cap_detection/"Untitled-3_Orientation copy.py" (main branch).
Robot and ball use the YOLO26n detector trained in training/.

Run from the repo root:
    .venv\\Scripts\\python.exe tracking\\track.py

Per frame:
  1. read a frame from VIDEO_PATH (or the first Basler camera if unset)
  2. lens correction with calibration/camera_matrix.npy
  3. YOLO pose in tracking mode: every person gets a persistent track ID
  4. YOLO26n detector: robot and ball boxes
  5. floor point of each person from the lowest visible keypoints
     (ankles, else knees, else hips); of each object the bottom centre of its box;
     all converted to metres with the floor map of this camera position
     (calibration/floor_map_<view>.npy) and smoothed
  6. draw boxes, skeletons, IDs and metre positions

Owner / stranger: not decided yet, every person is "unknown" (TODO(STT) 5).
Robot heading: not computed yet (TODO(STT) 13).
Keys: q or Esc quit, p pause, c continue.
"""

# VM-friendly GUI settings; only matter on a Linux virtual machine, harmless elsewhere.
import os
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["QT_X11_NO_MITSHM"] = "1"
os.environ["QT_OPENGL"] = "software"

import re
import time

import cv2
import numpy as np
from ultralytics import YOLO
try:
    from pypylon import pylon          # Basler camera driver, lab only
except ImportError:
    pylon = None

# ========================= CONFIG =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input: a recording from videos/ at the repo root (not in git, see README), None = first Basler camera found.
VIDEO_PATH = os.path.join(SCRIPT_DIR, "..", "videos", "basler_record_20261012_171200.mp4")

# Camera position of the input, which selects the floor map. None = derive it from
# the recording id in the file name (see RECORDING_VIEWS); set it explicitly for a live camera.
VIEW = None
RECORDING_VIEWS = {"165859": "steep", "171200": "steep", "171939": "steep", "180049": "low"}

POSE_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "yolo11s-pose.pt")
OBJECT_MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "training", "runs", "yolo26n_e30", "weights", "best.pt")
CALIB_DIR = os.path.join(SCRIPT_DIR, "calibration")

CAM_WIDTH, CAM_HEIGHT = 1900, 1200   # requested from a Basler camera; video input uses the file's size
DISPLAY_SCALE = 0.60
POSE_CONF_THRES = 0.25               # minimum confidence for a person box
KPT_CONF_THRES = 0.5                 # a keypoint below this is treated as not visible
OBJECT_CONF_THRES = 0.5              # minimum confidence for a robot / ball box
OBJECT_IMGSZ = 1920                  # detector input size; 2560 finds the far-away ball better but is slower
TRACKER = "bytetrack.yaml"           # Ultralytics built-in tracker config
SMOOTH_ALPHA = 0.3                   # exponential smoothing of the metre position (1 = no smoothing)
TRACK_TIMEOUT_S = 1.0                # forget a track not seen for this long
REALTIME = True                      # video input: keep the video clock by skipping frames when processing is slower
# Floor points outside this range (metres from the origin) are discarded: a detection on a wall or a
# shelf maps to a point far outside the room. Loose box around the floor seen by both cameras.
FLOOR_BOUNDS_M = ((-5.0, 4.0), (-3.0, 5.5))
SHOW_FPS = True
WIN = "People tracking"

# COCO keypoint indices
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16
LEFT_KPTS = {1, 3, 5, 7, 9, 11, 13, 15}
RIGHT_KPTS = {2, 4, 6, 8, 10, 12, 14, 16}
SKELETON_PAIRS = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16),
                  (5, 6), (11, 12), (5, 11), (6, 12)]

# Colours (BGR)
COL_LEFT = (0, 0, 255)
COL_RIGHT = (0, 255, 0)
COL_MID = (220, 220, 220)
COL_FLOOR_DOT = (255, 255, 255)
COL_BOX = {"unknown": (200, 200, 200), "owner": (255, 0, 0), "stranger": (0, 255, 255),
           "robot": (0, 200, 255), "ball": (0, 255, 0)}
TEXT_BG = (255, 255, 255)
TEXT_COL = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ========================= HELPERS =========================
def put_label(img, text, org, bg_color=TEXT_BG, txt_color=TEXT_COL, scale=0.6, thickness=2):
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x, y - th - baseline - 4), (x + tw + 6, y + 4), bg_color, -1)
    cv2.putText(img, text, (x + 3, y - 4), FONT, scale, txt_color, thickness)


def to_metres(H, pt_px):
    """One pixel point on the floor -> (x, y) metres through the floor map, or None if it lands outside the room."""
    m = cv2.perspectiveTransform(np.asarray(pt_px, np.float32).reshape(1, 1, 2), H)[0, 0]
    (x0, x1), (y0, y1) = FLOOR_BOUNDS_M
    return m if x0 <= m[0] <= x1 and y0 <= m[1] <= y1 else None


def floor_point_px(kxy, kconf):
    """
    Pixel point where the person touches the floor, from the lowest body part that is visible.
    Returns (point, source) with source in {"ankles", "ankle", "knees", "hips"}, or (None, None).
    Missing keypoints come back from the model as (0, 0) with low confidence, so visibility is
    judged on confidence, never on the coordinates.
    """
    vis = kconf >= KPT_CONF_THRES
    for pair, name in (((LEFT_ANKLE, RIGHT_ANKLE), "ankles"), ((LEFT_KNEE, RIGHT_KNEE), "knees"), ((LEFT_HIP, RIGHT_HIP), "hips")):
        a, b = pair
        if vis[a] and vis[b]:
            return (kxy[a] + kxy[b]) * 0.5, name
        if name == "ankles" and (vis[a] or vis[b]):
            return kxy[a] if vis[a] else kxy[b], "ankle"
    return None, None


def view_for(video_path):
    """Floor map name for a recording, from the recording id in its file name."""
    if VIEW:
        return VIEW
    for rec_id, view in RECORDING_VIEWS.items():
        if video_path and rec_id in os.path.basename(video_path):
            return view
    raise SystemExit("Cannot tell the camera position from the file name; set VIEW to 'steep' or 'low'.")


def box_floor_point(xyxy):
    """Pixel point where a boxed object touches the floor: bottom centre of the box.
    For the robot this is its near edge, so the position is biased ~15 cm towards the camera."""
    x1, y1, x2, y2 = xyxy
    return np.array([(x1 + x2) * 0.5, y2], np.float32)


class Track:
    """What we remember about one tracked person or object between frames."""

    def __init__(self, track_id, label="unknown"):
        self.id = track_id
        self.label = label          # person: "unknown" | "owner" | "stranger" (TODO(STT) 5); object: "robot" | "ball"
        self.floor_m = None         # smoothed (x, y) in metres
        self.source = None          # which keypoints gave the last floor point
        self.last_seen = 0.0        # video time in seconds

    def update(self, floor_m, source, t):
        if self.floor_m is None or t - self.last_seen > TRACK_TIMEOUT_S:
            self.floor_m = floor_m
        else:
            self.floor_m = SMOOTH_ALPHA * floor_m + (1 - SMOOTH_ALPHA) * self.floor_m
        self.source = source
        self.last_seen = t


# ========================= MAIN =========================
def main():
    pose_model = YOLO(POSE_MODEL_PATH)
    object_model = YOLO(OBJECT_MODEL_PATH)       # classes: robot, ball
    # TODO(STT) 5: load the owner's appearance signature here (owner.npz written by
    # enrol_owner.py, embedding from a person re-identification network). Every
    # "unknown" track is compared against it every few frames; three consecutive
    # matches make it "owner", a clear non-match makes it "stranger".

    # ---- Frame source: video file (development) or Basler camera (lab) ----
    if VIDEO_PATH:
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise SystemExit(f"Could not open video: {VIDEO_PATH}")
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        print(f"Video: {VIDEO_PATH}")
        clock = {"start": None}      # wall time at which video time 0 would have been, for real-time pacing

        def read_frame():
            if REALTIME:
                pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if clock["start"] is None:
                    clock["start"] = time.time() - pos / fps_in
                target = (time.time() - clock["start"]) * fps_in
                while pos < target - 1:          # behind the video clock: drop frames to catch up
                    if not cap.grab():
                        return None
                    pos += 1
            ok, f = cap.read()
            return f if ok else None

        def video_time():
            return cap.get(cv2.CAP_PROP_POS_FRAMES) / fps_in

        def resync():                            # after a pause, restart the clock from the current frame
            clock["start"] = None

        def close_source():
            cap.release()
    else:
        if pylon is None:
            raise SystemExit("VIDEO_PATH is not set and pypylon is not installed.")
        factory = pylon.TlFactory.GetInstance()
        devices = factory.EnumerateDevices()
        if not devices:
            raise SystemExit("No Basler camera found.")
        print("Cameras:", [d.GetFriendlyName() for d in devices])
        camera = pylon.InstantCamera(factory.CreateDevice(devices[0]))
        camera.Open()
        try:
            camera.Width.SetValue(CAM_WIDTH)
            camera.Height.SetValue(CAM_HEIGHT)
        except Exception as e:
            print(f"Resolution set warning: {e}")
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        t0 = time.time()

        def read_frame():
            grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            try:
                return converter.Convert(grab).GetArray() if grab.GrabSucceeded() else None
            finally:
                grab.Release()

        def video_time():
            return time.time() - t0

        def resync():
            pass

        def close_source():
            camera.StopGrabbing()
            camera.Close()

    frame0 = read_frame()
    if frame0 is None:
        close_source()
        raise SystemExit("Could not read the first frame.")
    frame_h, frame_w = frame0.shape[:2]

    # ---- Lens correction maps, built for the real frame size ----
    K = np.load(os.path.join(CALIB_DIR, "camera_matrix.npy"))
    D = np.load(os.path.join(CALIB_DIR, "distortion_coefficients.npy"))
    newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (frame_w, frame_h), 0, (frame_w, frame_h))
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, newK, (frame_w, frame_h), cv2.CV_16SC2)

    # ---- Floor map of this camera position (made once by calibrate.py) ----
    view = view_for(VIDEO_PATH)
    H = np.load(os.path.join(CALIB_DIR, f"floor_map_{view}.npy"))
    print(f"Floor map: {view}")

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WIN, int(frame_w * DISPLAY_SCALE), int(frame_h * DISPLAY_SCALE))
    print("Running. q/Esc quit, p pause, c continue.")

    tracks = {}                                  # people, by track id
    objects = {"robot": Track("robot", "robot"), "ball": Track("ball", "ball")}
    paused = False
    prev_t, fps, frames = time.time(), 0.0, 0
    frame = frame0
    resync()

    try:
        while True:
            if paused:
                k = cv2.waitKey(50) & 0xFF
                if k in (ord("c"), ord("p")):
                    paused = False
                    resync()
                elif k in (27, ord("q")):
                    break
                continue

            if frame is None:
                frame = read_frame()
                if frame is None:
                    if VIDEO_PATH:
                        print("End of video.")
                        break
                    continue
            frame_u = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
            frame = None
            t = video_time()

            # ---- Pose detection with tracking: one persistent ID per person ----
            r = pose_model.track(frame_u, conf=POSE_CONF_THRES, persist=True, tracker=TRACKER, verbose=False)[0]
            n = len(r.boxes)
            if n and r.boxes.id is not None:
                ids = r.boxes.id.cpu().numpy().astype(int)
                boxes = r.boxes.xyxy.cpu().numpy()
                kxy = r.keypoints.xy.cpu().numpy()
                kconf = r.keypoints.conf.cpu().numpy()
            else:
                ids, boxes = np.empty(0, int), np.empty((0, 4))
                kxy, kconf = np.empty((0, 17, 2)), np.empty((0, 17))

            # ---- Floor position per track ----
            canvas = frame_u.copy()
            for i, tid in enumerate(ids):
                tr = tracks.setdefault(tid, Track(tid))
                pt, source = floor_point_px(kxy[i], kconf[i])
                m = to_metres(H, pt) if pt is not None else None
                if m is not None:
                    tr.update(m, source, t)
                else:
                    tr.last_seen = t          # seen, but no usable floor point: keep the last position

                # ---- Draw ----
                x1, y1, x2, y2 = boxes[i].astype(int)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), COL_BOX[tr.label], 2)
                put_label(canvas, f"id {tid}  {tr.label}", (x1, y1 - 6))
                vis = kconf[i] >= KPT_CONF_THRES
                for k, (px, py) in enumerate(kxy[i]):
                    if vis[k]:
                        cv2.circle(canvas, (int(px), int(py)), 3, COL_LEFT if k in LEFT_KPTS else COL_RIGHT if k in RIGHT_KPTS else COL_MID, -1)
                for a, b in SKELETON_PAIRS:
                    if vis[a] and vis[b]:
                        lc = COL_LEFT if (a in LEFT_KPTS and b in LEFT_KPTS) else COL_RIGHT if (a in RIGHT_KPTS and b in RIGHT_KPTS) else COL_MID
                        cv2.line(canvas, tuple(kxy[i][a].astype(int)), tuple(kxy[i][b].astype(int)), lc, 2)
                if pt is not None:
                    cv2.circle(canvas, tuple(pt.astype(int)), 6, COL_FLOOR_DOT, -1)
                if tr.floor_m is not None:
                    px_lab = tuple(pt.astype(int)) if pt is not None else (x1, y2)
                    put_label(canvas, f"x {tr.floor_m[0]:.2f} m  y {tr.floor_m[1]:.2f} m  ({tr.source})", (px_lab[0] + 10, px_lab[1] - 10))
                # TODO(STT) 9: this is where (tid, tr.label, tr.floor_m) is published over ROS / logged to CSV.

            # ---- Forget tracks not seen for a while ----
            for tid in [k for k, tr in tracks.items() if t - tr.last_seen > TRACK_TIMEOUT_S]:
                del tracks[tid]

            # ---- Robot and ball: best box per class, floor point at the bottom centre ----
            ro = object_model(frame_u, imgsz=OBJECT_IMGSZ, conf=OBJECT_CONF_THRES, agnostic_nms=True, verbose=False)[0]
            best = {}
            for xyxy, conf, cls in zip(ro.boxes.xyxy.cpu().numpy(), ro.boxes.conf.cpu().numpy(), ro.boxes.cls.cpu().numpy().astype(int)):
                name = object_model.names[cls]
                if name in objects and (name not in best or conf > best[name][1]):
                    best[name] = (xyxy, float(conf))
            for name, (xyxy, conf) in best.items():
                obj = objects[name]
                pt = box_floor_point(xyxy)
                m = to_metres(H, pt)
                if m is None:
                    continue                  # not on the floor (e.g. something on a wall): ignore
                obj.update(m, "box", t)
                x1, y1, x2, y2 = xyxy.astype(int)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), COL_BOX[name], 2)
                cv2.circle(canvas, tuple(pt.astype(int)), 6, COL_FLOOR_DOT, -1)
                put_label(canvas, f"{name} {conf:.2f}  x {obj.floor_m[0]:.2f} m  y {obj.floor_m[1]:.2f} m", (x1, y1 - 6))
                # TODO(STT) 13: robot heading, e.g. from the direction of motion of its smoothed position.
            # Objects out of view keep their last position until TRACK_TIMEOUT_S, then are reported as not visible.
            for obj in objects.values():
                if obj.floor_m is not None and t - obj.last_seen > TRACK_TIMEOUT_S:
                    obj.floor_m = None

            # ---- HUD ----
            frames += 1
            now = time.time()
            if now - prev_t >= 1.0:
                fps, frames, prev_t = frames / (now - prev_t), 0, now
            seen = " ".join(k for k, o in objects.items() if o.floor_m is not None) or "-"
            put_label(canvas, f"view {view} | people {n} | tracks {len(tracks)} | objects {seen}" + (f" | {fps:.1f} fps" if SHOW_FPS else ""), (14, 36), bg_color=(230, 230, 230))

            cv2.imshow(WIN, cv2.resize(canvas, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("p"):
                paused = True
    except KeyboardInterrupt:
        pass
    finally:
        close_source()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
