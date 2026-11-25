"""
Track people (owner / stranger), the robot with its heading, and the ball from live cameras or one or two recordings.

    uv run python tracking/track.py --basler SERIAL VIEW
    uv run python tracking/track.py videos/a1.mp4 a1 videos/b1.mp4 b1

Use --video FILE [--position VIEW] per recording and --offset N to align a pair.
REALTIME below chooses between dropping frames to keep the video's pace and processing every frame.
Keys: q or Esc quit, p pause, c continue. Positions are also written to results/<start time>_positions.csv.
"""
import argparse
import csv
import os
import signal
import time
from collections import deque
from contextlib import ExitStack
from threading import Event

import cv2
import numpy as np

from common import (KPT_CONF, LEFT_ANKLE, OBJECT_MODEL_PATH, RESULTS_DIR, RIGHT_ANKLE, SMOOTH_ALPHA, TRACK_TIMEOUT_S, PeopleDetector,
                    camera_positions, lens_correction, load_floor_map, load_room, put_label, require_trained_weight, towards_camera)
from faces import FaceEncoder, face_visible, load_owner
from fusion import Fusion, compatible, fuse_objects
from orientation import HEADING_HOLD_S, blend, body_heading, degrees, floor_frame
from robot_heading import RobotHeading, floor_heading
from room_map import RoomMap
from live_calibration import load_live_profiles, validate_pairs
from sources import BaslerSource, LiveFrames, VideoSource, list_baslers, pylon_devices

# ---- Settings ----
REALTIME = True                # True: drop frames to keep the video's pace, as a live camera would; False: every frame, slower than real time
OBJECTS = ("robot", "ball")    # classes of the object detector that we track
OBJECT_CONF = 0.5              # minimum confidence for a robot / ball box
OBJECT_IMGSZ = 1920            # detector input size; 2560 finds the far-away ball better but is slower
ROBOT_RADIUS_M = 0.20          # the bottom of the robot's box is its edge nearest the camera; its centre is this much further away
WALL_SLACK_M = 0.2             # a position this far beyond a wall still counts; further out is a detection on the wall
DISPLAY_SCALE = 0.4
FACE_EVERY_S = 0.5             # how often a person whose face is turned to a camera is compared with the owner (thresholds in fusion.py)
ARROW_M = 0.4                  # length of the heading arrow drawn on the floor in the camera views

# Drawing
LEFT_KPTS = {1, 3, 5, 7, 9, 11, 13, 15}
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)]
LABEL_COLOUR = {"unknown": (200, 200, 200), "owner": (255, 0, 0), "stranger": (0, 255, 255), "robot": (0, 200, 255), "ball": (0, 255, 0)}
LEFT_COLOUR, RIGHT_COLOUR, BONE_COLOUR, DOT_COLOUR = (0, 0, 255), (0, 255, 0), (220, 220, 220), (255, 255, 255)


OUTLINE_COLOUR = (40, 40, 40)


# ---- Floor positions ----
def to_metres(H, point_px, room):
    """Pixel point on the floor -> (x, y) metres, or None if it lands beyond the walls of the room (a detection on a wall)."""
    m = cv2.perspectiveTransform(np.asarray(point_px, np.float32).reshape(1, 1, 2), H)[0, 0]
    (x0, x1), (y0, y1) = room
    s = WALL_SLACK_M
    return m if x0 - s <= m[0] <= x1 + s and y0 - s <= m[1] <= y1 + s else None


def person_floor_point(kxy, kconf):
    """Where the person touches the floor: between the ankles, or at the one ankle that is visible. None if neither is."""
    left, right = kconf[LEFT_ANKLE] >= KPT_CONF, kconf[RIGHT_ANKLE] >= KPT_CONF   # missing keypoints come back as (0, 0) with low confidence
    if left and right:
        return (kxy[LEFT_ANKLE] + kxy[RIGHT_ANKLE]) / 2
    if left or right:
        return kxy[LEFT_ANKLE] if left else kxy[RIGHT_ANKLE]
    return None


def box_floor_point(xyxy):
    """Where a boxed object touches the floor: bottom centre of the box, which is the object's edge nearest the camera."""
    x1, y1, x2, y2 = xyxy
    return np.array([(x1 + x2) / 2, y2], np.float32)


class Track:
    """A tracked person or object: smoothed floor position and heading, and when they were last updated."""

    def __init__(self):
        self.floor_m = None
        self.placed = -1e9           # time of the last position update
        self.heading = None          # smoothed unit vector on the floor the person or robot faces
        self.heading_w = 0.0         # weight of the last heading measurement
        self.heading_at = -1e9       # time of the last heading measurement

    def fresh(self, t):
        """True while the position is at most TRACK_TIMEOUT_S old."""
        return t - self.placed <= TRACK_TIMEOUT_S

    def update_position(self, floor_m, t):
        """Moves the smoothed position towards the new measurement; a position that is no longer fresh is replaced by it."""
        self.floor_m = SMOOTH_ALPHA * floor_m + (1 - SMOOTH_ALPHA) * self.floor_m if self.fresh(t) else floor_m
        self.placed = t

    def update_heading(self, measured, t):
        """Folds one heading measurement (unit vector, weight) or None into the smoothed heading; drops a stale one."""
        if measured is not None:
            self.heading, self.heading_w = blend(self.heading, measured[0]), measured[1]
            self.heading_at = t
        elif t - self.heading_at > HEADING_HOLD_S:
            self.heading = None


class Person(Track):
    """A person tracked by one camera: a Track plus when it was last detected and the running face similarity to the owner."""

    def __init__(self):
        super().__init__()
        self.last_seen = -1e9        # time of the last detection, placed or not
        self.sim = None
        self.last_check = -1e9

    def compare(self, score):
        """Folds one face score into the running similarity. A score that contradicts it restarts it: the track now follows someone else."""
        self.sim = (score + self.sim) / 2 if self.sim is not None and compatible(score, self.sim) else score


class Camera:
    """One camera: its video, floor map, detector, the people it tracks and the last frame."""

    def __init__(self, name, room, size, H, intrinsics_dir=None):
        self.name = name
        self.width, self.height = size
        self.H = H
        self.floor_frame = floor_frame(H, (self.width / 2, self.height - 1)) if H is not None else None
        self.room = room
        self.detector = PeopleDetector()
        self.correct = (lens_correction(*size) if intrinsics_dir is None
                        else lens_correction(*size, intrinsics_dir=intrinsics_dir))
        self.people = {}
        self.frame = None


# ---- Per-frame steps ----
def update_people(camera, encoder, t):
    """Detect people in the camera's frame, update its tracks, compare faces with the owner. Returns what to draw: (id, box, kxy, kconf, floor point if placed)."""
    ids, boxes, kxy, kconf = camera.detector.detect(camera.frame)
    seen = []
    for i, tid in enumerate(ids):
        person = camera.people.setdefault(tid, Person())
        person.last_seen = t
        pt = person_floor_point(kxy[i], kconf[i])
        m = to_metres(camera.H, pt, camera.room) if pt is not None and camera.H is not None else None
        if m is not None:                # feet hidden or beyond the walls: this camera does not place the person; the other one may
            person.update_position(m, t)
        if camera.H is not None:
            person.update_heading(body_heading(kxy[i], kconf[i], boxes[i], camera.H, camera.floor_frame, pt), t)
        if camera.H is not None and encoder is not None and t - person.last_check >= FACE_EVERY_S and face_visible(kxy[i], kconf[i]):
            person.last_check = t        # counts as a try even if no face is found in the crop
            score = encoder.score(camera.frame, kxy[i])
            if score is not None:
                person.compare(score)
        seen.append((tid, boxes[i], kxy[i], kconf[i], pt if m is not None else None))
    return seen


def detect_objects(camera, model, heading_model):
    """Robot and ball in the camera's frame, best box per class, and the robot's heading.
    Returns {name: (box, conf, floor point, metres, heading)}; heading is (front px, rear px, unit vector on the floor,
    confidence) for the robot when the heading model finds it, else None. Without a floor map, metres and the unit
    vector are None."""
    r = model(camera.frame, imgsz=OBJECT_IMGSZ, conf=OBJECT_CONF, agnostic_nms=True, verbose=False)[0]
    best = {}
    for xyxy, conf, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
        name = model.names[cls]
        if name in OBJECTS and conf > best.get(name, (None, 0))[1]:
            best[name] = (xyxy, float(conf))
    found = {}
    for name, (xyxy, conf) in best.items():
        pt = box_floor_point(xyxy)
        m = None
        if camera.H is not None:
            m = to_metres(camera.H, pt, camera.room)
            if m is None:                # beyond the walls: a detection on the wall
                continue
        heading = None
        if name == "robot":
            if m is not None:
                m = m - ROBOT_RADIUS_M * towards_camera(camera.H, pt)   # from the near edge to the centre
            measured = heading_model.measure(camera.frame, xyxy)
            if measured is not None:
                front, rear, h_conf = measured
                u = floor_heading(camera.H, front, rear) if camera.H is not None else None
                heading = (front, rear, u, h_conf)
        found[name] = (xyxy, conf, pt, m, heading)
    return found


# ---- Position log ----
def open_log():
    """Starts a new CSV in results/, named by the start time. Returns the file and its writer."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for index in range(1000):
        suffix = f"_{index}" if index else ""
        path = os.path.join(RESULTS_DIR, f"{stamp}{suffix}_positions.csv")
        try:
            log = open(path, "x", newline="")
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError("Too many position logs with the same start time.")
    writer = csv.writer(log)
    writer.writerow(["frame", "t", "kind", "id", "label", "x", "y", "cameras", "sim", "heading_deg"])
    print(f"Positions are logged to {path}")
    return log, writer


def log_positions(writer, frame_index, t, people, objects, found):
    """One CSV row per person and object placed in this frame."""
    for p in people:
        if p.placed == t:
            writer.writerow([frame_index, f"{t:.3f}", "person", f"P{p.id}", p.label, f"{p.floor_m[0]:.3f}", f"{p.floor_m[1]:.3f}",
                             "+".join(p.placed_by), "" if p.sim is None else f"{p.sim:.2f}",
                             "" if p.heading is None else f"{degrees(p.heading):.1f}"])
    for name, obj in objects.items():
        if obj.placed == t:
            writer.writerow([frame_index, f"{t:.3f}", name, name, name, f"{obj.floor_m[0]:.3f}", f"{obj.floor_m[1]:.3f}",
                             "+".join(sorted(cam for cam in found if name in found[cam] and found[cam][name][3] is not None)), "",
                             "" if obj.heading is None else f"{degrees(obj.heading):.1f}"])


# ---- Drawing ----
def draw_person(canvas, H, fused, tid, box, kxy, kconf, pt, person):
    """Box in the label's colour, joints and bones and, if this camera placed the person, the floor point, heading and metre position."""
    x1, y1, x2, y2 = box.astype(int)
    label = fused.label if fused is not None else "unknown"
    who = f"P{fused.id} {label}" if fused is not None else "unknown"
    cv2.rectangle(canvas, (x1, y1), (x2, y2), LABEL_COLOUR[label], 2)
    put_label(canvas, who + (f"  {person.sim:.2f}" if person.sim is not None else "") + f"  track {tid}", (x1, y1 - 6))
    visible = kconf >= KPT_CONF
    for k, (px, py) in enumerate(kxy):
        if visible[k]:
            cv2.circle(canvas, (int(px), int(py)), 3, LEFT_COLOUR if k in LEFT_KPTS else RIGHT_COLOUR, -1)
    for a, b in SKELETON:
        if visible[a] and visible[b]:
            cv2.line(canvas, tuple(kxy[a].astype(int)), tuple(kxy[b].astype(int)), BONE_COLOUR, 2)
    if pt is not None:
        cv2.circle(canvas, tuple(pt.astype(int)), 6, DOT_COLOUR, -1)
        if person.heading is not None:         # the arrow lies on the floor: tail at the floor point, tip ARROW_M further along the heading
            tail = cv2.perspectiveTransform(pt.reshape(1, 1, 2).astype(np.float32), H)[0, 0]
            tip = cv2.perspectiveTransform((tail + ARROW_M * person.heading).reshape(1, 1, 2).astype(np.float32), np.linalg.inv(H))[0, 0]
            cv2.arrowedLine(canvas, tuple(pt.astype(int)), tuple(tip.astype(int)), LABEL_COLOUR[label], 3, cv2.LINE_AA, tipLength=0.25)
        facing = f"  facing {degrees(fused.heading):.0f} deg" if fused is not None and fused.heading is not None else ""
        put_label(canvas, f"x {person.floor_m[0]:.2f} m  y {person.floor_m[1]:.2f} m{facing}", (int(pt[0]) + 10, int(pt[1]) - 10))


def draw_object(canvas, name, box, conf, pt, m, heading, smoothed):
    """Box, floor point and metre position of the robot or the ball; for the robot also the heading this camera
    measured (arrow rear -> front on the robot) and the smoothed heading of the cameras on the floor in degrees."""
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), LABEL_COLOUR[name], 2)
    cv2.circle(canvas, tuple(pt.astype(int)), 6, DOT_COLOUR, -1)
    if heading is not None:
        front, rear = heading[0], heading[1]
        tip = front + 0.6 * (front - rear)
        for colour, thick in ((OUTLINE_COLOUR, 6), (LABEL_COLOUR[name], 3)):
            cv2.arrowedLine(canvas, tuple(rear.astype(int)), tuple(tip.astype(int)), colour, thick, cv2.LINE_AA, tipLength=0.3)
    facing = f"  facing {degrees(smoothed):.0f} deg" if smoothed is not None else ""
    where = f"  x {m[0]:.2f} m  y {m[1]:.2f} m" if m is not None else ""
    put_label(canvas, f"{name} {conf:.2f}{where}{facing}", (x1, y1 - 6))


def camera_view(cam, seen, links, found, objects, t, fps):
    """The camera's frame with its people and objects drawn on it, scaled for the screen."""
    canvas = cam.frame.copy()
    for tid, box, kxy, kconf, pt in seen:
        draw_person(canvas, cam.H, links.get((cam.name, tid)), tid, box, kxy, kconf, pt, cam.people[tid])
    for name, (box, conf, pt, m, heading) in found.items():
        draw_object(canvas, name, box, conf, pt, m, heading, objects[name].heading)
    put_label(canvas, f"{cam.name} | people {len(seen)} | {t:.1f} s | {fps:.1f} fps", (14, 36), bg=(230, 230, 230))
    return cv2.resize(canvas, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)


def place_windows(cameras, with_room):
    """Tile the camera windows in two rows and place the room map beside them."""
    width = max(cam.width for cam in cameras.values()) * DISPLAY_SCALE
    height = max(cam.height for cam in cameras.values()) * DISPLAY_SCALE
    for index, name in enumerate(cameras):
        cv2.namedWindow(name)
        cv2.moveWindow(name, int(index // 2 * (width + 10)), int(index % 2 * (height + 40)))
    if with_room:
        cv2.namedWindow("Room")
        cv2.moveWindow("Room", int((len(cameras) + 1) // 2 * (width + 10)), 0)


def room_marks(people, objects, t):
    """What the room map shows, as RoomMap.draw takes it: the people, and the robot and ball while their position is fresh."""
    marks = [(p.floor_m, LABEL_COLOUR[p.label], f"P{p.id} {p.label}", p.trail, p.heading) for p in people if p.floor_m is not None and t - p.placed <= TRACK_TIMEOUT_S]
    marks += [(obj.floor_m, LABEL_COLOUR[name], name, [], obj.heading) for name, obj in objects.items() if obj.fresh(t)]
    return marks


# ---- Main loop ----
def object_detector():
    from ultralytics import YOLO

    return YOLO(require_trained_weight(OBJECT_MODEL_PATH))


def process_frames(cameras, frames, fusion, objects, model, heading_model, encoder, t):
    seen, found = {}, {}
    for name, cam in cameras.items():
        for tid in [tid for tid, person in cam.people.items() if t - person.last_seen > TRACK_TIMEOUT_S]:
            del cam.people[tid]
        if name in frames:
            cam.frame = cam.correct(frames[name])
            seen[name] = update_people(cam, encoder, t)
            found[name] = detect_objects(cam, model, heading_model)
    links = fusion.update({name: cam.people for name, cam in cameras.items()}, t)
    fuse_objects(objects, found, t)
    return seen, links, found


def run_tracking(stack, cameras, sources, room_m, grid_m, headless, window_s=None, shutdown=None):
    live = window_s is not None
    fusion, objects = Fusion(), {name: Track() for name in OBJECTS}
    model = object_detector()
    heading_model = RobotHeading()
    signature, owner_video = load_owner()
    encoder = FaceEncoder(signature) if signature is not None and room_m is not None else None
    print(f"Owner enrolled from: {owner_video}" if encoder else "Owner labels inactive.")
    log, writer = open_log()
    stack.enter_context(log)
    if not headless:
        stack.callback(cv2.destroyAllWindows)
        room = RoomMap(room_m, grid_m) if room_m is not None else None
        place_windows(cameras, room is not None)

    if live:
        capture = LiveFrames(sources, window_s)
        for source in sources.values():
            source.start()
        latency_ms = deque(maxlen=10_000)
        skew_ms = deque(maxlen=10_000)
    else:
        primary = next(iter(sources.values()))
    paused, after = False, float("-inf")
    count = 0
    started = capture.started_at if live else time.monotonic()
    shown_at = {}
    try:
        while shutdown is None or not shutdown.is_set():
            key = (-1 if headless else cv2.waitKey(20 if paused else 1) & 0xFF)
            if key in (27, ord("q")):
                break
            if key == ord("p") or paused and key == ord("c"):
                paused = not paused
                if not paused:
                    if live:
                        after = time.monotonic()
                    else:
                        primary.resync()
            if paused:
                if live:
                    capture.check_liveness(time.monotonic())
                continue
            if live:
                batch = capture.next(after)
                if not batch:
                    if not headless:
                        now = time.monotonic() - capture.started_at
                        show_waiting_views(cameras, shown_at, now)
                        cv2.imshow("Room", room.draw(room_marks(fusion.people.values(), objects, now)))
                    time.sleep(0.01)
                    continue
                frame_index = count
                newest = capture.last_update
                t = newest - capture.started_at
                frames = {name: frame.image for name, frame in batch.items()}
            else:
                frame_index = primary.due_index() if REALTIME else primary.index() + 1
                frames = {name: source.read_at(frame_index + source.shift) for name, source in sources.items()}
                if any(frame is None for frame in frames.values()):
                    break
                t = primary.time()

            seen, links, found = process_frames(cameras, frames, fusion, objects, model, heading_model, encoder, t)
            log_positions(writer, frame_index, t, fusion.people.values(), objects, found)
            count += 1
            if live:
                log.flush()
                oldest = min(frame.received_at for frame in batch.values())
                latency_ms.append((time.monotonic() - oldest) * 1000)
                if len(batch) > 1:
                    skew_ms.append((newest - oldest) * 1000)
            if not headless:
                fps = count / max(time.monotonic() - started, 0.001)
                for name in frames:
                    view = camera_view(cameras[name], seen[name], links, found[name], objects, t, fps)
                    cv2.imshow(name, view)
                    shown_at[name] = t
                if live:
                    show_waiting_views(cameras, shown_at, t)
                if room is not None:
                    cv2.imshow("Room", room.draw(room_marks(fusion.people.values(), objects, t)))
    finally:
        if live:
            for source in sources.values():
                source.stop.set()
            print_live_stats(sources, capture.discarded, count, time.monotonic() - started, latency_ms, skew_ms)


def show_waiting_views(cameras, shown_at, t):
    for name, cam in cameras.items():
        stamp = shown_at.get(name, float("-inf"))
        if stamp is not None and t - stamp > TRACK_TIMEOUT_S:
            canvas = np.zeros((int(cam.height * DISPLAY_SCALE), int(cam.width * DISPLAY_SCALE), 3), np.uint8)
            put_label(canvas, f"{name} | waiting for a fresh frame", (14, 36))
            cv2.imshow(name, canvas)
            shown_at[name] = None


def print_live_stats(sources, discarded, count, elapsed, latency_ms, skew_ms):
    print(f"Live: {count} updates in {elapsed:.1f} s ({count / max(elapsed, 0.001):.2f} updates/s incl. pauses)")
    for name, source in sources.items():
        with source.lock:
            print(f"{name} ({source.serial}): {source.received} received, {source.skipped} skipped by pylon, "
                  f"{source.overwritten} overwritten, {discarded[name]} discarded by timing"
                  + (f", capture stopped: {source.error}" if source.error else ""))
    for label, samples in (("Receipt-to-CSV latency", latency_ms), ("Multi-view arrival skew", skew_ms)):
        if samples:
            median, p95 = np.percentile(samples, (50, 95))
            print(f"{label}: median {median:.1f} ms, p95 {p95:.1f} ms (last {len(samples)} updates with samples)")


def main(inputs, *, headless=False, shutdown=None):
    """Recorded inputs are (path, window name, optional floor map, frame shift)."""
    maps = [view for _, _, view, _ in inputs if view is not None]
    room_m, grid_m = load_room(maps) if maps else (None, None)
    with ExitStack() as stack:
        cameras, sources = {}, {}
        for video, name, view, shift in inputs:
            source = VideoSource(video, shift)
            stack.callback(source.close)
            sources[name] = source
            H = load_floor_map(view) if view is not None else None
            cameras[name] = Camera(name, room_m, (source.width, source.height), H)
        start = max(0, -min(source.shift for source in sources.values()))
        if start:
            next(iter(sources.values())).read_at(start - 1)
        run_tracking(stack, cameras, sources, room_m, grid_m, headless, shutdown=shutdown)


def run_live(pairs, headless, window_s, shutdown=None):
    devices = pylon_devices()
    with ExitStack() as stack:
        sources = {}
        for serial, name in pairs:
            source = BaslerSource(serial, devices)
            stack.callback(source.close)
            sources[name] = source
        room_m, grid_m, profiles = load_live_profiles(sources)
        cameras = {}
        for name, source in sources.items():
            H, directory = profiles[name]
            size = source.geometry["Width"], source.geometry["Height"]
            cameras[name] = Camera(name, room_m, size, H, directory)
        run_tracking(stack, cameras, sources, room_m, grid_m, headless, window_s, shutdown)


class VideoArg(argparse.Action):
    """--video: a new [path, position] pair; the position is filled in by a --position after it."""
    def __call__(self, parser, namespace, value, option_string=None):
        videos = getattr(namespace, self.dest) or []
        setattr(namespace, self.dest, videos + [[value, None]])


class PositionArg(argparse.Action):
    """--position: the camera position of the --video given just before it."""
    def __call__(self, parser, namespace, value, option_string=None):
        videos = getattr(namespace, "video") or []
        if not videos or videos[-1][1] is not None:
            parser.error("give --position right after its --video")
        videos[-1][1] = value


def cli(argv=None, *, shutdown=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("recordings", nargs="*", metavar="INPUT", help="recorded input as VIDEO_A VIEW_A VIDEO_B VIEW_B")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--video", action=VideoArg, metavar="FILE", help="one or two recordings, each optionally followed by --position")
    ap.add_argument("--position", action=PositionArg, choices=camera_positions(), help="floor map for the preceding --video")
    ap.add_argument("--offset", type=int, help="skip N frames of the second recording (negative skips frames of the first)")
    mode.add_argument("--basler", nargs=2, action="append", metavar=("SERIAL", "VIEW"), help="select a live camera, repeat for each view")
    mode.add_argument("--list-cameras", action="store_true", help="list Basler serials, models and transports")
    ap.add_argument("--sync-ms", type=float, default=100, help="live frame freshness and arrival skew limit in milliseconds (default 100)")
    ap.add_argument("--headless", action="store_true", help="write positions without opening windows")
    a = ap.parse_args(argv)
    try:
        if a.recordings and (a.basler or a.list_cameras or a.video):
            ap.error("choose recorded input or a live camera command")
        if a.offset is not None and not (a.video and len(a.video) == 2):
            ap.error("--offset requires two --video inputs")
        if a.list_cameras:
            devices = list_baslers()
            for serial, model, transport in devices:
                print(f"{serial}\t{model}\t{transport}")
            if not devices:
                print("No Basler cameras found.")
        elif a.basler:
            validate_pairs(a.basler)
            if not np.isfinite(a.sync_ms) or a.sync_ms <= 0:
                ap.error("--sync-ms must be a positive finite number")
            run_live(a.basler, a.headless, a.sync_ms / 1000, shutdown)
        else:
            videos = a.video
            if not videos:
                if len(a.recordings) != 4:
                    ap.error("provide --video FILE, VIDEO_A VIEW_A VIDEO_B VIEW_B, or --basler SERIAL VIEW")
                videos = [a.recordings[:2], a.recordings[2:]]
                if any(view not in camera_positions() for _, view in videos):
                    ap.error(f"choose recorded views from {', '.join(camera_positions())}")
            if len(videos) > 2:
                ap.error("at most two --video inputs")
            inputs = []
            for index, (path, view) in enumerate(videos):
                if not os.path.isfile(path):
                    ap.error(f"video not found: {path}")
                name = view or f"video {index + 1}"
                if any(previous[1] == name for previous in inputs):
                    ap.error("choose distinct camera positions")
                inputs.append((path, name, view, (a.offset or 0) if index else 0))
            main(inputs, headless=a.headless, shutdown=shutdown)
    except (OSError, RuntimeError, ValueError) as exc:
        ap.exit(1, f"{exc}\n")
    except KeyboardInterrupt:
        ap.exit(130)


def entrypoint():
    shutdown = Event()
    previous = signal.signal(signal.SIGINT, lambda *_: shutdown.set())
    try:
        cli(shutdown=shutdown)
    finally:
        signal.signal(signal.SIGINT, previous)
    if shutdown.is_set():
        raise SystemExit(130)


if __name__ == "__main__":
    entrypoint()
