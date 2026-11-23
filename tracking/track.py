"""
Track people (owner / stranger), the robot with its heading and the ball from one or two cameras, and show where they
are in the room.

    uv run python tracking/track.py --video VIDEO_1 --position a4 --video VIDEO_2 --position b4 [--offset 23]
    uv run python tracking/track.py --video VIDEO [--position a4]

--position: the calibrated camera position the video was recorded from (its floor map in calibration/), right after
its --video. It turns pixels into metres; without it the camera is shown in the image only (people, the robot with
its heading arrow, the ball; no metres, no room map, no owner label). Two cameras are merged only if both have one.
--offset N: the two videos start together, after skipping the first N frames of the second video (a negative N skips
the first video's frames instead), so that they show the same moment.
REALTIME below chooses between dropping frames to keep the video's pace and processing every frame.
Keys: q or Esc quit, p pause, c continue. Positions are also written to results/<start time>_positions.csv.
"""
import argparse
import csv
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

from common import (KPT_CONF, LEFT_ANKLE, OBJECT_MODEL_PATH, RESULTS_DIR, RIGHT_ANKLE, SMOOTH_ALPHA, TRACK_TIMEOUT_S, PeopleDetector,
                    camera_positions, lens_correction, load_floor_map, load_room, put_label, towards_camera)
from faces import FaceEncoder, face_visible, load_owner
from fusion import Fusion, compatible, fuse_objects
from orientation import HEADING_HOLD_S, blend, body_heading, degrees, floor_frame
from robot_heading import RobotHeading, floor_heading
from room_map import RoomMap
from sources import VideoSource

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
LEFT_COLOUR, RIGHT_COLOUR, BONE_COLOUR, DOT_COLOUR, OUTLINE_COLOUR = (0, 0, 255), (0, 255, 0), (220, 220, 220), (255, 255, 255), (40, 40, 40)


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
    """One camera: its video, floor map (None: no metres), detector, the people it tracks and the last frame."""

    def __init__(self, video, name, floor_map, room, shift=0):
        self.name = name
        self.source = VideoSource(video)
        self.H = load_floor_map(floor_map) if floor_map else None
        self.floor_frame = None if self.H is None else floor_frame(self.H, (self.source.width / 2, self.source.height - 1))   # (handedness, w sign) of the floor map (orientation.py)
        self.room = room             # room bounds in metres, shared by both cameras
        self.shift = shift           # this video's frame = the first video's frame + shift
        self.detector = PeopleDetector()
        self.correct = lens_correction(self.source.width, self.source.height)
        self.people = {}
        self.frame = None

    def read(self, index):
        """Reads the frame with this index. False at the end of the video."""
        raw = self.source.read_at(index)
        if raw is None:
            return False
        self.frame = self.correct(raw)
        return True


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
        if encoder is not None and t - person.last_check >= FACE_EVERY_S and face_visible(kxy[i], kconf[i]):
            person.last_check = t        # counts as a try even if no face is found in the crop
            score = encoder.score(camera.frame, kxy[i])
            if score is not None:
                person.compare(score)
        seen.append((tid, boxes[i], kxy[i], kconf[i], pt if m is not None else None))
    for tid in [k for k, p in camera.people.items() if t - p.last_seen > TRACK_TIMEOUT_S]:
        del camera.people[tid]
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
    path = os.path.join(RESULTS_DIR, time.strftime("%Y%m%d_%H%M%S") + "_positions.csv")
    log = open(path, "w", newline="")
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
                             "+".join(sorted(cam for cam in found if name in found[cam])), "",
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
    """Opens the windows: the camera views on the left, one above the other, the room map to their right."""
    w, h = int(cameras[0].source.width * DISPLAY_SCALE), int(cameras[0].source.height * DISPLAY_SCALE)
    windows = [(cam.name, 0, i * (h + 40)) for i, cam in enumerate(cameras)] + ([("Room", w + 10, 0)] if with_room else [])
    for name, x, y in windows:
        cv2.namedWindow(name)
        cv2.moveWindow(name, x, y)


def room_marks(people, objects, t):
    """What the room map shows, as RoomMap.draw takes it: the people, and the robot and ball while their position is fresh."""
    marks = [(p.floor_m, LABEL_COLOUR[p.label], f"P{p.id} {p.label}", p.trail, p.heading) for p in people if p.floor_m is not None]
    marks += [(obj.floor_m, LABEL_COLOUR[name], name, [], obj.heading) for name, obj in objects.items() if obj.fresh(t)]
    return marks


# ---- Command line ----
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


# ---- Main loop ----
def main(inputs):
    """inputs: [(video, window name, floor map or None, shift)], one or two cameras."""
    maps = [m for _, _, m, _ in inputs if m]
    room_m, grid_m = load_room(maps) if maps else (None, None)
    cameras = [Camera(v, name, m, room_m, shift) for v, name, m, shift in inputs]
    start = max(0, -min(cam.shift for cam in cameras))
    if start:                        # a later video starts earlier on the timeline: skip the first video's frames before it
        cameras[0].source.read_at(start - 1)
    object_model = YOLO(OBJECT_MODEL_PATH)
    heading_model = RobotHeading()
    signature, owner_video = load_owner()
    encoder = FaceEncoder(signature) if signature is not None else None
    print(f"Owner enrolled from: {owner_video}" if encoder else "No owner enrolled: everyone is 'unknown'.")
    for v, name, m, shift in inputs:
        print(f"Input: {v} ({'floor map ' + m if m else 'no floor map: image only'}{f', {shift:+d} frames' if shift else ''})")

    room = RoomMap(room_m, grid_m) if room_m is not None else None
    fusion, objects = Fusion(), {name: Track() for name in OBJECTS}
    log, writer = open_log()
    place_windows(cameras, room is not None)

    fps, n_frames, fps_t0, paused = 0.0, 0, time.time(), False
    try:
        while True:
            if paused:
                key = cv2.waitKey(50) & 0xFF
                if key in (ord("c"), ord("p")):
                    paused = False
                    cameras[0].source.resync()
                elif key in (27, ord("q")):
                    break
                continue
            frame_index = cameras[0].source.due_index() if REALTIME else cameras[0].source.index() + 1
            if not all(cam.read(frame_index + cam.shift) for cam in cameras):
                break
            t = cameras[0].source.time()

            seen = {cam.name: update_people(cam, encoder, t) for cam in cameras}
            links = fusion.update({cam.name: cam.people for cam in cameras}, t)
            found = {cam.name: detect_objects(cam, object_model, heading_model) for cam in cameras}
            fuse_objects(objects, found, t)
            log_positions(writer, frame_index, t, fusion.people.values(), objects, found)

            n_frames += 1
            if time.time() - fps_t0 >= 1.0:
                fps, n_frames, fps_t0 = n_frames / (time.time() - fps_t0), 0, time.time()
            for cam in cameras:
                cv2.imshow(cam.name, camera_view(cam, seen[cam.name], links, found[cam.name], objects, t, fps))
            if room is not None:
                cv2.imshow("Room", room.draw(room_marks(fusion.people.values(), objects, t)))

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            paused = key == ord("p")
    except KeyboardInterrupt:
        pass
    finally:
        log.close()
        for cam in cameras:
            cam.source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", action=VideoArg, required=True, help="a video file; give it twice for two cameras")
    ap.add_argument("--position", action=PositionArg, choices=camera_positions(),
                    help="camera position the video before it was recorded from (its floor map)")
    ap.add_argument("--offset", type=int, default=0,
                    help="frames to skip at the start of the second video (negative: of the first) so both show the same moment")
    a = ap.parse_args()
    if len(a.video) > 2:
        ap.error("at most two --video")
    for v, _ in a.video:
        if not os.path.exists(v):
            ap.error(f"video not found: {v}")
    shifts = [0, a.offset] if len(a.video) == 2 else [0]
    main([(v, pos or f"video {i + 1}", pos, shift) for i, ((v, pos), shift) in enumerate(zip(a.video, shifts))])
