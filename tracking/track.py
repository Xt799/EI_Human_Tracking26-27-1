"""
Track people (owner / stranger), robot and ball from two cameras and show where they are in the room.

    uv run python tracking/track.py videos/a1.mp4 a1 videos/b1.mp4 b1

The two videos must be simultaneous recordings of the same scene, starting on the same frame and with the same frame count.
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

from common import (KPT_CONF, LEFT_ANKLE, OBJECT_MODEL_PATH, RESULTS_DIR, RIGHT_ANKLE, SMOOTH_HALF_S, TRACK_TIMEOUT_S, PeopleDetector,
                    camera_pose, camera_positions, lens_correction, load_floor_map, load_room, put_label, towards_camera)
from faces import FaceEncoder, face_visible, load_owner
from fusion import OWNER_SIM, STRANGER_SIM, Fusion, compatible, fuse_objects
from orientation import HEADING_HOLD_S, blend, body_heading, degrees, floor_frame
from room_map import RoomMap
from sources import VideoSource

# ---- Settings ----
REALTIME = True                # True: drop frames to keep the video's pace, as a live camera would; False: every frame, slower than real time
OBJECTS = ("robot", "ball")    # classes of the object detector that we track
OBJECT_CONF = 0.5              # minimum confidence for a robot / ball box
OBJECT_IMGSZ = 1920            # detector input size; 2560 finds the far-away ball better but is slower
ROBOT_RADIUS_M = 0.20          # the bottom of the robot's box is its edge nearest the camera; its centre is this much further away
WALL_SLACK_M = 0.1            # a position this far beyond a wall still counts; further out is a detection on the wall
ANKLE_H_M = 0.10               # height of the ankle joint above the floor (best fit on the pair 5 markers)
DISPLAY_SCALE = 0.4
FACE_EVERY_S = 0.5             # how often a person whose face is turned to a camera is compared with the owner (thresholds in fusion.py)
ARROW_M = 0.4                  # length of the heading arrow drawn on the floor in the camera views

# Drawing
LEFT_KPTS = {1, 3, 5, 7, 9, 11, 13, 15}
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)]
LABEL_COLOUR = {"unknown": (200, 200, 200), "owner": (255, 0, 0), "stranger": (0, 255, 255), "robot": (0, 200, 255), "ball": (0, 255, 0)}
LEFT_COLOUR, RIGHT_COLOUR, BONE_COLOUR, DOT_COLOUR = (0, 0, 255), (0, 255, 0), (220, 220, 220), (255, 255, 255)


# ---- Floor positions ----
def to_metres(H, point_px, room, above_m=0.0, camera=None):
    """Pixel point -> (x, y) metres on the floor, or None if it lands beyond the walls of the room (a detection on a wall).
    A point above_m above the floor maps too far from the camera; it is pulled back towards the camera ((x, y) below it, height)."""
    m = cv2.perspectiveTransform(np.asarray(point_px, np.float32).reshape(1, 1, 2), H)[0, 0]
    if above_m:
        below, height = camera
        m = below + (m - below) * (height - above_m) / height
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
    """A tracked object: smoothed floor position, the last measurement and when it was last placed."""

    def __init__(self):
        self.floor_m = None
        self.raw_m = None            # the last measurement, unsmoothed
        self.placed = -1e9           # time of the last position update

    def fresh(self, t):
        """True while the position is at most TRACK_TIMEOUT_S old."""
        return t - self.placed <= TRACK_TIMEOUT_S

    def update_position(self, floor_m, t):
        """Moves the smoothed position towards the new measurement, the more the longer since the last one; a position that is no longer fresh is replaced by it."""
        if self.fresh(t):
            keep = 0.5 ** ((t - self.placed) / SMOOTH_HALF_S)
            self.floor_m = keep * self.floor_m + (1 - keep) * floor_m
        else:
            self.floor_m = floor_m
        self.raw_m = floor_m
        self.placed = t


class Person(Track):
    """A person tracked by one camera: a Track plus when it was last detected and the running face similarity to the owner."""

    def __init__(self):
        super().__init__()
        self.last_seen = -1e9        # time of the last detection, placed or not
        self.range_m = None          # distance on the floor from its camera at the last placement
        self.sim = None
        self.looked = -1e9           # time of the last face score
        self.last_check = -1e9
        self.heading = None          # smoothed unit vector on the floor the body faces (orientation.py)
        self.heading_w = 0.0         # weight of the last heading measurement
        self.heading_at = -1e9       # time of the last heading measurement

    def update_heading(self, measured, t):
        """Folds one heading measurement (unit vector, weight) or None into the smoothed heading; drops a stale one."""
        if measured is not None:
            self.heading, self.heading_w = blend(self.heading, measured[0]), measured[1]
            self.heading_at = t
        elif t - self.heading_at > HEADING_HOLD_S:
            self.heading = None

    def compare(self, score, t):
        """Folds one face score into the running similarity. A score that contradicts it restarts it: the track now follows someone else."""
        self.sim = (score + self.sim) / 2 if self.sim is not None and compatible(score, self.sim) else score
        self.looked = t


class Camera:
    """One camera: its video, floor map, detector, the people it tracks and the last frame."""

    def __init__(self, video, name, room):
        self.name = name
        self.source = VideoSource(video)
        self.H = load_floor_map(name)
        self.pose = camera_pose(self.H, self.source.width, self.source.height)   # ((x, y) on the floor below the camera, height)
        self.floor_frame = floor_frame(self.H, (self.source.width / 2, self.source.height - 1))   # (handedness, w sign) of the floor map (orientation.py)
        self.room = room             # room bounds in metres, shared by both cameras
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
        m = to_metres(camera.H, pt, camera.room, ANKLE_H_M, camera.pose) if pt is not None else None
        if m is not None:                # feet hidden or beyond the walls: this camera does not place the person; the other one may
            person.update_position(m, t)
            person.range_m = float(np.linalg.norm(m - camera.pose[0]))
        person.update_heading(body_heading(kxy[i], kconf[i], boxes[i], camera.H, camera.floor_frame, pt), t)
        if encoder is not None and t - person.last_check >= FACE_EVERY_S and face_visible(kxy[i], kconf[i]):
            person.last_check = t        # counts as a try even if no face is found in the crop
            score = encoder.score(camera.frame, kxy[i])
            if score is not None:
                person.compare(score, t)
        seen.append((tid, boxes[i], kxy[i], kconf[i], pt if m is not None else None))
    for tid in [k for k, p in camera.people.items() if t - p.last_seen > TRACK_TIMEOUT_S]:
        del camera.people[tid]
    return seen


def detect_objects(camera, model):
    """Robot and ball in the camera's frame, best box per class. Returns {name: (box, conf, floor point, metres)}."""
    r = model(camera.frame, imgsz=OBJECT_IMGSZ, conf=OBJECT_CONF, agnostic_nms=True, verbose=False)[0]
    best = {}
    for xyxy, conf, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
        name = model.names[cls]
        if name in OBJECTS and conf > best.get(name, (None, 0))[1]:
            best[name] = (xyxy, float(conf))
    found = {}
    for name, (xyxy, conf) in best.items():
        pt = box_floor_point(xyxy)
        m = to_metres(camera.H, pt, camera.room)
        if m is not None:
            if name == "robot":
                m = m - ROBOT_RADIUS_M * towards_camera(camera.H, pt)   # from the near edge to the centre
            found[name] = (xyxy, conf, pt, m)
    return found


# ---- Position log ----
def open_log():
    """Starts a new CSV in results/, named by the start time. Returns the file and its writer."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, time.strftime("%Y%m%d_%H%M%S") + "_positions.csv")
    log = open(path, "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["frame", "t", "kind", "id", "label", "x", "y", "raw_x", "raw_y", "cameras", "sim", "heading_deg"])
    print(f"Positions are logged to {path}")
    return log, writer


def log_positions(writer, frame_index, t, people, objects, found):
    """One CSV row per person whose position is known (cameras "held" and no raw position when none placed them in this frame) and per object placed in this frame."""
    for p in people:
        if p.located(t):
            placed = p.placed == t
            writer.writerow([frame_index, f"{t:.3f}", "person", f"P{p.id}", p.label, f"{p.floor_m[0]:.3f}", f"{p.floor_m[1]:.3f}",
                             f"{p.raw_m[0]:.3f}" if placed else "", f"{p.raw_m[1]:.3f}" if placed else "",
                             "+".join(p.placed_by) if placed else "held", "" if p.sim is None else f"{p.sim:.2f}",
                             "" if p.heading is None else f"{degrees(p.heading):.1f}"])
    for name, obj in objects.items():
        if obj.placed == t:
            writer.writerow([frame_index, f"{t:.3f}", name, name, name, f"{obj.floor_m[0]:.3f}", f"{obj.floor_m[1]:.3f}",
                             f"{obj.raw_m[0]:.3f}", f"{obj.raw_m[1]:.3f}",
                             "+".join(sorted(cam for cam in found if name in found[cam])), "", ""])


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


def draw_object(canvas, name, box, conf, pt, m):
    """Box, floor point and metre position of the robot or the ball."""
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), LABEL_COLOUR[name], 2)
    cv2.circle(canvas, tuple(pt.astype(int)), 6, DOT_COLOUR, -1)
    put_label(canvas, f"{name} {conf:.2f}  x {m[0]:.2f} m  y {m[1]:.2f} m", (x1, y1 - 6))


def camera_view(cam, seen, links, found, t, fps):
    """The camera's frame with its people and objects drawn on it, scaled for the screen."""
    canvas = cam.frame.copy()
    for tid, box, kxy, kconf, pt in seen:
        draw_person(canvas, cam.H, links.get((cam.name, tid)), tid, box, kxy, kconf, pt, cam.people[tid])
    for name, (box, conf, pt, m) in found.items():
        draw_object(canvas, name, box, conf, pt, m)
    put_label(canvas, f"{cam.name} | people {len(seen)} | {t:.1f} s | {fps:.1f} fps", (14, 36), bg=(230, 230, 230))
    return cv2.resize(canvas, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)


def place_windows(cameras):
    """Opens the three windows: the two camera views on the left, one above the other, the room map to their right."""
    w, h = int(cameras[0].source.width * DISPLAY_SCALE), int(cameras[0].source.height * DISPLAY_SCALE)
    for name, x, y in ((cameras[0].name, 0, 0), (cameras[1].name, 0, h + 40), ("Room", w + 10, 0)):
        cv2.namedWindow(name)
        cv2.moveWindow(name, x, y)


def room_marks(people, objects, t):
    """What the room map shows, as RoomMap.draw takes it: the people whose position is known, and the robot and ball while their position is fresh."""
    marks = [(p.floor_m, LABEL_COLOUR[p.label], f"P{p.id} {p.label}", p.trail, p.heading) for p in people if p.located(t)]
    marks += [(obj.floor_m, LABEL_COLOUR[name], name, [], None) for name, obj in objects.items() if obj.fresh(t)]
    return marks


# ---- Main loop ----
def main(video_a, camera_a, video_b, camera_b):
    room_m, grid_m = load_room((camera_a, camera_b))
    cameras = [Camera(video_a, camera_a, room_m), Camera(video_b, camera_b, room_m)]
    object_model = YOLO(OBJECT_MODEL_PATH)
    signature, owner_video = load_owner()
    encoder = FaceEncoder(signature) if signature is not None else None
    print(f"Owner enrolled from: {owner_video}" if encoder else "No owner enrolled: everyone is 'unknown'.")
    print(f"Inputs: {video_a} ({camera_a}), {video_b} ({camera_b})")

    room = RoomMap(room_m, grid_m)
    fusion, objects = Fusion(), {name: Track() for name in OBJECTS}
    log, writer = open_log()
    place_windows(cameras)

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
            if not cameras[0].read(frame_index) or not cameras[1].read(frame_index):
                break
            t = cameras[0].source.time()

            seen = {cam.name: update_people(cam, encoder, t) for cam in cameras}
            links = fusion.update({cam.name: cam.people for cam in cameras}, t)
            found = {cam.name: detect_objects(cam, object_model) for cam in cameras}
            fuse_objects(objects, found, t)
            log_positions(writer, frame_index, t, fusion.people.values(), objects, found)

            n_frames += 1
            if time.time() - fps_t0 >= 1.0:
                fps, n_frames, fps_t0 = n_frames / (time.time() - fps_t0), 0, time.time()
            for cam in cameras:
                cv2.imshow(cam.name, camera_view(cam, seen[cam.name], links, found[cam.name], t, fps))
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
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("video_a", help="first video file")
    ap.add_argument("camera_a", choices=camera_positions(), help="camera position of the first video (selects its floor map)")
    ap.add_argument("video_b", help="second video file, recorded at the same time")
    ap.add_argument("camera_b", choices=camera_positions(), help="camera position of the second video")
    a = ap.parse_args()
    main(a.video_a, a.camera_a, a.video_b, a.camera_b)
