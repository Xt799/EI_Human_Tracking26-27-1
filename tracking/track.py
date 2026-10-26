"""
Track people (owner / stranger), robot and ball and show their floor positions in metres.

    .venv\\Scripts\\python.exe tracking\\track.py videos\\clips\\tracking\\steep_a.mp4 steep
    .venv\\Scripts\\python.exe tracking\\track.py camera low        # live Basler camera

Keys: q or Esc quit, p pause, c continue.
"""
import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO

from appearance import AppearanceEncoder, load_owner, similarity
from common import OBJECT_MODEL_PATH, PeopleDetector, camera_positions, lens_correction, load_floor_map, put_label
from sources import open_source

# ---- Settings ----
KPT_CONF = 0.5                 # a keypoint below this confidence is treated as not visible
OBJECT_CONF = 0.5              # minimum confidence for a robot / ball box
OBJECT_IMGSZ = 1920            # detector input size; 2560 finds the far-away ball better but is slower
SMOOTH_ALPHA = 0.3             # weight of the newest position in the smoothed one (1 = no smoothing)
TRACK_TIMEOUT_S = 1.0          # a person or object not seen for this long is forgotten
FLOOR_BOUNDS_M = ((-5.0, 4.0), (-3.0, 5.5))   # positions outside the room are rejected (detections on walls)
DISPLAY_SCALE = 0.6

# Owner recognition (needs tracking/owner.npz from enrol_owner.py)
REID_EVERY_S = 0.5             # how often each person is compared against the owner signature
OWNER_SIM = 0.80               # running similarity at or above this: owner candidate
STRANGER_SIM = 0.70            # running similarity below this: stranger
OWNER_CHECKS = 3               # comparisons needed before a person gets a label
OWNER_SWITCH_MARGIN = 0.05     # another person must beat the current owner by this much to take over

# COCO keypoints
LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE = 11, 12, 13, 14, 15, 16
LEFT_KPTS = {1, 3, 5, 7, 9, 11, 13, 15}
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)]

# Colours (BGR)
LABEL_COLOUR = {"unknown": (200, 200, 200), "owner": (255, 0, 0), "stranger": (0, 255, 255), "robot": (0, 200, 255), "ball": (0, 255, 0)}
LEFT_COLOUR, RIGHT_COLOUR, BONE_COLOUR, DOT_COLOUR = (0, 0, 255), (0, 255, 0), (220, 220, 220), (255, 255, 255)


# ---- Floor positions ----
def to_metres(H, point_px):
    """Pixel point on the floor -> (x, y) metres, or None if it lands outside the room."""
    m = cv2.perspectiveTransform(np.asarray(point_px, np.float32).reshape(1, 1, 2), H)[0, 0]
    (x0, x1), (y0, y1) = FLOOR_BOUNDS_M
    return m if x0 <= m[0] <= x1 and y0 <= m[1] <= y1 else None


def person_floor_point(kxy, kconf):
    """Where the person touches the floor: between the ankles, else one ankle, else knees, else hips. Returns (point, source)."""
    visible = kconf >= KPT_CONF   # missing keypoints come back as (0, 0) with low confidence
    for (a, b), name in (((LEFT_ANKLE, RIGHT_ANKLE), "ankles"), ((LEFT_KNEE, RIGHT_KNEE), "knees"), ((LEFT_HIP, RIGHT_HIP), "hips")):
        if visible[a] and visible[b]:
            return (kxy[a] + kxy[b]) / 2, name
        if name == "ankles" and (visible[a] or visible[b]):
            return kxy[a] if visible[a] else kxy[b], "ankle"
    return None, None


def box_floor_point(xyxy):
    """Where a boxed object touches the floor: bottom centre of the box (for the robot its near edge, ~15 cm towards the camera)."""
    x1, y1, x2, y2 = xyxy
    return np.array([(x1 + x2) / 2, y2], np.float32)


class Track:
    """A tracked object: smoothed floor position and when it was last seen."""

    def __init__(self, label):
        self.label = label
        self.floor_m = None
        self.source = None           # which keypoints or box edge gave the floor point
        self.last_seen = 0.0

    def update_position(self, floor_m, source, t):
        self.floor_m = floor_m if self.floor_m is None else SMOOTH_ALPHA * floor_m + (1 - SMOOTH_ALPHA) * self.floor_m
        self.source = source
        self.last_seen = t


class Person(Track):
    """A tracked person: a Track plus the running similarity to the owner. Label: unknown, owner or stranger."""

    def __init__(self):
        super().__init__("unknown")
        self.sim = None
        self.checks = 0
        self.last_check = -1e9

    def compare(self, vec, signature, t):
        s = similarity(vec, signature)
        self.sim = s if self.sim is None else (s + self.sim) / 2
        self.checks += 1
        self.last_check = t


# ---- Owner / stranger ----
def decide_labels(people):
    """One owner at most: the best match above OWNER_SIM. An owner keeps the label while the track lives unless clearly beaten."""
    judged = [p for p in people if p.checks >= OWNER_CHECKS]
    if not judged:
        return
    current = next((p for p in judged if p.label == "owner"), None)
    best = max(judged, key=lambda p: p.sim)
    takes_over = current is None or best is current or best.sim > current.sim + OWNER_SWITCH_MARGIN
    owner = best if best.sim >= OWNER_SIM and takes_over else current
    for p in judged:
        if p is owner:
            p.label = "owner"
        elif owner is not None or p.sim < STRANGER_SIM:
            p.label = "stranger"
        else:
            p.label = "unknown"


# ---- Per-frame steps ----
def update_people(people, detector, encoder, signature, frame, H, t):
    """Detect people, update their tracks, compare with the owner. Returns what to draw: (id, box, kxy, kconf, floor point)."""
    ids, boxes, kxy, kconf = detector.detect(frame)
    seen = []
    for i, tid in enumerate(ids):
        person = people.setdefault(tid, Person())
        pt, source = person_floor_point(kxy[i], kconf[i])
        m = to_metres(H, pt) if pt is not None else None
        if m is not None:
            person.update_position(m, source, t)
        person.last_seen = t
        if encoder is not None and t - person.last_check >= REID_EVERY_S:
            person.compare(encoder.embed(frame, boxes[i]), signature, t)
        seen.append((tid, boxes[i], kxy[i], kconf[i], pt))
    for tid in [k for k, p in people.items() if t - p.last_seen > TRACK_TIMEOUT_S]:
        del people[tid]
    if encoder is not None:
        decide_labels(people.values())      # all live tracks, so a person missed for one frame keeps their label
    return seen


def update_objects(objects, model, frame, H, t):
    """Detect robot and ball (best box per class), update their tracks. Returns what to draw: (name, box, conf, floor point)."""
    r = model(frame, imgsz=OBJECT_IMGSZ, conf=OBJECT_CONF, agnostic_nms=True, verbose=False)[0]
    best = {}
    for xyxy, conf, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
        name = model.names[cls]
        if name in objects and conf > best.get(name, (None, 0))[1]:
            best[name] = (xyxy, float(conf))
    seen = []
    for name, (xyxy, conf) in best.items():
        pt = box_floor_point(xyxy)
        m = to_metres(H, pt)
        if m is not None:
            objects[name].update_position(m, "box", t)
            seen.append((name, xyxy, conf, pt))
    for obj in objects.values():
        if t - obj.last_seen > TRACK_TIMEOUT_S:
            obj.floor_m = None
    return seen


def draw_person(canvas, person, tid, box, kxy, kconf, pt):
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), LABEL_COLOUR[person.label], 2)
    put_label(canvas, f"id {tid}  {person.label}" + (f"  {person.sim:.2f}" if person.sim is not None else ""), (x1, y1 - 6))
    visible = kconf >= KPT_CONF
    for k, (px, py) in enumerate(kxy):
        if visible[k]:
            cv2.circle(canvas, (int(px), int(py)), 3, LEFT_COLOUR if k in LEFT_KPTS else RIGHT_COLOUR, -1)
    for a, b in SKELETON:
        if visible[a] and visible[b]:
            cv2.line(canvas, tuple(kxy[a].astype(int)), tuple(kxy[b].astype(int)), BONE_COLOUR, 2)
    if pt is not None:
        cv2.circle(canvas, tuple(pt.astype(int)), 6, DOT_COLOUR, -1)
    if person.floor_m is not None:
        at = tuple(pt.astype(int)) if pt is not None else (x1, y2)
        put_label(canvas, f"x {person.floor_m[0]:.2f} m  y {person.floor_m[1]:.2f} m  ({person.source})", (at[0] + 10, at[1] - 10))


def draw_object(canvas, obj, name, box, conf, pt):
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), LABEL_COLOUR[name], 2)
    cv2.circle(canvas, tuple(pt.astype(int)), 6, DOT_COLOUR, -1)
    put_label(canvas, f"{name} {conf:.2f}  x {obj.floor_m[0]:.2f} m  y {obj.floor_m[1]:.2f} m", (x1, y1 - 6))


# ---- Main loop ----
def main(video, camera):
    detector = PeopleDetector()
    object_model = YOLO(OBJECT_MODEL_PATH)
    signature, owner_videos = load_owner()
    encoder = AppearanceEncoder() if signature is not None else None
    print(f"Owner enrolled from: {', '.join(owner_videos)}" if encoder else "No owner enrolled: everyone is 'unknown'.")

    source = open_source(video)
    first = source.read()
    if first is None:
        raise SystemExit("Could not read the first frame.")
    correct = lens_correction(first.shape[1], first.shape[0])
    H = load_floor_map(camera)
    print(f"Input: {video}, floor map: {camera}")

    people, objects = {}, {"robot": Track("robot"), "ball": Track("ball")}
    window = "Tracking"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window, int(first.shape[1] * DISPLAY_SCALE), int(first.shape[0] * DISPLAY_SCALE))
    fps, n_frames, fps_t0 = 0.0, 0, time.time()
    frame, paused = first, False
    source.resync()
    try:
        while True:
            if paused:
                key = cv2.waitKey(50) & 0xFF
                if key in (ord("c"), ord("p")):
                    paused = False
                    source.resync()
                elif key in (27, ord("q")):
                    break
                continue
            frame = frame if frame is not None else source.read()
            if frame is None:
                break
            frame = correct(frame)
            t = source.time()

            seen_people = update_people(people, detector, encoder, signature, frame, H, t)
            seen_objects = update_objects(objects, object_model, frame, H, t)

            canvas = frame.copy()
            for tid, box, kxy, kconf, pt in seen_people:
                draw_person(canvas, people[tid], tid, box, kxy, kconf, pt)
            for name, box, conf, pt in seen_objects:
                draw_object(canvas, objects[name], name, box, conf, pt)
            n_frames += 1
            if time.time() - fps_t0 >= 1.0:
                fps, n_frames, fps_t0 = n_frames / (time.time() - fps_t0), 0, time.time()
            visible_objects = " ".join(n for n, o in objects.items() if o.floor_m is not None) or "-"
            put_label(canvas, f"{camera} | people {len(seen_people)} | tracks {len(people)} | objects {visible_objects} | {fps:.1f} fps", (14, 36), bg=(230, 230, 230))
            cv2.imshow(window, cv2.resize(canvas, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE))
            frame = None

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            paused = key == ord("p")
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("video", help="video file, or 'camera' for the live Basler camera")
    ap.add_argument("camera", choices=camera_positions(), help="camera position the video was recorded from (selects the floor map)")
    a = ap.parse_args()
    main(a.video, a.camera)
