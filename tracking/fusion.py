"""Fuse person tracks, robot and ball positions, and robot heading across camera views."""
from collections import deque

import numpy as np

from common import SMOOTH_ALPHA, TRACK_TIMEOUT_S
from orientation import HEADING_HOLD_S, combine

ATTACH_M = 0.6       # a new track joins a person this close that has no live track from its camera
SECOND_BOX_M = 1.2   # a new track this close to a person its camera already tracks is a second box on that person
SPLIT_M = 1.5        # a person's two tracks this far apart for SPLIT_S seconds are two people
SPLIT_S = 0.5
TRAIL_N = 45         # positions kept for drawing the trail

# Owner rule, on the running face similarity (same person about 0.6, others about 0.0 on our recordings)
OWNER_SIM = 0.45               # at or above this: owner candidate
STRANGER_SIM = 0.25            # below this: stranger
OWNER_SWITCH_MARGIN = 0.10     # another person must beat the current owner by this much to take over


def compatible(sim_a, sim_b):
    """False when one face score says owner and the other says stranger: not the same person, whatever the distance."""
    def looks(s):
        return None if s is None or STRANGER_SIM <= s < OWNER_SIM else s >= OWNER_SIM
    a, b = looks(sim_a), looks(sim_b)
    return a is None or b is None or a == b


def distance(a, b):
    return float(np.linalg.norm(a - b))


def fuse_objects(objects, found, t):
    """Robot and ball positions: the average of the cameras that see them in this frame; the robot's heading: the
    cameras' headings weighted by the heading model's confidence, then smoothed over time.
    found: {camera: {name: (box, conf, pixel, metres, heading or None)}}, heading = (front, rear, unit vector, conf)."""
    for name, obj in objects.items():
        seen = [found[cam][name] for cam in found if name in found[cam]]
        points = [s[3] for s in seen if s[3] is not None]          # None: a camera without a floor map
        if points:
            obj.update_position(np.mean(points, axis=0), t)
        heading = combine([(s[4][2], s[4][3]) for s in seen if s[4] is not None and s[4][2] is not None])
        obj.update_heading((heading, 1.0) if heading is not None else None, t)


class FusedPerson:
    """One person in the room: which track in each camera is them, their position and their owner score."""

    def __init__(self, pid):
        self.id = pid
        self.members = {}            # camera name -> track id
        self.floor_m = None
        self.label = "unknown"
        self.sim = None
        self.placed = -1e9           # when the person was last placed on the floor
        self.placed_by = []          # the cameras that placed them then
        self.offset = {}             # camera -> its point minus the two-camera average, learned while both cameras see the person
        self.apart_since = None
        self.trail = deque(maxlen=TRAIL_N)
        self.heading = None          # unit vector on the floor the body faces: the cameras' headings, weighted (orientation.py)
        self.heading_at = -1e9


class Fusion:
    """The people in the room, kept from frame to frame and rebuilt from the cameras' tracks, and the owner among them."""

    def __init__(self):
        self.people = {}             # id -> FusedPerson
        self.next_id = 1             # ids are never reused
        self.cameras = {}            # the cameras' live tracks in the frame being processed: {camera: {track id: Person}}
        self.t = 0.0                 # time of that frame

    def update(self, cameras, t):
        """Updates the people from the cameras' live tracks ({camera: {track id: Person}}). Returns {(camera, track id): FusedPerson}."""
        self.cameras, self.t = cameras, t
        self.drop_ended_tracks()
        self.split_people()
        self.attach_free_tracks()
        self.create_people()
        self.place_people()
        self.merge_people()
        self.forget_people()
        self.decide_labels()
        return {(cam, tid): p for p in self.people.values() for cam, tid in p.members.items()}

    # ---- The steps of update, in order ----
    def drop_ended_tracks(self):
        """A track that ended, or whose face no longer matches its person's, leaves the person."""
        for p in self.people.values():
            p.members = {cam: tid for cam, tid in p.members.items()
                         if tid in self.cameras[cam] and compatible(self.cameras[cam][tid].sim, p.sim)}

    def split_people(self):
        """Detach a member that persistently disagrees with the other fresh views."""
        for p in self.people.values():
            points = {cam: self.cameras[cam][tid].floor_m for cam, tid in p.members.items()
                      if self.cameras[cam][tid].placed == self.t}
            outlier = self.outlier(points)
            if outlier is None:
                if p.apart_since is not None:
                    suspect, since = p.apart_since
                    if (suspect not in p.members or self.t - since > TRACK_TIMEOUT_S
                            or suspect in points and len(points) > 1):
                        p.apart_since = None
            elif p.apart_since is None or p.apart_since[0] != outlier:
                p.apart_since = outlier, self.t
            elif self.t - p.apart_since[1] > SPLIT_S:
                del p.members[outlier]
                p.offset.pop(outlier, None)
                p.apart_since = None

    def attach_free_tracks(self):
        """A track that belongs to nobody joins the nearest person who has no live track from its camera."""
        pairs = [(distance(self.cameras[cam][tid].floor_m, p.floor_m), cam, tid, p)
                 for cam, tid in self.free_tracks() for p in self.people.values() if p.floor_m is not None and self.open_for(p, cam)]
        attached = set()
        for d, cam, tid, p in sorted(pairs, key=lambda pair: pair[0]):
            if d <= ATTACH_M and compatible(self.cameras[cam][tid].sim, p.sim) and (cam, tid) not in attached and self.open_for(p, cam):
                p.members[cam] = tid
                attached.add((cam, tid))

    def create_people(self):
        """A track still free becomes a new person, unless it is a second box on someone."""
        for cam, tid in self.free_tracks():
            if not self.second_box(cam, self.cameras[cam][tid]):
                p = FusedPerson(self.next_id)
                self.next_id += 1
                p.members[cam] = tid
                self.people[p.id] = p

    def place_people(self):
        """Every person's position, heading and face score, from their tracks in this frame."""
        for p in self.people.values():
            self.place_person(p)

    def place_person(self, p):
        tracks = {cam: self.cameras[cam][tid] for cam, tid in p.members.items()}
        placed = {cam: tr for cam, tr in tracks.items() if tr.placed == self.t}
        if placed:
            self.place(p, placed)
        measured = [(tr.heading, tr.heading_w) for tr in tracks.values() if tr.heading is not None and tr.heading_at == self.t]
        heading = combine(measured)
        if heading is not None:
            p.heading, p.heading_at = heading, self.t
        elif self.t - p.heading_at > HEADING_HOLD_S:
            p.heading = None
        sims = [tr.sim for tr in tracks.values() if tr.sim is not None]
        if sims:
            p.sim = max(sims)

    def merge_people(self):
        """Two people with tracks from different cameras standing on the same spot are one person: the older one keeps the id."""
        for q in sorted(self.people.values(), key=lambda p: p.id, reverse=True):     # newest first
            for p in sorted(self.people.values(), key=lambda p: p.id):               # against everyone older, oldest first
                if p.id >= q.id:
                    break
                if self.same_spot(p, q):
                    p.members.update(q.members)
                    p.trail.pop()  # replace this update's position with the combined measurement
                    self.place_person(p)
                    p.sim = max((s for s in (p.sim, q.sim) if s is not None), default=None)
                    p.label = "owner" if "owner" in (p.label, q.label) else p.label
                    del self.people[q.id]
                    break

    def forget_people(self):
        """A person not placed for TRACK_TIMEOUT_S is forgotten."""
        for pid in [pid for pid, p in self.people.items() if self.t - p.placed > TRACK_TIMEOUT_S]:
            del self.people[pid]

    def decide_labels(self):
        """One owner at most: the best match above OWNER_SIM. The owner keeps the label while tracked unless clearly beaten."""
        judged = [p for p in self.people.values() if p.sim is not None]
        if not judged:
            return
        current = next((p for p in judged if p.label == "owner"), None)
        best = max(judged, key=lambda p: p.sim)
        takes_over = current is None or best is current or best.sim > current.sim + OWNER_SWITCH_MARGIN
        owner = best if best.sim >= OWNER_SIM and takes_over else current
        for p in judged:
            if p is owner:
                p.label = "owner"
            elif p.sim < STRANGER_SIM or (owner is not None and p.sim < OWNER_SIM):
                p.label = "stranger"
            else:
                p.label = "unknown"       # looks like the owner but is not the chosen one: do not call them a stranger

    # ---- Helpers of the steps ----
    def seen(self, cam, tid):
        """True if this camera detected the track in this frame."""
        return tid in self.cameras[cam] and self.cameras[cam][tid].last_seen == self.t

    def open_for(self, p, cam):
        """True if the person has no live track from this camera, so that one may join."""
        return cam not in p.members or not self.seen(cam, p.members[cam])

    def free_tracks(self):
        """The tracks placed in this frame that belong to nobody: [(camera, track id)]."""
        taken = {(cam, tid) for p in self.people.values() for cam, tid in p.members.items()}
        return [(cam, tid) for cam, tracks in self.cameras.items() for tid, track in tracks.items()
                if (cam, tid) not in taken and track.placed == self.t]

    def second_box(self, cam, track):
        """True for a free track close to a person this camera already tracks, with no face score against it: a second box on that person."""
        return any(p.floor_m is not None and not self.open_for(p, cam) and compatible(track.sim, p.sim)
                   and distance(track.floor_m, p.floor_m) <= SECOND_BOX_M for p in self.people.values())

    def outlier(self, points):
        """Return the fresh member with the most disagreements. Ties favor earlier members."""
        disagreements = {cam: sum(distance(point, other) > SPLIT_M for other in points.values())
                         for cam, point in points.items()}
        if any(disagreements.values()):
            return max(reversed(disagreements), key=disagreements.get)
        return None

    def place(self, p, placed):
        """Position of a person from their tracks placed in this frame ({camera: track})."""
        # each camera alone is off by a roughly constant offset (the ankle joint is above the floor), so the offset is
        # learned while both cameras see the person and subtracted when only one does
        if len(placed) > 1:
            p.floor_m = np.mean([tr.floor_m for tr in placed.values()], axis=0)
            for cam, tr in placed.items():
                offset = tr.floor_m - p.floor_m
                p.offset[cam] = offset if cam not in p.offset else SMOOTH_ALPHA * offset + (1 - SMOOTH_ALPHA) * p.offset[cam]
        else:
            cam, tr = next(iter(placed.items()))
            p.floor_m = tr.floor_m - p.offset.get(cam, 0.0)
        p.trail.append(p.floor_m)
        p.placed, p.placed_by = self.t, sorted(placed)

    def same_spot(self, p, q):
        """True for two people seen by different cameras, both placed in this frame within ATTACH_M, whose faces do not contradict."""
        return (not set(p.members) & set(q.members) and p.placed == q.placed == self.t
                and distance(p.floor_m, q.floor_m) <= ATTACH_M and compatible(p.sim, q.sim))
