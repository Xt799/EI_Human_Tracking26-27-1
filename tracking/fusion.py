"""Merge the two cameras' person tracks into one set of people in room coordinates and decide who the owner is; average the robot and the ball."""
from collections import deque

import numpy as np

from common import TRACK_TIMEOUT_S
from orientation import HEADING_HOLD_S, combine

ATTACH_M = 0.6       # a new track joins a person this close that has no track from its camera detected in this frame
SECOND_BOX_M = 0.4   # a new track this close to its camera's own track on a person is a second box on that person
SPLIT_M = 1.5        # a person's two tracks this far apart for SPLIT_S seconds are two people
SPLIT_S = 0.5
OFFSET_ALPHA = 0.3   # weight of the newest sample in a learned camera offset
HOLD_S = 2.0         # a person's last position is kept this long without seeing their feet; then they have no position until placed again
TRAIL_N = 45         # positions kept for drawing the trail

# Owner rule, on the running face similarity (same person about 0.6, others about 0.0 on our recordings)
OWNER_SIM = 0.45               # at or above this: owner candidate
STRANGER_SIM = 0.25            # below this a face score says stranger; a score in between does not block a join
OWNER_SWITCH_MARGIN = 0.10     # another person must beat the current owner by this much to take over
FRESH_S = 5.0                  # a face score this recent may block a join or win the owner label; an older one only keeps a label


def compatible(sim_a, sim_b):
    """False when one face score says owner and the other says stranger: not the same person, whatever the distance."""
    def looks(s):
        return None if s is None or STRANGER_SIM <= s < OWNER_SIM else s >= OWNER_SIM
    a, b = looks(sim_a), looks(sim_b)
    return a is None or b is None or a == b


def distance(a, b):
    return float(np.linalg.norm(a - b))


def fuse_objects(objects, found, t):
    """Robot and ball positions: the average of the cameras that see them in this frame. found: {camera: {name: (box, conf, pixel, metres)}}."""
    for name, obj in objects.items():
        points = [found[cam][name][3] for cam in found if name in found[cam]]
        if points:
            obj.update_position(np.mean(points, axis=0), t)


class FusedPerson:
    """One person in the room: which track in each camera is them, their position and their owner score."""

    def __init__(self, pid):
        self.id = pid
        self.members = {}            # camera name -> track id
        self.floor_m = None          # the position
        self.raw_m = None            # the same position from the unsmoothed measurements, None when not placed in this frame
        self.label = "unknown"
        self.sim = None
        self.placed = -1e9           # when the person was last placed on the floor
        self.placed_by = []          # the cameras that placed them then
        self.seen = -1e9             # when a camera last detected them, placed or not
        self.offset = {}             # camera -> its point minus the two-camera average, learned while both cameras see the person
        self.apart_since = None
        self.trail = deque(maxlen=TRAIL_N)
        self.heading = None          # unit vector on the floor the body faces: the cameras' headings, weighted (orientation.py)
        self.heading_at = -1e9

    def located(self, t):
        """True while the position is known: placed at most HOLD_S ago."""
        return t - self.placed <= HOLD_S


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
                         if tid in self.cameras[cam] and compatible(self.fresh(self.cameras[cam][tid]), p.sim)}

    def split_people(self):
        """A person whose two tracks stay more than SPLIT_M apart for SPLIT_S is two people: the second track leaves, and the camera
        offsets learned from the wrong pair are forgotten."""
        for p in self.people.values():
            if not self.apart(p):
                p.apart_since = None
            elif p.apart_since is None:
                p.apart_since = self.t
            elif self.t - p.apart_since > SPLIT_S:
                p.members.popitem()
                p.apart_since, p.offset = None, {}

    def attach_free_tracks(self):
        """A track that belongs to nobody joins the nearest person with a known position who has no track from its camera detected in this frame."""
        pairs = [(distance(self.cameras[cam][tid].floor_m, p.floor_m), cam, tid, p)
                 for cam, tid in self.free_tracks() for p in self.people.values() if p.located(self.t) and self.open_for(p, cam)]
        attached = set()
        for d, cam, tid, p in sorted(pairs, key=lambda pair: pair[0]):
            if d <= ATTACH_M and compatible(self.fresh(self.cameras[cam][tid]), self.fresh_sim(p)) and (cam, tid) not in attached and self.open_for(p, cam):
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
            tracks = {cam: self.cameras[cam][tid] for cam, tid in p.members.items()}
            placed = {cam: tr for cam, tr in tracks.items() if tr.placed == self.t}
            if placed:
                self.place(p, placed)
            else:
                p.raw_m = None
            measured = [(tr.heading, tr.heading_w) for tr in tracks.values() if tr.heading is not None and tr.heading_at == self.t]
            heading = combine(measured)
            if heading is not None:
                p.heading, p.heading_at = heading, self.t
            elif self.t - p.heading_at > HEADING_HOLD_S:
                p.heading = None
            sims = [tr.sim for tr in tracks.values() if tr.sim is not None]
            if sims:
                p.sim = max(sims)
            if any(tr.last_seen == self.t for tr in tracks.values()):
                p.seen = self.t

    def merge_people(self):
        """Two people on the same spot that no camera sees as two bodies are one person, the closest pair first:
        the owner keeps their id, otherwise the older one does."""
        while True:
            pairs = [(distance(p.floor_m, q.floor_m), p, q) for p in self.people.values() for q in self.people.values()
                     if p.id < q.id and self.same_spot(p, q)]
            if not pairs:
                return
            _, p, q = min(pairs, key=lambda pair: pair[0])
            if q.label == "owner":
                p, q = q, p
            for cam, tid in q.members.items():           # a camera both have a track from keeps the one it sees now
                if cam not in p.members or self.seen(cam, tid):
                    p.members[cam] = tid
            p.sim = max((s for s in (p.sim, q.sim) if s is not None), default=None)
            p.label = "owner" if "owner" in (p.label, q.label) else p.label
            del self.people[q.id]

    def forget_people(self):
        """A person no camera detected for TRACK_TIMEOUT_S is forgotten; one still detected keeps their id and label, with or without a position."""
        for pid in [pid for pid, p in self.people.items() if self.t - p.seen > TRACK_TIMEOUT_S]:
            del self.people[pid]

    def decide_labels(self):
        """One owner at most: the best match above OWNER_SIM, kept while tracked unless clearly beaten, or beaten by someone whose owner-like
        face is fresh while the owner's is not. Everyone else is a stranger, except while nobody is the owner: then a person whose face was
        never seen is unknown."""
        judged = [p for p in self.people.values() if p.sim is not None]
        current = next((p for p in judged if p.label == "owner"), None)
        best = max(judged, key=lambda p: p.sim, default=None)
        owner = current
        if best is not None and best.sim >= OWNER_SIM and (current is None or best is current or best.sim > current.sim + OWNER_SWITCH_MARGIN):
            owner = best
        fresh = {p: self.fresh_sim(p) for p in judged}
        challengers = [p for p, s in fresh.items() if s is not None and s >= OWNER_SIM]
        if owner is not None and fresh[owner] is None and challengers:
            owner = max(challengers, key=fresh.get)
        for p in self.people.values():
            p.label = "owner" if p is owner else "unknown" if owner is None and p.sim is None else "stranger"

    # ---- Helpers of the steps ----
    def fresh(self, track):
        """The track's face score if a face was scored within FRESH_S, else None: after an id swap an old score may be someone else's."""
        return track.sim if self.t - track.looked <= FRESH_S else None

    def fresh_sim(self, p):
        """The best fresh face score among the person's tracks, or None."""
        return max((s for s in (self.fresh(self.cameras[cam][tid]) for cam, tid in p.members.items()) if s is not None), default=None)

    def seen(self, cam, tid):
        """True if this camera detected the track in this frame."""
        return self.cameras[cam][tid].last_seen == self.t

    def open_for(self, p, cam):
        """True if the person has no track from this camera detected in this frame, so that one may join."""
        return cam not in p.members or not self.seen(cam, p.members[cam])

    def free_tracks(self):
        """The tracks placed in this frame that belong to nobody: [(camera, track id)]."""
        taken = {(cam, tid) for p in self.people.values() for cam, tid in p.members.items()}
        return [(cam, tid) for cam, tracks in self.cameras.items() for tid, track in tracks.items()
                if (cam, tid) not in taken and track.placed == self.t]

    def second_box(self, cam, track):
        """True for a free track close to this camera's own recently placed track on a person, with no face score against it: a second box on that person."""
        for p in self.people.values():
            own = self.cameras[cam][p.members[cam]] if cam in p.members else None
            if (own is not None and self.t - own.placed <= HOLD_S and compatible(self.fresh(track), self.fresh_sim(p))
                    and distance(track.floor_m, own.floor_m) <= SECOND_BOX_M):
                return True
        return False

    def apart(self, p):
        """True if both cameras place the person in this frame, more than SPLIT_M apart."""
        tracks = [self.cameras[cam][tid] for cam, tid in p.members.items()]
        return len(tracks) == 2 and all(tr.placed == self.t for tr in tracks) and distance(tracks[0].floor_m, tracks[1].floor_m) > SPLIT_M

    def place(self, p, placed):
        """Position of a person from their tracks placed in this frame ({camera: track})."""
        if not p.located(self.t):            # the position was lost: the trail starts again
            p.trail.clear()
        # a camera's error grows with distance, so the closer camera counts more (weight 1 / distance²); each camera alone is
        # off by a roughly constant offset, learned while both cameras see the person and subtracted when only one does
        if len(placed) > 1:
            w = np.array([1 / tr.range_m ** 2 for tr in placed.values()])
            fused = w @ np.array([tr.floor_m for tr in placed.values()]) / w.sum()
            p.raw_m = w @ np.array([tr.raw_m for tr in placed.values()]) / w.sum()
            for cam, tr in placed.items():
                offset = tr.floor_m - fused
                p.offset[cam] = offset if cam not in p.offset else OFFSET_ALPHA * offset + (1 - OFFSET_ALPHA) * p.offset[cam]
        else:
            cam, tr = next(iter(placed.items()))
            fused = tr.floor_m - p.offset.get(cam, 0.0)
            p.raw_m = tr.raw_m - p.offset.get(cam, 0.0)
        p.floor_m = fused
        p.trail.append(p.floor_m)
        p.placed, p.placed_by = self.t, sorted(placed)

    def same_spot(self, p, q):
        """True for two people both placed in this frame within ATTACH_M, that no camera detects as two bodies now, whose fresh face scores do not contradict."""
        two_bodies = any(self.seen(cam, p.members[cam]) and self.seen(cam, q.members[cam]) for cam in set(p.members) & set(q.members))
        return (not two_bodies and p.placed == q.placed == self.t
                and distance(p.floor_m, q.floor_m) <= ATTACH_M and compatible(self.fresh_sim(p), self.fresh_sim(q)))
