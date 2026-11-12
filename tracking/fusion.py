"""Merge the two cameras' person tracks into one set of people in room coordinates, and decide who the owner is."""
from collections import deque

import numpy as np

from common import FOOT_SOURCES, SMOOTH_ALPHA, TRACK_TIMEOUT_S
from orientation import HEADING_HOLD_S, combine

ATTACH_M = 0.6       # a new track joins a person this close that has no live track from its camera (both placed by the feet)
ATTACH_LOOSE_M = 1.2 # the same when one of them is placed by the knees or hips, which can be half a metre off
SPLIT_M = 1.5        # a person's two tracks this far apart for SPLIT_S seconds are two people
SPLIT_S = 0.5
TRAIL_N = 45         # positions kept for drawing the trail

# Owner rule, on the running face similarity (same person about 0.6, others about 0.0 on our recordings)
OWNER_SIM = 0.45               # at or above this: owner candidate
STRANGER_SIM = 0.25            # below this: stranger
OWNER_CHECKS = 2               # face comparisons needed before a person gets a label
OWNER_SWITCH_MARGIN = 0.10     # another person must beat the current owner by this much to take over


def compatible(sim_a, sim_b):
    """False when one face score says owner and the other says stranger: not the same person, whatever the distance."""
    def looks(s):
        return None if s is None or STRANGER_SIM <= s < OWNER_SIM else s >= OWNER_SIM
    a, b = looks(sim_a), looks(sim_b)
    return a is None or b is None or a == b


def distance(a, b):
    return float(np.linalg.norm(a - b))


class FusedPerson:
    """One person in the room: which track in each camera is them, their position and their owner score."""

    def __init__(self, pid, t):
        self.id = pid
        self.members = {}            # camera name -> (track id, time joined)
        self.floor_m = None
        self.source = None           # which keypoints placed the person
        self.label = "unknown"
        self.sim = None
        self.checks = 0
        self.last_seen = t           # when the person was last placed on the floor
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

    def update(self, cameras, t):
        """Updates the people from the cameras' live tracks ({camera: {track id: Person}}). Returns {(camera, track id): FusedPerson}."""
        people = self.people

        def seen(cam, tid):
            return tid in cameras[cam] and cameras[cam][tid].last_seen == t

        for p in people.values():                     # a track that ended, or whose face no longer matches the person's, leaves
            p.members = {cam: m for cam, m in p.members.items()
                         if m[0] in cameras[cam] and compatible(cameras[cam][m[0]].sim, p.sim)}

        for p in people.values():                     # two tracks far apart for a while, or with contradicting faces, are two people
            if len(p.members) < 2 or not all(seen(cam, tid) for cam, (tid, _) in p.members.items()):
                p.apart_since = None
                continue
            (ca, (ta, ja)), (cb, (tb, jb)) = p.members.items()
            pa, pb = cameras[ca][ta], cameras[cb][tb]
            if pa.floor_m is None or pb.floor_m is None or (distance(pa.floor_m, pb.floor_m) <= SPLIT_M and compatible(pa.sim, pb.sim)):
                p.apart_since = None
                continue
            if p.apart_since is None:
                p.apart_since = t
            if t - p.apart_since > SPLIT_S:
                a_feet, b_feet = pa.source == "ankles", pb.source == "ankles"
                del p.members[cb if (a_feet and not b_feet) or (a_feet == b_feet and ja <= jb) else ca]
                p.apart_since = None

        # a track that belongs to nobody joins the nearest person without a live track from its camera, else becomes a new person
        def open_for(p, cam):
            return cam not in p.members or not seen(cam, p.members[cam][0])

        taken = {(cam, m[0]) for p in people.values() for cam, m in p.members.items()}
        free = [(cam, tid) for cam, tracks in cameras.items() for tid, person in tracks.items()
                if (cam, tid) not in taken and person.placed == t]
        pairs = [(distance(cameras[cam][tid].floor_m, p.floor_m), cam, tid, p)
                 for cam, tid in free for p in people.values() if p.floor_m is not None and open_for(p, cam)]
        for d, cam, tid, p in sorted(pairs, key=lambda pair: pair[0]):
            limit = ATTACH_M if cameras[cam][tid].source == "ankles" and p.source == "ankles" else ATTACH_LOOSE_M
            if d <= limit and compatible(cameras[cam][tid].sim, p.sim) and (cam, tid) not in taken and open_for(p, cam):
                p.members[cam] = (tid, t)
                taken.add((cam, tid))

        def second_box(cam, tr):
            """Still free, yet close to a person this camera already tracks, with no face score against it: a second box on that person."""
            return any(p.floor_m is not None and not open_for(p, cam) and compatible(tr.sim, p.sim)
                       and distance(tr.floor_m, p.floor_m) <= ATTACH_LOOSE_M for p in people.values())

        for cam, tid in free:                         # the rest are new people, if placed by the feet: a knee or hip point can be a metre off
            tr = cameras[cam][tid]
            if (cam, tid) in taken or tr.source not in FOOT_SOURCES or second_box(cam, tr):
                continue
            p = FusedPerson(self.next_id, t)
            self.next_id += 1
            p.members[cam] = (tid, t)
            people[p.id] = p
            taken.add((cam, tid))

        # position from the tracks placed this frame, feet first. Each camera alone is off by a roughly constant offset (the ankle
        # joint is above the floor), so the offset is learned while both cameras see the person and subtracted when only one does
        for p in people.values():
            tracks = {cam: cameras[cam][tid] for cam, (tid, _) in p.members.items()}
            placed = {cam: tr for cam, tr in tracks.items() if tr.placed == t}
            if placed:
                feet = {cam: tr for cam, tr in placed.items() if tr.source == "ankles"} or placed
                if len(feet) > 1:
                    p.floor_m = np.mean([tr.floor_m for tr in feet.values()], axis=0)
                    for cam, tr in feet.items():
                        offset = tr.floor_m - p.floor_m
                        p.offset[cam] = offset if cam not in p.offset else SMOOTH_ALPHA * offset + (1 - SMOOTH_ALPHA) * p.offset[cam]
                else:
                    cam, tr = next(iter(feet.items()))
                    p.floor_m = tr.floor_m - p.offset.get(cam, 0.0)
                p.source = tr.source if len({tr.source for tr in feet.values()}) == 1 else "mixed"
                p.trail.append(p.floor_m)
                p.last_seen = t
            measured = [(tr.heading, tr.heading_w) for tr in tracks.values() if tr.heading is not None and tr.heading_at == t]
            heading = combine(measured)
            if heading is not None:
                p.heading, p.heading_at = heading, t
            elif t - p.heading_at > HEADING_HOLD_S:
                p.heading = None
            sims = [tr.sim for tr in tracks.values() if tr.sim is not None]
            if sims:
                p.sim = max(sims)
            p.checks = max(p.checks, sum(tr.checks for tr in tracks.values()))

        # two people with tracks from different cameras standing on the same spot are one person: the older one keeps the id
        for q in sorted(people.values(), key=lambda p: -p.id):
            for p in sorted(people.values(), key=lambda p: p.id):
                if p.id >= q.id or q.id not in people or set(p.members) & set(q.members):
                    continue
                if p.last_seen == q.last_seen == t and p.source == q.source == "ankles" and distance(p.floor_m, q.floor_m) <= ATTACH_M and compatible(p.sim, q.sim):
                    p.members.update(q.members)
                    p.sim = max((s for s in (p.sim, q.sim) if s is not None), default=None)
                    p.checks = max(p.checks, q.checks)
                    p.label = "owner" if "owner" in (p.label, q.label) else p.label
                    del people[q.id]

        for pid in [pid for pid, p in people.items() if t - p.last_seen > TRACK_TIMEOUT_S]:
            del people[pid]
        return {(cam, tid): p for p in people.values() for cam, (tid, _) in p.members.items()}

    def decide_labels(self):
        """One owner at most: the best match above OWNER_SIM. The owner keeps the label while tracked unless clearly beaten."""
        people = list(self.people.values())
        current = next((p for p in people if p.label == "owner"), None)
        judged = [p for p in people if p.checks >= OWNER_CHECKS and p.sim is not None]
        if not judged:
            return
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
