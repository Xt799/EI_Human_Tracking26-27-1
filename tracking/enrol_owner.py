"""
Record who the owner is, from a short video in which they are visible.

    .venv\\Scripts\\python.exe tracking\\enrol_owner.py videos\\clips\\enrolment\\steep_a.mp4

Writes tracking/owner.npz (the owner's appearance signature) and two images to check:
enrol_candidates.jpg (everyone found, with their ids) and owner.jpg (the crops used).
"""
import argparse
import os

import cv2
import numpy as np

from appearance import OWNER_FILE, AppearanceEncoder, save_owner
from common import SCRIPT_DIR, PeopleDetector, lens_correction

FRAMES_PER_SEC = 3        # frames sampled per second of video
MIN_CROPS = 5             # a person seen in fewer frames is not offered as a candidate
N_EXTRA_VECTORS = 5       # posture variants stored next to the mean vector
MIN_CROP_SIM = 0.6        # a crop less similar than this to the owner's mean is not the owner (tracker id swap) and is dropped


def collect_people(video):
    """Tracks everyone in the video. Returns {id: [crop, ...]}."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    step = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 30) / FRAMES_PER_SEC))
    correct = lens_correction(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    detector = PeopleDetector()
    people = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = correct(frame)
        ids, boxes, _, _ = detector.detect(frame)
        for tid, (x1, y1, x2, y2) in zip(ids, boxes.astype(int)):
            if y2 - y1 >= 120 and x2 - x1 >= 40:     # skip people mostly out of frame
                people.setdefault(int(tid), []).append(frame[y1:y2, x1:x2].copy())
        for _ in range(step - 1):                   # skip ahead without decoding every frame
            cap.grab()
    cap.release()
    return {tid: c for tid, c in people.items() if len(c) >= MIN_CROPS}


def contact_sheet(crops, n=8):
    """One row of up to n crops."""
    tiles = [cv2.resize(c, (96, 192)) for c in crops[:: max(1, len(crops) // n)][:n]]
    row = cv2.hconcat(tiles)
    return cv2.copyMakeBorder(row, 0, 0, 0, n * 96 - row.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40))


def ask_owner_ids(people):
    """Shows the candidates and asks which ids are the owner (several when the owner got a new id after leaving the frame)."""
    rows = []
    for tid, crops in sorted(people.items()):
        row = contact_sheet(crops)
        cv2.putText(row, f"id {tid}  ({len(crops)} frames)", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        rows.append(row)
    sheet = os.path.join(SCRIPT_DIR, "enrol_candidates.jpg")
    cv2.imwrite(sheet, cv2.vconcat(rows))
    print(f"People found: {sorted(people)}. Open {sheet} to see them.")
    print("The same person can appear under several ids; type all of them, e.g.  1 7")
    while True:
        try:
            ids = sorted({int(x) for x in input("Which id(s) is the owner? ").split()})
            if ids and all(i in people for i in ids):
                return ids
        except ValueError:
            pass
        print(f"  type one or more of {sorted(people)}")


def signature_of(vecs):
    """Mean vector plus the N_EXTRA_VECTORS least typical crops (other postures), after dropping crops that are not the same person."""
    mean = vecs.mean(0)
    mean /= np.linalg.norm(mean)
    sims = vecs @ mean
    print(f"Similarity of the crops to their mean: min {sims.min():.2f}, mean {sims.mean():.2f}; {int((sims < MIN_CROP_SIM).sum())} crop(s) below {MIN_CROP_SIM} dropped")
    vecs, sims = vecs[sims >= MIN_CROP_SIM], sims[sims >= MIN_CROP_SIM]
    mean = vecs.mean(0)
    mean /= np.linalg.norm(mean)
    extra = vecs[np.argsort(sims)[:N_EXTRA_VECTORS]]
    return np.vstack([mean, extra]).astype(np.float32)


def main(video):
    people = collect_people(video)
    if not people:
        raise SystemExit("Nobody was tracked long enough in this video.")
    crops = [c for i in ask_owner_ids(people) for c in people[i]]
    encoder = AppearanceEncoder()
    vecs = np.array([encoder.embed(c, (0, 0, c.shape[1], c.shape[0])) for c in crops])
    save_owner(signature_of(vecs), [os.path.basename(video)])
    cv2.imwrite(os.path.join(SCRIPT_DIR, "owner.jpg"), contact_sheet(crops, 12))
    print(f"Owner enrolled from {len(crops)} crops: {OWNER_FILE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("video", help="video with the owner in view (a short clip is best)")
    main(ap.parse_args().video)
