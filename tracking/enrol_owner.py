"""
Record who the owner is, from a short video in which their face is visible.

    uv run python tracking/enrol_owner.py videos/<clip with the owner's face>.mp4

Use a different video than the one you track with, otherwise the recognition is not tested at all.
Writes tracking/owner/: owner.npz (the owner's face vectors) and two images to check,
enrol_candidates.jpg (everyone whose face was seen, with their ids) and owner.jpg (the faces used).
"""
import argparse
import os

import cv2
import numpy as np

from common import PeopleDetector, lens_correction
from faces import OWNER_DIR, OWNER_FILE, FaceEncoder, face_visible, save_owner

FRAMES_PER_SEC = 3        # frames sampled per second of video
MIN_FACES = 3             # a person with fewer face shots is not offered as a candidate
MIN_FACE_SIM = 0.3        # a face less similar than this to the owner's mean is someone else (tracker id swap) and is dropped


def collect_faces(video):
    """Tracks everyone in the video and encodes their face whenever it is turned to the camera. Returns {id: [(vector, face image), ...]}."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    step = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 30) / FRAMES_PER_SEC))
    correct = lens_correction(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    detector, encoder = PeopleDetector(), FaceEncoder()
    people = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = correct(frame)
        ids, _, kxy, kconf = detector.detect(frame)
        for tid, k, kc in zip(ids, kxy, kconf):
            if face_visible(k, kc):
                found = encoder.embed(frame, k)
                if found is not None:
                    people.setdefault(int(tid), []).append(found)
        for _ in range(step - 1):                   # skip ahead without decoding every frame
            cap.grab()
    cap.release()
    return {tid: f for tid, f in people.items() if len(f) >= MIN_FACES}


def contact_sheet(faces, n=10):
    """One row of up to n face images (96 px), spread over the whole list."""
    tiles = [cv2.resize(f, (96, 96)) for f in faces[:: max(1, len(faces) // n)][:n]]
    row = cv2.hconcat(tiles)
    return cv2.copyMakeBorder(row, 0, 0, 0, n * 96 - row.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40))


def ask_owner_ids(people):
    """Shows the candidates' faces and asks which ids are the owner (several when the owner got a new id after leaving the frame)."""
    rows = []
    for tid, faces in sorted(people.items()):
        row = contact_sheet([img for _, img in faces])
        cv2.putText(row, f"id {tid}  ({len(faces)} faces)", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        rows.append(row)
    os.makedirs(OWNER_DIR, exist_ok=True)
    sheet = os.path.join(OWNER_DIR, "enrol_candidates.jpg")
    cv2.imwrite(sheet, cv2.vconcat(rows))
    print(f"People found: {sorted(people)}. Open {sheet} to see them.")
    print("The same person can appear under several ids; type all of them, e.g.  1 5")
    while True:
        try:
            ids = sorted({int(x) for x in input("Which id(s) is the owner? ").split()})
            if ids and all(i in people for i in ids):
                return ids
        except ValueError:
            pass
        print(f"  type one or more of {sorted(people)}")


def signature_of(vecs):
    """The owner's mean vector plus every face at least MIN_FACE_SIM like it; less similar faces are someone else."""
    mean = vecs.mean(0)
    mean /= np.linalg.norm(mean)
    sims = vecs @ mean
    print(f"Similarity of the faces to their mean: min {sims.min():.2f}, mean {sims.mean():.2f}; {int((sims < MIN_FACE_SIM).sum())} face(s) below {MIN_FACE_SIM} dropped")
    keep = vecs[sims >= MIN_FACE_SIM]
    if len(keep) == 0:
        raise SystemExit("The chosen id's faces do not agree with each other; pick a cleaner clip or another id.")
    mean = keep.mean(0)
    mean /= np.linalg.norm(mean)
    return np.vstack([mean, keep]).astype(np.float32)


def main(video):
    people = collect_faces(video)
    if not people:
        raise SystemExit("Nobody's face was seen often enough in this video.")
    faces = [f for i in ask_owner_ids(people) for f in people[i]]
    save_owner(signature_of(np.array([vec for vec, _ in faces])), os.path.basename(video))
    cv2.imwrite(os.path.join(OWNER_DIR, "owner.jpg"), contact_sheet([img for _, img in faces], 16))
    print(f"Owner enrolled from {len(faces)} face shots: {OWNER_FILE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("video", help="video with the owner's face in view (a short clip is best)")
    main(ap.parse_args().video)
