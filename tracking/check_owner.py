"""
Test the enrolled owner on another video: is the owner recognised there, and is nobody else mistaken for them?

    python tracking/check_owner.py videos/a2.mp4
    python tracking/check_owner.py videos/a2.mp4 --start 60 --end 180      # only this stretch, in seconds

Uses tracking/owner/owner.npz from enrol_owner.py, so test on a video that was NOT used for enrolment.
No calibration needed: it only looks at faces. Everyone is tracked in the video; each face turned to the camera is
compared with the owner, the same comparison track.py makes. Prints a table per person and writes
tracking/owner/check_<video>.jpg: one row of faces per tracked id, with its scores and the verdict.
Same person scores about 0.6, others about 0.0; the verdict uses the thresholds of fusion.py.
"""
import argparse
import os
import time

import cv2
import numpy as np

from common import PeopleDetector, lens_correction
from faces import OWNER_DIR, FaceEncoder, face_visible, load_owner, similarity
from fusion import OWNER_SIM, STRANGER_SIM

FRAMES_PER_SEC = 3        # frames sampled per second of video
MIN_FACES = 3             # ids with fewer face shots are not judged
SHOW_FACES = 10           # face images per row in the check picture


def verdict(mean):
    return "OWNER" if mean >= OWNER_SIM else "stranger" if mean < STRANGER_SIM else "unsure"


def collect(video, start_s, end_s, signature):
    """Tracks everyone between start_s and end_s and scores every frontal face. Returns {id: [(score, face image, time)]}."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first, last = int(start_s * fps), min(n, int(end_s * fps)) if end_s else n
    step = max(1, round(fps / FRAMES_PER_SEC))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    correct = lens_correction(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    detector, encoder = PeopleDetector(), FaceEncoder()
    people, index, t0, shown = {}, first, time.time(), 0
    while index < last:
        ok, frame = cap.read()
        if not ok:
            break
        frame = correct(frame)
        ids, _, kxy, kconf = detector.detect(frame)
        for tid, k, kc in zip(ids, kxy, kconf):
            if face_visible(k, kc):
                found = encoder.embed(frame, k)
                if found is not None:
                    people.setdefault(int(tid), []).append((similarity(found[0], signature), found[1], index / fps))
        if time.time() - shown > 10:
            shown = time.time()
            print(f"  {index / fps:6.0f} s of {last / fps:.0f} s   ({time.time() - t0:.0f} s elapsed, {len(people)} people with faces so far)")
        for _ in range(step - 1):
            cap.grab()
        index += step
    cap.release()
    return people


def sheet_row(tid, shots):
    """Up to SHOW_FACES faces spread over the id's shots, each with its score, plus a header with the id's summary."""
    scores = np.array([s for s, _, _ in shots])
    picks = shots[:: max(1, len(shots) // SHOW_FACES)][:SHOW_FACES]
    tiles = []
    for s, img, _ in picks:
        tile = cv2.resize(img, (96, 96))
        colour = (255, 0, 0) if s >= OWNER_SIM else (0, 200, 255) if s < STRANGER_SIM else (200, 200, 200)
        cv2.rectangle(tile, (0, 78), (96, 96), colour, -1)
        cv2.putText(tile, f"{s:.2f}", (24, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        tiles.append(tile)
    row = cv2.hconcat(tiles)
    row = cv2.copyMakeBorder(row, 26, 0, 0, SHOW_FACES * 96 - row.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40))
    text = f"id {tid}: {len(shots)} faces, mean {scores.mean():.2f}, max {scores.max():.2f} -> {verdict(scores.mean())}"
    cv2.putText(row, text, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return row


def main(video, start_s, end_s):
    signature, enrolled_from = load_owner()
    if signature is None:
        raise SystemExit("No owner enrolled: run enrol_owner.py first.")
    print(f"Owner enrolled from {enrolled_from} ({len(signature) - 1} face shots); testing on {video}")
    if os.path.basename(video) == enrolled_from:
        print("Warning: this is the enrolment video itself, so this tests nothing about overfitting.")
    people = {tid: shots for tid, shots in collect(video, start_s, end_s, signature).items() if len(shots) >= MIN_FACES}
    if not people:
        raise SystemExit("Nobody's face was seen often enough.")

    print(f"\n{'id':>4} {'faces':>6} {'mean':>6} {'median':>7} {'min':>6} {'max':>6}  {'seen (s)':>14}  verdict")
    rows = []
    for tid, shots in sorted(people.items(), key=lambda kv: -np.mean([s for s, _, _ in kv[1]])):
        scores = np.array([s for s, _, _ in shots])
        times = [t for _, _, t in shots]
        print(f"{tid:>4} {len(shots):>6} {scores.mean():>6.2f} {np.median(scores):>7.2f} {scores.min():>6.2f} {scores.max():>6.2f}"
              f"  {min(times):>6.0f}-{max(times):<6.0f}  {verdict(scores.mean())}")
        rows.append(sheet_row(tid, shots))
    out = os.path.join(OWNER_DIR, f"check_{os.path.splitext(os.path.basename(video))[0]}.jpg")
    cv2.imwrite(out, cv2.vconcat(rows))
    print(f"\nThresholds: owner at mean >= {OWNER_SIM}, stranger below {STRANGER_SIM}. Faces per id: {out}")
    print("Good result: the owner's id(s) clearly above, everyone else near 0. The same person can appear under several ids.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("video", help="video to test on, not the one used for enrolment")
    ap.add_argument("--start", type=float, default=0, help="start at this second (default 0)")
    ap.add_argument("--end", type=float, default=None, help="stop at this second (default: end of video)")
    a = ap.parse_args()
    main(a.video, a.start, a.end)
