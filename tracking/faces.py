"""
The owner's face. InsightFace finds, aligns and encodes the face inside the head region that the pose
keypoints point at: a 512-number vector. Measured on our recordings: the same person scores about 0.6
against their own faces, other people about 0.0.
"""
import os

import numpy as np

from common import KPT_CONF, LEFT_EYE, NOSE, RIGHT_EYE, SCRIPT_DIR

FACE_MODEL = "buffalo_l"       # InsightFace model pack (face detector + ArcFace), downloaded to ~/.insightface on first use
DET_SIZE = 320                 # detector input size for the head crop
MIN_EYE_PX = 8                 # eyes closer than this in the image: face too small or turned away
NOSE_MARGIN = 0.3              # the nose may be this far outside the eyes (fraction of the eye spacing) and still count as frontal
HEAD_HALF_EYES = 2.2           # half-width of the head crop, in eye spacings
HEAD_MIN_HALF_PX = 24          # but at least this many pixels
HEAD_CHIN = 1.3                # the crop reaches this many half-widths below the eyes, to include the chin
OWNER_DIR = os.path.join(SCRIPT_DIR, "owner")        # everything enrol_owner.py writes; not in git
OWNER_FILE = os.path.join(OWNER_DIR, "owner.npz")


def face_visible(kxy, kconf):
    """True when nose and both eyes are confident and the nose lies between the eyes: the face is turned to this camera."""
    if min(kconf[NOSE], kconf[LEFT_EYE], kconf[RIGHT_EYE]) < KPT_CONF:
        return False
    lx, rx = sorted((kxy[LEFT_EYE][0], kxy[RIGHT_EYE][0]))
    eye = rx - lx
    return eye >= MIN_EYE_PX and lx - NOSE_MARGIN * eye <= kxy[NOSE][0] <= rx + NOSE_MARGIN * eye


def head_box(kxy, width, height):
    """Square around the head, from the eye keypoints, clamped to the frame: (x1, y1, x2, y2)."""
    eye = abs(kxy[LEFT_EYE][0] - kxy[RIGHT_EYE][0])
    cx, cy = (kxy[LEFT_EYE] + kxy[RIGHT_EYE]) / 2
    half = max(HEAD_HALF_EYES * eye, HEAD_MIN_HALF_PX)
    return int(max(0, cx - half)), int(max(0, cy - half)), int(min(width, cx + half)), int(min(height, cy + HEAD_CHIN * half))


class FaceEncoder:
    """Face detector + ArcFace on the CPU: a few crops per second, and no CUDA library clash with torch."""

    def __init__(self):
        from insightface.app import FaceAnalysis
        from insightface.utils import face_align
        self.app = FaceAnalysis(name=FACE_MODEL, allowed_modules=["detection", "recognition"], providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(DET_SIZE, DET_SIZE))
        self.align = face_align.norm_crop

    def embed(self, frame, kxy):
        """(unit-length face vector, aligned 112x112 face image) of the largest face in the head crop, or None."""
        x1, y1, x2, y2 = head_box(kxy, frame.shape[1], frame.shape[0])
        crop = frame[y1:y2, x1:x2]
        faces = self.app.get(crop) if crop.size else []
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return face.normed_embedding, self.align(crop, landmark=face.kps)


def similarity(vec, signature):
    """Best cosine similarity between a face vector and the owner's vectors."""
    return float(np.max(signature @ vec))


def save_owner(signature, video):
    """signature: face vectors (n x 512); video: the enrolment video they came from."""
    os.makedirs(OWNER_DIR, exist_ok=True)
    np.savez(OWNER_FILE, signature=signature, video=video)


def load_owner():
    """The owner's face vectors (n x 512) and the enrolment video name, or (None, None) if nobody has been enrolled."""
    if not os.path.exists(OWNER_FILE):
        return None, None
    d = np.load(OWNER_FILE)
    return d["signature"], str(d["video"])
