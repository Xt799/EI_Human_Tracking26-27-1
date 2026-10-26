"""
Appearance signature of a person: a re-identification network turns a person crop into a 512-number
vector; the same person scores about 0.9 against their own signature, different people 0.55 to 0.75.
"""
import os
from pathlib import Path

import numpy as np
import torch

REID_WEIGHTS = "osnet_x1_0_msmt17.pt"   # downloaded by boxmot on first use
OWNER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "owner.npz")   # written by enrol_owner.py, not in git


class AppearanceEncoder:
    def __init__(self):
        from boxmot.reid.backends.pytorch_backend import PyTorchBackend
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.backend = PyTorchBackend(weights=Path(REID_WEIGHTS), device=device, half=False)

    def embed(self, frame, xyxy):
        """Unit-length appearance vector of the person inside box xyxy."""
        return self.backend.get_features(np.asarray(xyxy, np.float32).reshape(1, 4), frame)[0]


def similarity(vec, signature):
    """Best cosine similarity between a vector and the signature's vectors."""
    return float(np.max(signature @ vec))


def save_owner(signature, videos):
    """signature: vectors (n x 512); videos: the enrolment videos they came from."""
    np.savez(OWNER_FILE, signature=signature, videos=np.array(videos))


def load_owner():
    """Signature vectors (n x 512) and the list of enrolment videos, or (None, None) if nobody has been enrolled."""
    if not os.path.exists(OWNER_FILE):
        return None, None
    d = np.load(OWNER_FILE)
    return d["signature"], list(d["videos"])
