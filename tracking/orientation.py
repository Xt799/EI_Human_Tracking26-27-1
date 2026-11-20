"""
Which way a person's body faces, on the floor, from the shoulder and hip keypoints of one camera.

The floor map H is only right for points on the floor, but two points at the same height (the two shoulders, the two
hips) are both pushed away from the camera by the same factor, so the direction of the line between them on the floor
is exact. The body faces at right angles to that line; which right angle comes from the person's left/right sides and
the handedness of the floor frame. Derivation and limits: README.md, "Body orientation".
"""
import numpy as np

from common import (KPT_CONF, LEFT_EYE, LEFT_HIP, LEFT_SHOULDER, NOSE, RIGHT_EYE, RIGHT_HIP, RIGHT_SHOULDER,
                    towards_camera)

MIN_WIDTH_RATIO = 0.25   # shoulder width / torso length below this: side-on to the camera, the line is too short to trust
FRONTAL_RATIO = 0.7      # shoulder width / torso length of a person square to the camera: full weight
TORSO_OF_BOX = 0.3       # torso length as a share of the box height, when the hips are hidden
HIDDEN_CONF = 0.2        # nose and both eyes below this: the face is turned away from the camera
FACE_CHECK_DOT = 0.3     # the face check may flip a heading only if it points this clearly towards or away from the camera
                         # (cosine); in a profile view the face says nothing and the model's left/right labels decide
HEADING_ALPHA = 0.3      # weight of the newest heading in the smoothed one
HEADING_HOLD_S = 0.5     # a heading not measured for this long is dropped


def _homogeneous(H, px):
    """Pixel -> (floor point in metres, w): w is the third homogeneous coordinate before the division."""
    q = H @ np.array([px[0], px[1], 1.0])
    return q[:2] / q[2], q[2]


def floor_frame(H, ref_px):
    """(handedness, w sign) of a camera, from a pixel that is surely on the floor (the bottom middle of the frame).
    handedness: +1 if the floor's x/y seen from above turn the same way as the image's x right / y down, -1 if mirrored.
    w sign: the sign of w for floor points; a pixel with the other sign is above the camera's height (see README)."""
    a, w = _homogeneous(H, ref_px)
    b, _ = _homogeneous(H, (ref_px[0] + 1, ref_px[1]))
    c, _ = _homogeneous(H, (ref_px[0], ref_px[1] - 1))
    jac = np.column_stack((b - a, a - c))   # derivatives along image x and image y
    return np.sign(np.linalg.det(jac)), np.sign(w)


def side_line(H, w_sign, left_px, right_px):
    """Floor direction of the person's left -> right line through two keypoints at the same height, or None."""
    l, wl = _homogeneous(H, left_px)
    r, wr = _homogeneous(H, right_px)
    if np.sign(wl) != np.sign(wr):
        return None                         # the line crosses the horizon: the two points are not at one height
    v = (r - l) * np.sign(wl) * w_sign      # above the camera's height the projection is mirrored through the camera
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def body_heading(kxy, kconf, box, H, frame, floor_px):
    """Unit vector on the floor the body faces, and its weight in (0, 1], or None.
    frame: (handedness, w sign) from floor_frame. floor_px: the person's floor point in pixels, or None."""
    vis = kconf >= KPT_CONF
    if not (vis[LEFT_SHOULDER] and vis[RIGHT_SHOULDER]):
        return None
    handed, w_sign = frame

    shoulders_mid = (kxy[LEFT_SHOULDER] + kxy[RIGHT_SHOULDER]) / 2
    hips = vis[LEFT_HIP] and vis[RIGHT_HIP]
    torso = np.linalg.norm(shoulders_mid - (kxy[LEFT_HIP] + kxy[RIGHT_HIP]) / 2) if hips else TORSO_OF_BOX * (box[3] - box[1])
    ratio = np.linalg.norm(kxy[RIGHT_SHOULDER] - kxy[LEFT_SHOULDER]) / max(torso, 1.0)
    if ratio < MIN_WIDTH_RATIO:
        return None

    lines = [side_line(H, w_sign, kxy[LEFT_SHOULDER], kxy[RIGHT_SHOULDER])]
    if hips:
        lines.append(side_line(H, w_sign, kxy[LEFT_HIP], kxy[RIGHT_HIP]))
    lines = [v for v in lines if v is not None]
    if not lines:
        return None
    v = np.sum(lines, axis=0)
    if np.linalg.norm(v) < 1e-6:
        return None                         # shoulders and hips disagree completely
    forward = handed * np.array([v[1], -v[0]])
    forward /= np.linalg.norm(forward)

    # left/right check: a visible face means the person faces this camera, a hidden one that they face away from it
    at = floor_px if floor_px is not None else np.array([(box[0] + box[2]) / 2, box[3]])
    to_cam = forward @ towards_camera(H, at)
    face = kconf[[NOSE, LEFT_EYE, RIGHT_EYE]]
    if (face.min() >= KPT_CONF and to_cam < -FACE_CHECK_DOT) or (face.max() < HIDDEN_CONF and to_cam > FACE_CHECK_DOT):
        forward = -forward
    return forward, float(min(ratio / FRONTAL_RATIO, 1.0))


def blend(old, new, alpha=HEADING_ALPHA):
    """Smoothed unit vector: the newest one weighted by alpha, renormalised. Averaging vectors, not angles, keeps 359° and 1° together."""
    if old is None:
        return new
    v = alpha * new + (1 - alpha) * old
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else new


def combine(headings):
    """Weighted mean direction of (unit vector, weight) pairs, or None."""
    if not headings:
        return None
    v = np.sum([h * w for h, w in headings], axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def degrees(heading):
    """Angle of a heading in the floor frame: 0 along +x, 90 along +y."""
    return float(np.degrees(np.arctan2(heading[1], heading[0])) % 360)
