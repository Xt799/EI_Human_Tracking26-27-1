#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------- VM-friendly GUI env (must be before importing cv2) --------
import os
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["QT_X11_NO_MITSHM"] = "1"
os.environ["QT_OPENGL"] = "software"  # helps HighGUI/Qt in VMs
# os.environ["LIBGL_DRI3_DISABLE"] = "1"  # uncomment if needed on your VM

import time
import cv2
import numpy as np
from pypylon import pylon
from ultralytics import YOLO

# ========================= CONFIG =========================
POSE_MODEL_PATH = "./yolo11s-pose.pt"   # COCO 17-kp pose model
CAP_MODEL_PATH  = "/home/haichen/ros2_ws/src/cap_detection/cap_detection/best.pt"           # single-class 'cap' model

CAM_WIDTH, CAM_HEIGHT = 1900, 1200
DISPLAY_SCALE = 0.60
POSE_CONF_THRES = 0.25
CAP_CONF_THRES  = 0.30
SHOW_FPS = True

# Calibration files
CALIB_DIR = "./src/basler_yolo_node"
CAMERA_MATRIX_PATH = os.path.join(CALIB_DIR, "camera_matrix.npy")
DIST_COEFFS_PATH   = os.path.join(CALIB_DIR, "distortion_coefficients.npy")

# Ground-plane size for homography (meters) — edit for your field
PLANE_WIDTH_M, PLANE_HEIGHT_M = 3.60, 2.10
WARP_SCALE_PX_PER_M = 100  # only if you later want a bird’s-eye warp

# ---- Window name (single persistent window) ----
WIN = "Human + Cap — calibrated + ROI XY"

# ---- COCO keypoint indices ----
NOSE = 0
LEFT_EYE, RIGHT_EYE   = 1, 2
LEFT_EAR, RIGHT_EAR   = 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP   = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

LEFT_KPTS  = {1, 3, 5, 7, 9, 11, 13, 15}
RIGHT_KPTS = {2, 4, 6, 8, 10, 12, 14, 16}
SKELETON_PAIRS = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 6), (11, 12), (5, 11), (6, 12)
]

# ---- Colors (BGR) ----
COL_LEFT          = (0, 0, 255)      # 左侧：红色
COL_RIGHT         = (0, 255, 0)      # 右侧：绿色
COL_MID           = (220, 220, 220)  # 中性（跨左右）
COL_CENTER_DOT    = (255, 255, 255)  # 身体中心点
COL_OWNER_BOX     = (255, 0, 0)      # Owner BBox：蓝色
COL_STRANGER_BOX  = (0, 255, 255)    # Stranger BBox：黄色
COL_CAP           = (170, 0, 170)    # 帽子：紫色
COL_LINK          = (0, 255, 255)    # 连接线：黄色
TEXT_BG           = (255, 255, 255)
TEXT_COL          = (0, 0, 0)
FONT              = cv2.FONT_HERSHEY_SIMPLEX

# ========================= UTILITIES ======================
def put_label(img, text, org, bg_color=TEXT_BG, txt_color=TEXT_COL, scale=0.6, thickness=2):
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x, y - th - baseline - 4), (x + tw + 6, y + 4), bg_color, -1)
    cv2.putText(img, text, (x + 3, y - 4), FONT, scale, txt_color, thickness)

def box_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)

def body_center_from_feet(kps):
    """Midpoint between ankles if both valid; else NaN (pixel coords)."""
    if kps is None or len(kps) <= RIGHT_ANKLE:
        return np.array([np.nan, np.nan], dtype=np.float32)
    l = kps[LEFT_ANKLE]; r = kps[RIGHT_ANKLE]
    if np.any(np.isnan(l)) or np.any(np.isnan(r)):
        return np.array([np.nan, np.nan], dtype=np.float32)
    return (l + r) * 0.5

def safe_person_center(kps, box):
    """Prefer feet-midpoint; fallback to bbox center (pixel coords)."""
    c = body_center_from_feet(kps) if kps is not None else np.array([np.nan, np.nan], dtype=np.float32)
    if np.isnan(c).any():
        c = box_center(box)
    return c

def keep_top_two_people(ppl_xyxy, ppl_kxy, ppl_conf):
    """Return at most two people (by confidence if available, else by bbox area)."""
    n = len(ppl_xyxy)
    if n <= 2:
        return ppl_xyxy, ppl_kxy, ppl_conf
    if ppl_conf is not None and len(ppl_conf) == n and n > 0:
        order = np.argsort(-ppl_conf)
    else:
        areas = (ppl_xyxy[:, 2] - ppl_xyxy[:, 0]) * (ppl_xyxy[:, 3] - ppl_xyxy[:, 1])
        order = np.argsort(-areas)
    sel = order[:2]
    ppl_xyxy = ppl_xyxy[sel]
    ppl_kxy  = ppl_kxy[sel] if len(ppl_kxy) else ppl_kxy
    ppl_conf = ppl_conf[sel] if (ppl_conf is not None and len(ppl_conf)) else ppl_conf
    return ppl_xyxy, ppl_kxy, ppl_conf

def homography_map_points(H, pts_xy):
    """Map pixel points -> ROI XY meters using homography (cv2.perspectiveTransform)."""
    if H is None or len(pts_xy) == 0:
        return np.empty((0, 2), dtype=np.float32)
    pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return mapped

def homography_map_points_inv(Hinv, pts_xy_m):
    """Map ROI XY meters -> pixel points using inverse homography."""
    if Hinv is None or len(pts_xy_m) == 0:
        return np.empty((0, 2), dtype=np.float32)
    pts = np.asarray(pts_xy_m, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, Hinv).reshape(-1, 2)
    return mapped

def pick_roi_points(frame, display_scale=0.5, win="Select ROI (click TL, TR, BL, BR)"):
    pts = []
    disp = cv2.resize(frame.copy(), None, fx=display_scale, fy=display_scale)
    def click(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            ox, oy = int(x / display_scale), int(y / display_scale)
            pts.append((ox, oy))
            cv2.circle(disp, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(win, disp)
    cv2.imshow(win, disp)
    cv2.setMouseCallback(win, click)
    while True:
        if len(pts) == 4 or cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow(win)
    return np.array(pts, dtype=np.float32)

def build_homography_from_clicks(frame_u):
    """
    Click TL, TR, BL, BR on the undistorted frame.
    Returns H (pixel->meters), M (optional warp), warp size.
    """
    roi_pts = pick_roi_points(frame_u, display_scale=0.5)
    if roi_pts.shape[0] != 4:
        print("⚠️  ROI not selected; homography disabled.")
        return None, None, None, None

    # World (meters) with origin at TL, X→right, Y→down
    world_pts = np.array([
        [0.0, 0.0],
        [PLANE_WIDTH_M, 0.0],
        [0.0, PLANE_HEIGHT_M],
        [PLANE_WIDTH_M, PLANE_HEIGHT_M]
    ], dtype=np.float32)

    # Homography: pixels -> meters (points on the plane)
    H, _ = cv2.findHomography(roi_pts, world_pts)

    # Optional top-down warp (not used now, but returned)
    warp_w = int(PLANE_WIDTH_M * WARP_SCALE_PX_PER_M)
    warp_h = int(PLANE_HEIGHT_M * WARP_SCALE_PX_PER_M)
    dst_pts = np.array([[0, 0], [warp_w, 0], [0, warp_h], [warp_w, warp_h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(roi_pts, dst_pts)

    return H, M, warp_w, warp_h

# -------- Orientation (feet-only, on ROI plane with right-hand rule) --------
def angle_deg_signed(vec2):
    """
    返回与 ROI 轴对齐的**有符号角度**（单位度）：
      +X -> 0, +Y -> +90, -Y -> -90, -X -> +180
    取值范围 (-180, 180]，并把 -180 规整成 +180。
    """
    if vec2 is None or not np.all(np.isfinite(vec2)):
        return np.nan
    n = np.linalg.norm(vec2)
    if n < 1e-6:
        return np.nan
    ang = np.degrees(np.arctan2(vec2[1], vec2[0]))  # (-180, 180]
    if ang <= -180.0:
        ang += 360.0
    if abs(ang + 180.0) < 1e-6:
        ang = 180.0
    return ang

def forward_from_feet_on_plane(kps, H):
    """
    用左右脚踝在 ROI 地面坐标系下计算前向：
      s = R_ankle_m - L_ankle_m
      forward f = (-s_y, s_x)    # 右手法则：s × f = +Z
    返回：(center_m, f_unit, yaw_deg_signed)。任一失败 -> (None, None, np.nan)
    """
    if H is None or kps is None or len(kps) <= RIGHT_ANKLE:
        return None, None, np.nan
    L = kps[LEFT_ANKLE]; R = kps[RIGHT_ANKLE]
    if np.any(np.isnan(L)) or np.any(np.isnan(R)):
        return None, None, np.nan

    feet_m = homography_map_points(H, [L, R]).astype(np.float32)  # [[Lx,Ly],[Rx,Ry]] in meters
    s = feet_m[1] - feet_m[0]                                     # right - left
    n = np.linalg.norm(s)
    if n < 1e-6:
        return None, None, np.nan

    f = np.array([-s[1], s[0]], dtype=np.float32)                 # right-hand forward
    fn = np.linalg.norm(f)
    if fn < 1e-6:
        return None, None, np.nan
    f_unit = f / fn

    center_m = (feet_m[0] + feet_m[1]) * 0.5
    yaw_deg  = angle_deg_signed(f_unit)
    return center_m, f_unit, yaw_deg

def draw_forward_arrow_from_plane(img, Hinv, center_m, f_unit, length_m=0.35, color=(255,255,255), thickness=2):
    """在图像上绘制基于平面坐标的前向箭头：平面两点 -> Hinv -> 像素，再画线。"""
    if Hinv is None or center_m is None or f_unit is None:
        return None, None
    p0_m = np.asarray(center_m, dtype=np.float32)
    p1_m = p0_m + f_unit * float(length_m)
    p0_px, p1_px = homography_map_points_inv(Hinv, [p0_m, p1_m])
    p0_px = tuple(np.round(p0_px).astype(int))
    p1_px = tuple(np.round(p1_px).astype(int))
    cv2.arrowedLine(img, p0_px, p1_px, color, thickness, tipLength=0.25)
    return p0_px, p1_px  # 便于在箭头端点附近标注文本

# ========================= MAIN ===========================
def main():
    # ---- Load models (fuse Conv+BN when available) ----
    pose_model = YOLO(POSE_MODEL_PATH)
    cap_model  = YOLO(CAP_MODEL_PATH)
    for m in (pose_model, cap_model):
        for attr in ("fuse", "model"):
            try:
                (m if attr == "fuse" else m.model).fuse()
                break
            except Exception:
                continue

    # ---- Basler camera (robust enumeration) ----
    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()
    if not devices:
        print("❌ No Basler camera found. Is it connected/in use?")
        return
    print("✅ Cameras:", [d.GetFriendlyName() for d in devices])
    camera = pylon.InstantCamera(factory.CreateDevice(devices[0]))
    camera.Open()
    try:
        camera.Width.SetValue(CAM_WIDTH)
        camera.Height.SetValue(CAM_HEIGHT)
    except Exception as e:
        print(f"Resolution set warning: {e}")
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    # ---- Calibration → undistort maps ----
    use_undistort = False
    map1 = map2 = None
    try:
        K = np.load(CAMERA_MATRIX_PATH)
        D = np.load(DIST_COEFFS_PATH)
        newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (CAM_WIDTH, CAM_HEIGHT), 0, (CAM_WIDTH, CAM_HEIGHT))
        map1, map2 = cv2.initUndistortRectifyMap(K, D, None, newK, (CAM_WIDTH, CAM_HEIGHT), cv2.CV_16SC2)
        use_undistort = True
        print("✅ Calibration loaded. Undistortion enabled.")
    except Exception as e:
        print(f"⚠️  Calibration not loaded ({e}). Running without undistortion.")

    # ---- Grab one frame to pick ROI & build homography ----
    print("▶️  Grab a frame to select ROI (press 'q' to abort)…")
    frame0 = None
    while camera.IsGrabbing():
        grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grab.GrabSucceeded():
            frame0 = converter.Convert(grab).GetArray()
            grab.Release()
            break
    if frame0 is None:
        print("❌ Could not grab initial frame. Exiting.")
        camera.StopGrabbing(); camera.Close()
        return

    frame0_u = cv2.remap(frame0, map1, map2, interpolation=cv2.INTER_LINEAR) if use_undistort else frame0
    H, M, warp_w, warp_h = build_homography_from_clicks(frame0_u)
    if H is None:
        print("⚠️  Proceeding without homography mapping; XY labels will be skipped.")
    Hinv = None if H is None else np.linalg.inv(H).astype(np.float32)

    # ---- Create a single persistent window ----
    print("✅ Running. Press ESC / q to quit. Press p to pause, c to continue.")
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WIN, int(CAM_WIDTH * DISPLAY_SCALE), int(CAM_HEIGHT * DISPLAY_SCALE))
    paused = False

    prev_t, fps, frames = time.time(), 0.0, 0

    try:
        while camera.IsGrabbing():
            # If paused, wait here (still allow quit)
            if paused:
                k2 = cv2.waitKey(50) & 0xFF
                if k2 in (ord('c'), ord('p')):   # continue
                    paused = False
                elif k2 in (27, ord('q')):       # quit
                    break
                continue

            grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if not grab.GrabSucceeded():
                continue
            frame = converter.Convert(grab).GetArray()
            grab.Release()

            # Undistort per frame (fast with precomputed maps)
            frame_u = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR) if use_undistort else frame

            # ---------------- YOLO Pose ----------------
            pres = pose_model(frame_u, conf=POSE_CONF_THRES, verbose=False)
            pr = pres[0]

            ppl_xyxy = pr.boxes.xyxy.cpu().numpy() if (getattr(pr, "boxes", None) is not None and len(pr.boxes) > 0) else np.empty((0, 4), np.float32)
            ppl_conf = pr.boxes.conf.cpu().numpy()  if (getattr(pr, "boxes", None) is not None and len(pr.boxes) > 0) else np.empty((0,), np.float32)
            ppl_kxy  = pr.keypoints.xy.cpu().numpy() if (getattr(pr, "keypoints", None) is not None) else np.empty((0, 17, 2), np.float32)

            # Filter by confidence and trim to at most 2 people
            keep = [i for i in range(len(ppl_xyxy)) if (len(ppl_conf) == 0 or ppl_conf[i] >= POSE_CONF_THRES)]
            ppl_xyxy = ppl_xyxy[keep] if len(keep) else ppl_xyxy[:0]
            ppl_kxy  = ppl_kxy[keep]  if len(keep)  else ppl_kxy[:0]
            ppl_conf = ppl_conf[keep] if len(keep)  else ppl_conf[:0]

            ppl_xyxy, ppl_kxy, ppl_conf = keep_top_two_people(ppl_xyxy, ppl_kxy, ppl_conf)

            noses = []
            centers_px = []
            for i in range(len(ppl_kxy)):
                kps = ppl_kxy[i]
                nx, ny = kps[NOSE] if kps.shape[0] > NOSE else (np.nan, np.nan)
                noses.append([nx, ny] if not (np.isnan(nx) or np.isnan(ny)) else [np.nan, np.nan])
                centers_px.append(safe_person_center(kps, ppl_xyxy[i]))
            noses = np.asarray(noses, dtype=np.float32)
            centers_px = np.asarray(centers_px, dtype=np.float32)

            # ---------------- YOLO Cap ----------------
            cres = cap_model(frame_u, conf=CAP_CONF_THRES, verbose=False)
            cr = cres[0]
            cap_xyxy = cr.boxes.xyxy.cpu().numpy() if (getattr(cr, "boxes", None) is not None and len(cr.boxes) > 0) else np.empty((0, 4), np.float32)
            cap_conf = cr.boxes.conf.cpu().numpy()  if (getattr(cr, "boxes", None) is not None and len(cr.boxes) > 0) else np.empty((0,), np.float32)
            cap_centers_px = np.asarray([box_center(b) for b in cap_xyxy], dtype=np.float32)

            # ---------------- Owner assignment (two-people simplification) ----------------
            owner_person_idx = None
            owner_cap_idx    = None
            min_dist = np.inf

            # Prefer noses; fallback to person centers if noses are missing
            ref_pts = noses.copy() if (noses.size and not np.all(np.isnan(noses))) else centers_px.copy()
            valid_mask = ~np.isnan(ref_pts).any(axis=1) if ref_pts.size else np.zeros((0,), bool)
            ref_valid = ref_pts[valid_mask]

            if len(ref_valid) and len(cap_centers_px):
                try:
                    from scipy.spatial import cKDTree
                    tree = cKDTree(ref_valid)
                    dists, idxs = tree.query(cap_centers_px, k=1)
                    j = int(np.argmin(dists))
                    min_dist = float(dists[j])
                    valid_person_indices = np.nonzero(valid_mask)[0]
                    owner_person_idx = int(valid_person_indices[int(idxs[j])])
                    owner_cap_idx = int(j)
                except Exception:
                    diff = cap_centers_px[:, None, :] - ref_valid[None, :, :]
                    d2 = (diff * diff).sum(axis=2)
                    j, i = np.unravel_index(np.argmin(d2), d2.shape)
                    min_dist = float(np.sqrt(d2[j, i]))
                    valid_person_indices = np.nonzero(valid_mask)[0]
                    owner_person_idx = int(valid_person_indices[int(i)])
                    owner_cap_idx = int(j)

            # ---------------- DRAW ----------------
            canvas = frame_u.copy()

            # caps
            for ci, box in enumerate(cap_xyxy):
                x1, y1, x2, y2 = map(int, box)
                is_owner_cap = (owner_cap_idx is not None and ci == owner_cap_idx)
                color = COL_OWNER_BOX if is_owner_cap else COL_CAP
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                conf = float(cap_conf[ci]) if ci < len(cap_conf) else 0.0
                put_label(canvas, f"cap {conf:.2f}", (x1, y1))

            # people (bbox + colored sides + feet/bbox center + XY label + orientation-from-feet-on-plane)
            for pi, box in enumerate(ppl_xyxy):
                x1, y1, x2, y2 = map(int, box)
                is_owner = (owner_person_idx is not None and pi == owner_person_idx)
                box_color = COL_OWNER_BOX if is_owner else COL_STRANGER_BOX
                cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)
                put_label(canvas, "owner" if is_owner else "stranger", (x1, y1 - 6))

                # draw keypoints & skeleton (仅可视化)
                if pi < len(ppl_kxy):
                    kps = ppl_kxy[pi]
                    valid = ~np.isnan(kps).any(axis=1)
                    for k_idx, (px, py) in enumerate(kps):
                        if not valid[k_idx]:
                            continue
                        if k_idx in LEFT_KPTS:
                            color = COL_LEFT
                        elif k_idx in RIGHT_KPTS:
                            color = COL_RIGHT
                        else:
                            color = COL_MID
                        cv2.circle(canvas, (int(px), int(py)), 3, color, -1)
                    for (a, b) in SKELETON_PAIRS:
                        if a < len(kps) and b < len(kps) and valid[a] and valid[b]:
                            lc = COL_LEFT if (a in LEFT_KPTS and b in LEFT_KPTS) else (COL_RIGHT if (a in RIGHT_KPTS and b in RIGHT_KPTS) else COL_MID)
                            p1, p2 = kps[a], kps[b]
                            cv2.line(canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), lc, 2)

                # 像素中心点（脚/框）
                cxy_px = centers_px[pi] if pi < len(centers_px) else np.array([np.nan, np.nan], dtype=np.float32)
                if not np.isnan(cxy_px).any():
                    cx, cy = int(cxy_px[0]), int(cxy_px[1])
                    cv2.circle(canvas, (cx, cy), 5, COL_CENTER_DOT, -1)

                # === feet-only orientation on ROI plane ===
                yaw_deg = np.nan
                center_m = None
                f_unit = None
                if (H is not None) and (pi < len(ppl_kxy)):
                    center_m, f_unit, yaw_deg = forward_from_feet_on_plane(ppl_kxy[pi], H)

                # XY label（平面坐标）
                if (H is not None) and (cxy_px is not None) and (not np.isnan(cxy_px).any()):
                    xy_m = homography_map_points(H, [cxy_px])[0]  # (X, Y) in meters
                    put_label(canvas, f"X:{xy_m[0]:.2f}m  Y:{xy_m[1]:.2f}m", (int(cxy_px[0]), int(cxy_px[1]) - 12))

                # 前向箭头（用平面两点映射回像素来绘制，方向与地面坐标一致）
                if (Hinv is not None) and (center_m is not None) and (f_unit is not None):
                    arrow_color = COL_OWNER_BOX if is_owner else COL_STRANGER_BOX
                    p0_px, p1_px = draw_forward_arrow_from_plane(
                        canvas, Hinv, center_m, f_unit, length_m=0.35, color=arrow_color, thickness=2
                    )
                    if p1_px is not None and np.isfinite(yaw_deg):
                        # 用 "deg" 文本而不是 "°"
                        put_label(canvas, f"{yaw_deg:+.0f} deg" if yaw_deg != 180.0 else "180 deg",
                                  (p1_px[0] + 8, p1_px[1] + 8))

            # link owner nose <-> owner cap center (if nose available)
            if owner_person_idx is not None and owner_cap_idx is not None:
                nose = ppl_kxy[owner_person_idx, NOSE] if ppl_kxy.shape[0] > owner_person_idx else np.array([np.nan, np.nan], dtype=np.float32)
                capc = cap_centers_px[owner_cap_idx] if len(cap_centers_px) else np.array([np.nan, np.nan], dtype=np.float32)
                if not np.isnan(nose).any() and not np.isnan(capc).any():
                    cv2.line(canvas, (int(nose[0]), int(nose[1])), (int(capc[0]), int(capc[1])), COL_LINK, 2)
                    put_label(canvas, f"dist {min_dist:.1f}px", (int(capc[0]) + 6, int(capc[1]) + 6),
                              bg_color=(220, 220, 220))

            # HUD + FPS
            status = "Homography: ON" if H is not None else "Homography: OFF"
            put_label(canvas, f"{status} | people: {len(ppl_xyxy)} | caps: {len(cap_xyxy)}", (14, 36), bg_color=(230, 230, 230))
            frames += 1
            now = time.time()
            if now - prev_t >= 1.0:
                fps = frames / (now - prev_t)
                frames, prev_t = 0, now
            if SHOW_FPS:
                cv2.putText(canvas, f"FPS: {fps:.1f}", (14, 70), FONT, 0.8, (255, 255, 255), 3)
                cv2.putText(canvas, f"FPS: {fps:.1f}", (14, 70), FONT, 0.8, (0, 0, 0), 1)

            # ---- show continuous video in the single window ----
            disp = cv2.resize(canvas, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            cv2.imshow(WIN, disp)

            key = cv2.waitKey(1) & 0xFF  # non-blocking
            if key in (27, ord('q')):    # ESC or 'q'
                break
            elif key == ord('p'):
                paused = True

    except KeyboardInterrupt:
        pass
    finally:
        # Clean up camera
        try:
            camera.StopGrabbing()
            camera.Close()
        except Exception:
            pass
        # Safe destroy of the single window
        try:
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow(WIN)
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("✅ Clean exit.")

if __name__ == "__main__":
    main()
