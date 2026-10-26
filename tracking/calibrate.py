"""
calibrate.py  --  build the floor map (pixel -> metres homography) for one viewpoint.

Set VIDEO_PATH, FRAME_SEC and VIEW below, then run from the repo root:
    .venv\\Scripts\\python.exe tracking\\calibrate.py

What happens:
  1. one frame is read from the video (at FRAME_SEC) and lens-corrected
  2. a window shows it; click POINTS tile corners, spread over the WHOLE floor:
     near and far, left and right. More points average out click errors.
     Each click is snapped to the nearest real corner in the image.
  3. in the terminal, type each point's tile coordinates (columns, rows) counted
     from the origin corner; metres = tiles x TILE_PITCH_M
  4. the homography is saved to tracking/calibration/floor_map_<VIEW>.npy, together
     with a .json of the points and a .jpg overlay of the tile grid for checking

Keys in the window:  u = undo last click,  q = abort.
"""
import json
import os

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = os.path.join(SCRIPT_DIR, "calibration")

# ---- Settings ----
# The existing maps were made from: steep = 171200 at 20 s, low = 180049 at 30 s.
VIDEO_PATH = os.path.join(SCRIPT_DIR, "..", "videos", "basler_record_20261012_171200.mp4")
FRAME_SEC = 20          # second of the video to take the frame from (choose a moment with the floor visible)
VIEW = "steep"          # name of the camera position; the map is saved as floor_map_<VIEW>.npy
POINTS = 6              # tile corners to click, at least 4

TILE_PITCH_M = 0.198    # 19.4 cm tile + 0.4 cm gap
DISPLAY_SCALE = 0.5


def undistort(frame):
    """Lens correction with the stored camera matrix; returns the frame unchanged if files are missing."""
    try:
        K = np.load(os.path.join(CALIB_DIR, "camera_matrix.npy"))
        D = np.load(os.path.join(CALIB_DIR, "distortion_coefficients.npy"))
    except FileNotFoundError:
        print("No lens files found, using the raw frame.")
        return frame
    h, w = frame.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0, (w, h))
    return cv2.undistort(frame, K, D, None, newK)


def click_points(frame, n):
    """Show the frame, collect n clicks, snap each to the nearest corner (sub-pixel)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pts = []
    win = f"calibrate: click {n} tile corners spread over the whole floor (u = undo, q = abort)"

    def redraw():
        disp = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        for i, (x, y) in enumerate(pts):
            p = (int(x * DISPLAY_SCALE), int(y * DISPLAY_SCALE))
            cv2.circle(disp, p, 6, (0, 0, 255), 2)
            cv2.putText(disp, str(i + 1), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(win, disp)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < n:
            p = np.array([[x / DISPLAY_SCALE, y / DISPLAY_SCALE]], dtype=np.float32)
            # snap to the nearest corner within an 11x11 px window
            cv2.cornerSubPix(gray, p, (5, 5), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            pts.append((float(p[0, 0]), float(p[0, 1])))
            redraw()

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    redraw()
    while True:
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            cv2.destroyWindow(win)
            return None
        if k == ord("u") and pts:
            pts.pop()
            redraw()
        if len(pts) == n:
            cv2.waitKey(300)
            break
    cv2.destroyWindow(win)
    return pts


def ask_tile_coords(n):
    """Terminal input: tile column and row of each clicked corner, counted from the origin corner."""
    print("\nFor each clicked point, type its tile coordinates as  col row  (integers, counted from the origin corner).")
    print("Example: the origin itself is  0 0 ; the corner 3 tiles right and 5 tiles down is  3 5 .")
    coords = []
    for i in range(n):
        while True:
            try:
                c, r = map(float, input(f"  point {i + 1}: ").split())
                coords.append((c, r))
                break
            except ValueError:
                print("  two numbers please, e.g.  3 5")
    return coords


def draw_grid(frame, H, span=60):
    """Overlay the tile grid implied by H over the whole image (±span tiles around the origin), for visual checking.
    The origin tile corner is marked in red, the column axis in blue, the row axis in yellow."""
    Hinv = np.linalg.inv(H)
    h, w = frame.shape[:2]
    out = frame.copy()

    def px(c, r):
        p = cv2.perspectiveTransform(np.array([[[c * TILE_PITCH_M, r * TILE_PITCH_M]]], np.float32), Hinv)[0, 0]
        return p

    def seg(p, q, color, thick=1):
        # draw only if both ends are finite and roughly on the canvas (lines behind the camera explode)
        if np.all(np.isfinite([p, q])) and max(abs(p).max(), abs(q).max()) < 20 * max(w, h):
            cv2.line(out, tuple(p.astype(int)), tuple(q.astype(int)), color, thick, cv2.LINE_AA)

    for c in range(-span, span + 1):
        for r in range(-span, span):
            seg(px(c, r), px(c, r + 1), (0, 255, 0))
    for r in range(-span, span + 1):
        for c in range(-span, span):
            seg(px(c, r), px(c + 1, r), (0, 255, 0))
    seg(px(0, 0), px(3, 0), (255, 128, 0), 3)      # column axis
    seg(px(0, 0), px(0, 3), (0, 255, 255), 3)      # row axis
    o = px(0, 0)
    if np.all(np.isfinite(o)):
        cv2.circle(out, tuple(o.astype(int)), 8, (0, 0, 255), -1)
    return out


def main():
    if POINTS < 4:
        raise SystemExit("POINTS must be at least 4")

    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_MSEC, FRAME_SEC * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read a frame from {VIDEO_PATH}")
    frame = undistort(frame)

    pts_px = click_points(frame, POINTS)
    if pts_px is None:
        raise SystemExit("Aborted.")
    tiles = ask_tile_coords(len(pts_px))
    pts_m = [(c * TILE_PITCH_M, r * TILE_PITCH_M) for c, r in tiles]

    # Least-squares fit over all points (with exactly 4 points the fit is exact)
    H, _ = cv2.findHomography(np.array(pts_px, np.float32), np.array(pts_m, np.float32), method=0)

    # Self-check: map the clicked pixels through H and compare with the typed metres
    back = cv2.perspectiveTransform(np.array(pts_px, np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)
    err_cm = np.linalg.norm(back - np.array(pts_m), axis=1) * 100
    print(f"\nResidual per point (cm): {np.round(err_cm, 1)}")
    print("A few cm is normal. One point far above the others means its tile count is wrong; rerun.")

    os.makedirs(CALIB_DIR, exist_ok=True)
    base = os.path.join(CALIB_DIR, f"floor_map_{VIEW}")
    np.save(base + ".npy", H)
    with open(base + ".json", "w") as f:
        json.dump({"video": os.path.basename(VIDEO_PATH), "sec": FRAME_SEC, "tile_pitch_m": TILE_PITCH_M,
                   "points_px": pts_px, "points_tiles": tiles, "points_m": pts_m}, f, indent=2)
    overlay = draw_grid(frame, H)
    cv2.imwrite(base + ".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"Saved {base}.npy, .json and .jpg")
    print("Open the .jpg: the green grid must sit on the tile joints everywhere, not only at the clicked points.")

    cv2.imshow("check: grid should match the tiles (any key closes)", cv2.resize(overlay, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
