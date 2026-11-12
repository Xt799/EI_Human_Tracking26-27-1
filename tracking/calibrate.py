"""
Build the floor maps (pixels -> metres on the floor) for the cameras of any room, by clicking floor points in each view.

    uv run python tracking/calibrate.py a1 b1 --video videos/a1.mp4 videos/b1.mp4 --rect 3.0 2.0
    uv run python tracking/calibrate.py a1 b1 --metres                       # reads calibration/frames/a1.png and b1.png

How the clicked points get their metres, one of:
  --rect W H    the 4 corners of a rectangle on the floor that you measured, W metres from corner A to B, H from A to D
                (tape, a mat, a carpet pattern; the method of new_with_pub_.py). A is the origin, x runs A->B, y runs A->D.
                The corners are named on the floor, not in the image, so every camera clicks the same physical points.
  --metres      any floor points; type each one's x y in metres

All cameras given in one call share the origin. The tracked area is the rectangle (with --metres: the box around the
points); positions outside it are dropped. Procedure: calibration/README.md.
Keys: frame picker a/d -1/+1 s, w/s -10/+10 s, Enter accept; clicking u undo, q abort.
"""
import argparse
import json
import os

import cv2
import numpy as np

from common import CALIB_DIR, ORIGIN_COLOUR, X_AXIS_COLOUR, Y_AXIS_COLOUR, lens_correction

POINTS = 6              # points to click with --metres, at least 4
DISPLAY_SCALE = 0.5
GRID_M = 0.5            # grid drawn over the floor in the check picture and on the room map
ENTER, ESC = (13, 10), 27
POINT_COLOUR, OUTLINE_COLOUR, GRID_COLOUR = (0, 0, 255), (255, 0, 255), (0, 255, 0)
CORNERS = "ABCD"


# ---- Getting a frame ----
def pick_frame(video, start_frame, name):
    """Lets the user step through the video to a frame where the floor points are not hidden. Returns the raw frame."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    last = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    index, shown, frame = max(0, start_frame), None, None
    win = f"{name}: pick a frame with the floor points visible (a/d 1 s, w/s 10 s, Enter accept, q abort)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    while True:
        if index != shown:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, f = cap.read()
            if ok:
                frame, shown = f, index
            disp = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            cv2.putText(disp, f"frame {shown}  ({shown / fps:.1f} s)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(win, disp)
        k = cv2.waitKey(30) & 0xFF
        step = {ord("a"): -1, ord("d"): 1, ord("w"): -10, ord("s"): 10}.get(k)
        if step is not None:
            index = int(np.clip(index + step * fps, 0, last))
        elif k in ENTER:
            break
        elif k in (ord("q"), ESC):
            raise SystemExit("Aborted.")
    cap.release()
    cv2.destroyWindow(win)
    print(f"{name}: frame {shown} of {video}")
    return frame


def load_frame(name, video, start_frame):
    """The raw frame to calibrate on: picked from the video and saved as frames/<name>.png, or read from that file."""
    path = os.path.join(CALIB_DIR, "frames", f"{name}.png")
    if video is not None:
        frame = pick_frame(video, start_frame, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, frame)
        return frame, path
    frame = cv2.imread(path)
    if frame is None:
        raise SystemExit(f"Could not read {path}: pass --video, or save one raw camera frame with the floor visible there.")
    return frame, path


# ---- Clicking ----
def rect_diagram(disp, w, h, current):
    """Small top-down sketch of the rectangle in the top-right corner, the corner to click next highlighted."""
    size = 150
    scale = (size - 40) / max(w, h)
    x0, y0 = disp.shape[1] - size - 10 + 20, 20 + 20
    cv2.rectangle(disp, (x0 - 20, y0 - 20), (x0 - 20 + size, y0 - 20 + int(h * scale) + 40), (255, 255, 255), -1)
    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    pts = [(int(x0 + cx * scale), int(y0 + cy * scale)) for cx, cy in corners]
    cv2.polylines(disp, [np.array(pts)], True, (80, 80, 80), 1)
    cv2.arrowedLine(disp, pts[0], pts[1], X_AXIS_COLOUR, 2, tipLength=0.15)
    cv2.arrowedLine(disp, pts[0], pts[3], Y_AXIS_COLOUR, 2, tipLength=0.15)
    for i, p in enumerate(pts):
        colour = POINT_COLOUR if i == current else (0, 0, 0)
        cv2.circle(disp, p, 7 if i == current else 4, colour, -1)
        cv2.putText(disp, CORNERS[i], (p[0] + (6 if i in (1, 2) else -18), p[1] + (16 if i in (2, 3) else -6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)


def click_points(frame, title, prompt, n, labels=None, decorate=None):
    """Collects n clicks on the frame, each snapped to the nearest image corner. prompt(i) is the text shown for the next
    click, decorate(disp, i) may draw more. Returns the clicks in pixels, None if aborted."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pts = []
    win = title

    def redraw():
        disp = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        for i, (x, y) in enumerate(pts):
            p = (int(x * DISPLAY_SCALE), int(y * DISPLAY_SCALE))
            cv2.circle(disp, p, 6, POINT_COLOUR, 2)
            cv2.putText(disp, labels[i] if labels else str(i + 1), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, POINT_COLOUR, 2)
        text = prompt(len(pts))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(disp, (5, 5), (15 + tw, 15 + th + 8), (255, 255, 255), -1)
        cv2.putText(disp, text, (10, 10 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        if decorate is not None:
            decorate(disp, len(pts))
        cv2.imshow(win, disp)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < n:
            p = np.array([[x / DISPLAY_SCALE, y / DISPLAY_SCALE]], dtype=np.float32)
            cv2.cornerSubPix(gray, p, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            pts.append((float(p[0, 0]), float(p[0, 1])))
            redraw()

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    redraw()
    while True:
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), ESC):
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


def ask_coords(n):
    """Asks for each clicked point's floor coordinates in metres from the origin."""
    print("\nFor each clicked point, type its position on the floor as  x y  in metres from your origin.")
    coords = []
    for i in range(n):
        while True:
            try:
                a, b = map(float, input(f"  point {i + 1}: ").split())
                coords.append((a, b))
                break
            except ValueError:
                print("  two numbers please, e.g.  1.5 2.0")
    return coords


# ---- Fitting and checking ----
def to_metres(H, pts_px):
    return cv2.perspectiveTransform(np.asarray(pts_px, np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)


def to_pixel(Hinv, xy):
    """Floor point in metres -> pixel, or None if it lies behind the camera."""
    p = Hinv @ np.array([xy[0], xy[1], 1.0])
    return None if p[2] <= 1e-9 else p[:2] / p[2]


def draw_overlay(frame, H, grid_m, room_m, ref_m):
    """Draws the metre grid implied by H over the image, the reference points, the tracked area and the origin and axes."""
    Hinv = np.linalg.inv(H)
    h, w = frame.shape[:2]
    out = frame.copy()

    def seg(a, b, colour, thick=1):
        p, q = to_pixel(Hinv, a), to_pixel(Hinv, b)
        if p is not None and q is not None and max(abs(p).max(), abs(q).max()) < 20 * max(w, h):
            cv2.line(out, tuple(p.astype(int)), tuple(q.astype(int)), colour, thick, cv2.LINE_AA)

    (x0, x1), (y0, y1) = room_m
    xs = np.arange(np.floor(x0 / grid_m), np.ceil(x1 / grid_m) + 1) * grid_m
    ys = np.arange(np.floor(y0 / grid_m), np.ceil(y1 / grid_m) + 1) * grid_m
    for x in xs:                       # short segments: a long line crossing the horizon would be drawn wrong
        for ya, yb in zip(ys[:-1], ys[1:]):
            seg((x, ya), (x, yb), GRID_COLOUR)
    for y in ys:
        for xa, xb in zip(xs[:-1], xs[1:]):
            seg((xa, y), (xb, y), GRID_COLOUR)
    for a, b in ((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0)):
        steps = np.linspace(0, 1, 30)
        for s, t in zip(steps[:-1], steps[1:]):
            seg(np.add(a, s * np.subtract(b, a)), np.add(a, t * np.subtract(b, a)), OUTLINE_COLOUR, 2)
    seg((0, 0), (3 * grid_m, 0), X_AXIS_COLOUR, 3)
    seg((0, 0), (0, 3 * grid_m), Y_AXIS_COLOUR, 3)
    for xy in ref_m:
        p = to_pixel(Hinv, xy)
        if p is not None:
            cv2.circle(out, tuple(p.astype(int)), 9, POINT_COLOUR, 2)
    o =to_pixel(Hinv, (0, 0))
    if o is not None and np.all(np.abs(o) < 20 * max(w, h)):
        cv2.circle(out, tuple(o.astype(int)), 8, ORIGIN_COLOUR, -1)
    return out


def calibrate_camera(name, frame, mode, rect):
    """Clicks and fit for one camera. Returns a dict with the floor map and what it was made from."""
    if mode == "rect":
        w, h = rect
        corners_m = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
        pts_px = click_points(
            frame, f"{name}: click the rectangle's corners A B C D (u undo, q abort)",
            lambda i: f"{name}: click corner {CORNERS[i]}  (x {corners_m[i][0]:.2f} m, y {corners_m[i][1]:.2f} m)" if i < 4 else "",
            n=4, labels=list(CORNERS), decorate=lambda disp, i: rect_diagram(disp, w, h, i))
        if pts_px is None:
            raise SystemExit("Aborted.")
        coords, pts_m = corners_m, corners_m
    else:
        pts_px = click_points(frame, f"{name}: click {POINTS} floor points spread over the whole floor (u undo, q abort)",
                              lambda i: f"{name}: click point {i + 1} of {POINTS}", n=POINTS)
        if pts_px is None:
            raise SystemExit("Aborted.")
        coords = pts_m = ask_coords(len(pts_px))

    H, _ = cv2.findHomography(np.array(pts_px, np.float32), np.array(pts_m, np.float32), method=0)   # least squares over all points
    if H is None:
        raise SystemExit(f"{name}: no floor map fits these points (three of them in a line?); rerun.")
    err_cm = np.linalg.norm(to_metres(H, pts_px) - np.array(pts_m), axis=1) * 100
    if len(pts_px) > 4:
        print(f"\n{name}: residual per point (cm): {np.round(err_cm, 1)}")
        print("A few cm is normal. One point far above the others means its coordinates are wrong; rerun.")

    return {"H": H, "points_px": pts_px, "points_coords": coords, "points_m": [list(map(float, p)) for p in pts_m]}


def room_bounds(results):
    """((x0, x1), (y0, y1)): the area spanned by the reference points, for a rectangle its 4 corners."""
    pts = np.array([p for r in results.values() for p in r["points_m"]])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return (round(float(lo[0]), 3), round(float(hi[0]), 3)), (round(float(lo[1]), 3), round(float(hi[1]), 3))


def main(a):
    if a.video is not None and len(a.video) != len(a.cameras):
        raise SystemExit(f"{len(a.cameras)} camera names but {len(a.video)} videos: give one video per camera, in the same order.")
    mode = "rect" if a.rect is not None else "metres"
    if mode != "rect" and POINTS < 4:
        raise SystemExit("POINTS must be at least 4")
    if mode == "rect":
        print(f"Rectangle {a.rect[0]} m (A->B, x) by {a.rect[1]} m (A->D, y). Decide now which physical corner is A, B, C, D,")
        print("going around the rectangle, and click the same corners in every camera.")

    frames, results = {}, {}
    for i, name in enumerate(a.cameras):
        frame, image_path = load_frame(name, None if a.video is None else a.video[i], a.frame)
        frames[name] = (lens_correction(frame.shape[1], frame.shape[0])(frame), image_path)
        results[name] = calibrate_camera(name, frames[name][0], mode, a.rect)

    room_m = room_bounds(results)
    grid_m = GRID_M
    print(f"\nTracked area: x {room_m[0][0]} .. {room_m[0][1]} m, y {room_m[1][0]} .. {room_m[1][1]} m"
          f" ({room_m[0][1] - room_m[0][0]:.2f} x {room_m[1][1] - room_m[1][0]:.2f} m)")

    overlays = []
    for name, r in results.items():
        frame, image_path = frames[name]
        base = os.path.join(CALIB_DIR, f"floor_map_{name}")
        np.save(base + ".npy", r["H"])
        with open(base + ".json", "w") as f:
            json.dump({"image": os.path.basename(image_path), "mode": mode, "rect_m": a.rect,
                       "calibrated_with": [c for c in a.cameras if c != name], "room_m": room_m, "grid_m": grid_m,
                       "points_px": r["points_px"], "points_coords": r["points_coords"], "points_m": r["points_m"]}, f, indent=2)
        overlay = draw_overlay(frame, r["H"], grid_m, room_m, r["points_m"])
        cv2.putText(overlay, name, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imwrite(base + ".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
        overlays.append(cv2.resize(overlay, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE))
        print(f"Saved {base}.npy, .json and .jpg")

    print("Check the .jpg files: the green grid must lie flat on the floor everywhere, and the same grid square must cover")
    print(f"the same spot of the floor in every camera ({grid_m} m squares, origin red, x blue, y yellow, tracked area magenta).")
    h = min(o.shape[0] for o in overlays)
    check = np.hstack([cv2.resize(o, (int(o.shape[1] * h / o.shape[0]), h)) for o in overlays])
    if check.shape[1] > 1900:
        check = cv2.resize(check, None, fx=1900 / check.shape[1], fy=1900 / check.shape[1])
    cv2.imshow("check: the grid should lie on the floor, the same in every camera (any key closes)", check)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.strip().splitlines()[1:]))
    ap.add_argument("cameras", nargs="+", help="names of the camera positions, e.g. a1 b1; writes calibration/floor_map_<name>.*")
    ap.add_argument("--video", nargs="+", help="one video per camera, to pick the calibration frame from (saved as frames/<name>.png); "
                                               "without it, calibration/frames/<name>.png is read")
    ap.add_argument("--frame", type=int, default=0, help="frame index the frame picker starts at (default 0)")
    how = ap.add_mutually_exclusive_group(required=True)
    how.add_argument("--rect", nargs=2, type=float, metavar=("W", "H"), help="click the 4 corners of a measured W x H m rectangle on the floor")
    how.add_argument("--metres", action="store_true", help=f"click {POINTS} floor points and type their x y in metres")
    main(ap.parse_args())
