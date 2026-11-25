# calibration/

Pixel → metres on the floor. Lens correction (`camera_matrix.npy`, `distortion_coefficients.npy`, 1900×1200, the Basler cameras), plus one floor map per camera position: `floor_map_<view>.npy` (used by the scripts), `.json` (clicked points, room bounds, grid pitch), `.jpg` (grid overlay for checking). `frames/<view>.png` is the raw camera frame each map was made from.

Every camera position needs its own map; a new room or a moved camera means a new calibration. The two cameras of a pair are calibrated in one call so that they share the origin, the axes and the room bounds.

## Any room: a measured rectangle

Four corners of a rectangle on the floor whose size you know. It can be tape on the floor, a mat, or a pattern in the carpet; measure its two sides. Both cameras must see all four corners. A larger rectangle gives more accurate positions far from it.

1. Name the corners on the floor, not in the image: A, B, C, D going around the rectangle. A is the origin, x runs from A to B and y from A to D. Write this down, or mark A with tape, so that every camera clicks the same physical corners.
2. Run, with the two videos of the pair and the rectangle's size in metres (A→B, then A→D), under new camera position names; existing names are overwritten:

   ```
   uv run python tracking/calibrate.py a6 b6 --video videos/a6.mp4 videos/b6.mp4 --rect 3.0 4.0
   ```

3. For each camera: step through the video to a frame where no one stands on the corners (a/d ±1 s, w/s ±10 s, Enter). Click A, B, C, D. The sketch in the top-right corner shows which corner comes next. Clicks snap to the nearest image corner; u undoes.
4. The rectangle is also the tracked area: x 0…W, y 0…H. A position more than 10 cm outside it (`WALL_SLACK_M` in `track.py`) is dropped, and the room map shows just this area.
5. Check the window at the end, also saved as `floor_map_<view>.jpg`. The 0.5 m grid must lie flat on the floor, and the same square must cover the same spot of floor in both cameras. With four points the fit is exact at the corners, so this picture is the only check.

Then track with the same names: `uv run python tracking/track.py --video videos/a6.mp4 --position a6 --video videos/b6.mp4 --position b6`.

## Other reference points

- `--metres`: click 6 points and type each one's `x y` in metres. Use this when the points are not a rectangle, or when a camera cannot see all four corners. Both cameras must use the same origin. The tracked area is the box around the points. The residuals printed should be a few centimetres; one point far above the others means its coordinates were typed wrong.

## Existing maps

`a1`–`a3` and `b1`–`b3`: a 3 × 3 m taped rectangle per pair. `a5` and `b5`: a 3 × 4 m taped rectangle with measured marks inside it; `a5` was made from the measuring clip itself, because camera A had turned slightly since the first calibration.

`a4` and `b4`, the tiled lab, made from clicked tile corners (19.8 cm centre to centre). Both share the origin, the tile corner at the red tape cross. Their room bounds are the walls: x −2.84 … 2.48 m, y −3.07 … 3.04 m, which is 14 ⅓ and 12 ½ tiles along x and 15 ½ and 15 ⅓ along y (5.3 × 6.1 m). They were checked against each other: the same tape marks give the same metre coordinates from either map. Their bounds were set by hand in the `.json` files (`room_m`); recalibrating them with `calibrate.py` would replace those bounds with the clicked points' area.

## Live Basler profiles

List serials with `uv run python tracking/track.py --list-cameras`. Calibrate all live views in one invocation with repeated `--basler SERIAL VIEW` and either `--rect W H` or `--metres`. Select a frame with the floor visible, click the common measured points, and accept the final grid preview. Use the same view names in tracking.

Live calibration writes `tracking/calibration/live/profiles.json` and grid overlays. Each profile records its serial, observed camera geometry, lens-file fingerprints and the room coordinate system shared by that calibration session. The tracker validates these on startup. Recalibrate after moving a camera, changing geometry or replacing the lens. Checked-in recording maps are historical. To use those camera positions live, repeat the measurement with the connected cameras and verify the new overlay against the physical floor marks.

The checked-in lens model is available for the original 1900×1200 Basler setup. To use different optics or frame geometry, supply the corresponding OpenCV `camera_matrix.npy` and `distortion_coefficients.npy` in `tracking/calibration/live/intrinsics/VIEW/` before calibrating. Display scaling affects previews only. The calibration and tracking both project from the full-size corrected frame.
