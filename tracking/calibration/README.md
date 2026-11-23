# calibration/

Pixel → metres on the floor. Lens correction (`camera_matrix.npy`, `distortion_coefficients.npy`, 1900×1200, the Basler cameras), plus one floor map per camera position: `floor_map_<view>.npy` (used by the scripts), `.json` (clicked points, room bounds, grid pitch), `.jpg` (grid overlay for checking). `frames/<view>.png` is the raw camera frame each map was made from.

Every camera position needs its own map; a new room or a moved camera means a new calibration. The two cameras of a pair are calibrated in one call so that they share the origin, the axes and the room bounds.

## Any room: a measured rectangle

The method of `new_with_pub_.py`: four corners of a rectangle on the floor whose size you know. It can be tape on the floor, a mat, or a pattern in the carpet; measure its two sides. Both cameras must see all four corners. A larger rectangle gives more accurate positions far from it.

1. Name the corners on the floor, not in the image: A, B, C, D going around the rectangle. A is the origin, x runs from A to B and y from A to D. Write this down, or mark A with tape, so that every camera clicks the same physical corners.
2. Run, with the two videos (or `camera` frames) of the pair and the rectangle's size in metres (A→B, then A→D):

   ```
   uv run python tracking/calibrate.py a1 b1 --video videos/a1.mp4 videos/b1.mp4 --rect 3.0 2.0
   ```

3. For each camera: step through the video to a frame where no one stands on the corners (a/d ±1 s, w/s ±10 s, Enter). Click A, B, C, D. The sketch in the top-right corner shows which corner comes next. Clicks snap to the nearest image corner; u undoes.
4. The rectangle is also the tracked area: x 0…W, y 0…H. A position more than 20 cm outside it (`WALL_SLACK_M` in `track.py`) is dropped, and the room map shows just this area.
5. Check the window at the end, also saved as `floor_map_<view>.jpg`. The 0.5 m grid must lie flat on the floor, and the same square must cover the same spot of floor in both cameras. With four points the fit is exact at the corners, so this picture is the only check.

Then track with the same names: `uv run python tracking/track.py --video videos/a1.mp4 --position a1 --video videos/b1.mp4 --position b1`.

## Other reference points

- `--metres`: click 6 points and type each one's `x y` in metres. Use this when the points are not a rectangle, or when a camera cannot see all four corners. Both cameras must use the same origin. The tracked area is the box around the points. The residuals printed should be a few centimetres; one point far above the others means its coordinates were typed wrong.

## Existing maps

`a4` and `b4`, the tiled lab, made from clicked tile corners (19.8 cm centre to centre). Both share the origin, the tile corner at the red tape cross. Their room bounds are the walls: x −2.84 … 2.48 m, y −3.07 … 3.04 m, which is 14 ⅓ and 12 ½ tiles along x and 15 ½ and 15 ⅓ along y (5.3 × 6.1 m). They were checked against each other: the same tape marks give the same metre coordinates from either map. Their bounds were set by hand in the `.json` files (`room_m`); recalibrating them with `calibrate.py` would replace those bounds with the clicked points' area.
