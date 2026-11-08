# calibration/

Pixel → metres on the floor. Lens correction (`camera_matrix.npy`, `distortion_coefficients.npy`, 1900×1200), plus one floor map per camera position: `floor_map_<view>.npy` (used by the scripts), `.json` (clicked points), `.jpg` (grid overlay for checking). `frames/<view>.png` is the raw camera frame each map was made from.

Calibration has been done for both camera positions of the pair, `low` and `opposite`, and the results are saved here. The scripts load them directly; nothing needs to be rerun for the existing recordings. Both maps share the same origin (the tile corner at the red tape cross) and the same axis directions, and were checked against each other: the same tape marks give the same metre coordinates from either map.

Only a new camera position needs a new map. Save one raw frame with the floor visible as `frames/<view>.png`, then:

```
uv run python tracking/calibrate.py <view>
```

Click 6 tile corners spread over the whole floor; type each one's tile count from the origin as `col row`. Tiles are 19.8 cm centre to centre. The residuals printed at the end should be below a centimetre; one point far above the others means a miscounted tile.