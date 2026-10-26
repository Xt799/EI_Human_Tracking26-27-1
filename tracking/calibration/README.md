# calibration/

Pixel → metres on the floor. Lens correction (`camera_matrix.npy`, `distortion_coefficients.npy`, 1900×1200), plus one floor map per camera position: `floor_map_<view>.npy` (used by the scripts), `.json` (clicked points), `.jpg` (grid overlay for checking).

Calibration has been done for both camera positions in the recordings, `steep` and `low`, and the results are saved here. The scripts load them directly; nothing needs to be rerun for the existing recordings. Both maps share the same origin (the tile corner at the red tape cross) and were checked against each other: the same tape marks give the same metre coordinates from either map.

Only a new camera position needs a new map:

```
.venv\Scripts\python.exe tracking\calibrate.py --view <name> --video <clip with the floor visible>
```

Click 6 tile corners spread over the whole floor, one of them the origin; type each one's tile count from the origin as `col row`. Tiles are 19.8 cm centre to centre.