# Tracking: people, owner and stranger, robot, ball

This folder turns the lab recordings into floor positions. A fixed camera looks at the room; for every frame we detect the people with a pose model and the robot and the ball with the YOLO26n model from `training/`, and convert each detection's position on the floor from pixels into metres. The output is a trajectory for every person and object, with each person labelled as the owner or a stranger.

## Running

```
.venv\Scripts\python.exe tracking\people_tracking.py        # people on a clip; set VIDEO_PATH at the top of the file
```

The environment is the shared one from the repo root (`uv sync`, or the existing `.venv`). The recordings are not in the repo; the scripts expect them in `../videos/`, with purpose-named excerpts in `../videos/clips/`. Three recordings (`165859`, `171200`, `171939`) were made from a high camera position we call the steep view, one (`180049`) from a lower position we call the low view.

## How positions become metres

Pixels cannot be compared between frames or cameras, so every position is converted into room coordinates with a floor map: a 3×3 matrix, fitted once per camera position, that maps any pixel on the floor to metres from a fixed origin. The maps for both camera positions are in `calibration/` and are loaded by the scripts; nothing needs to be calibrated to run on the existing recordings. A new camera position needs a new map, made with `calibrate.py` as described in [calibration/README.md](calibration/README.md). The conversion is only valid for points on the floor, so a person's position is taken at the feet.

## Files

- `people_tracking.py` – detects people, assigns owner or stranger, computes floor positions and draws the result.
- `robot_toy_tracking.py` – robot and ball detection with ROS 2 output, from the previous group.
- `calibrate.py` – builds a floor map from clicked tile corners.
- `calibration/` – lens correction files and the floor maps `steep` and `low`.
- `models/yolo11s-pose.pt` – the pose model. Robot and ball weights are in `../training/runs/yolo26n_e30/weights/best.pt`.
- `tools/basler_record_video.py` – records the camera to MP4, used for the recordings above.
