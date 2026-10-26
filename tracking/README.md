# Tracking: people, owner and stranger, robot, ball

This folder turns the lab recordings into floor positions. A fixed camera looks at the room; for every frame we detect the people with a pose model and the robot and the ball with the YOLO26n model from `training/`, and convert each detection's position on the floor from pixels into metres. The output is a trajectory for every person and object, with each person labelled as the owner or a stranger.

## Running

```
.venv\Scripts\python.exe tracking\track.py        # people, robot and ball on a recording; VIDEO_PATH at the top of the file
```

The environment is the shared one from the repo root (`uv sync`, or the existing `.venv`). The recordings are not in git: copy the four `basler_record_20261012_*.mp4` files into `videos/` at the repo root, the same folder `inference/live_view.py` uses. Three of them (`165859`, `171200`, `171939`) were made from a high camera position we call the steep view, one (`180049`) from a lower position we call the low view; the scripts pick the matching floor map from the recording id in the file name.

## How positions become metres

Pixels cannot be compared between frames or cameras, so every position is converted into room coordinates with a floor map: a 3×3 matrix, fitted once per camera position, that maps any pixel on the floor to metres from a fixed origin. The maps for both camera positions are in `calibration/` and are loaded by the scripts; nothing needs to be calibrated to run on the existing recordings. A new camera position needs a new map, made with `calibrate.py` as described in [calibration/README.md](calibration/README.md). The conversion is only valid for points on the floor, so a person's position is taken at the feet.

## Files

- `track.py` – the tracking loop: detects people with persistent IDs, the robot and the ball, computes their floor positions in metres and draws the result. The floor map is chosen from the recording id in the file name. Playback follows the video clock, dropping frames when processing is slower than the recording.
- `calibrate.py` – builds a floor map from clicked tile corners.
- `calibration/` – lens correction files and the floor maps `steep` and `low`.
- `models/yolo11s-pose.pt` – the pose model. Robot and ball weights are in `../training/runs/yolo26n_e30/weights/best.pt`.
- `tools/basler_record_video.py` – records the camera to MP4, used for the recordings above.
