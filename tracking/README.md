# Tracking: people, owner and stranger, robot, ball

This folder turns the lab recordings into floor positions. A fixed camera looks at the room; for every frame we detect the people with a pose model and the robot and the ball with the YOLO26n model from `training/`, and convert each detection's position on the floor from pixels into metres. The output is a trajectory for every person and object, with each person labelled as the owner or a stranger.

## Running

```
.venv\Scripts\python.exe tracking\enrol_owner.py <video>             # once per session: who is the owner, from a short video of them
.venv\Scripts\python.exe tracking\track.py <video> <camera>          # people, owner, robot and ball; camera = steep or low
```

`<camera>` is the position the video was recorded from and selects the floor map; `camera` in place of a video file uses the live Basler camera. The environment is the shared one from the repo root (`uv sync`, or the existing `.venv`). The recordings are not in git: copy the four `basler_record_20261012_*.mp4` files into `videos/` at the repo root, the same folder `inference/live_view.py` uses. Three of them (`165859`, `171200`, `171939`) were made from a high camera position we call `steep`, one (`180049`) from a lower position we call `low`.

## How positions become metres

Pixels cannot be compared between frames or cameras, so every position is converted into room coordinates with a floor map: a 3×3 matrix, fitted once per camera position, that maps any pixel on the floor to metres from a fixed origin. The maps for both camera positions are in `calibration/` and are loaded by the scripts; nothing needs to be calibrated to run on the existing recordings. A new camera position needs a new map, made with `calibrate.py` as described in [calibration/README.md](calibration/README.md). The conversion is only valid for points on the floor, so a person's position is taken at the feet.

## How the owner is recognised

The owner is not identified by any fixed feature but enrolled per session. `enrol_owner.py` shows the people found in a short video of the owner, asks which one is the owner, and saves that person's appearance signature: vectors computed from their image by a person re-identification network, which describe clothing and build in a way that survives changes of posture and viewing angle. `track.py` compares every tracked person against the signature every half second. The best match above a threshold becomes the owner and keeps the label for as long as the track lives, even when the score drops while sitting or turning away; everyone else is a stranger. The signature depends on clothing, so enrolment is repeated when the owner or their clothes change. The owner file is not in git.

## Files

- `track.py` – the tracking loop: people with persistent IDs, robot and ball, floor positions in metres, owner label, drawing. Playback follows the video clock, dropping frames when processing is slower than the recording.
- `enrol_owner.py` – records who the owner is (see above); writes `owner.npz` and two check images.
- `calibrate.py` – builds a floor map from clicked tile corners.
- `common.py` – shared by the three scripts: paths, lens correction, people detection, floor map loading.
- `sources.py` – frame sources: video file at real speed, or the Basler camera.
- `appearance.py` – the appearance signature: encoder, similarity, owner file.
- `calibration/` – lens correction files and the floor maps `steep` and `low`.
- `models/yolo11s-pose.pt` – the pose model. Robot and ball weights are in `../training/runs/yolo26n_e30/weights/best.pt`.
- `tools/basler_record_video.py` – records the camera to MP4, used for the recordings above.
