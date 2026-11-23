# Dataset

Our own labelled images, from the recordings in `../videos/` (not in git). Two tasks:
- `robot_heading/` – crops around the robot with a `front` and a `rear` point: which way the robot faces.
- `detection/` – full frames with boxes around the robot and the ball, for the detector that finds them (and gives
  the heading model its crop). Most of the detector's ball training data are public images, in `detection/train/coco/` and
  `detection/train/openimages/` (git-ignored, downloaded by `download_balls.py`).

Every task has a `train/` and a `val/` folder, nothing else. Which stretches of a recording go to which is set in
`splits.yaml` (stretches of time, so neighbouring frames never end up on both sides); the scripts that add images put
them in the right folder. Where an image comes from is in its name:
`<recording>_<camera>_t<frame>` (the camera's own frame number), for a crop followed by `_x<left>_y<top>`, the crop's
top-left corner in the lens-corrected frame (a point in the crop is (left + x, top + y) in the frame).

## Files
- `recordings.yaml` – the recordings: the video of each camera, the floor map (camera position in
  `tracking/calibration/`), and the frame offset between paired cameras.
- `splits.yaml` – train / val as stretches of time per recording (per camera when the cameras are not paired).
- `robot_heading/train/`, `robot_heading/val/` – robot crops (`.jpg`) and their labels (`.json`).
- `detection/train/`, `detection/val/` – one folder per source:
  - `recordings/` – lens-corrected frames of our recordings and their boxes, reviewed by hand. The room_1 frames of
    the single-camera recordings come from the first robot/ball dataset (in git history up to commit `5794c00`), cut
    again from the videos with lens correction;
  - `screenshots/` (train only) – 16 pasted screenshots of the first room_1 videos, `pasted_<date>_<time>` (source
    frame unknown, not lens-corrected);
  - `coco/`, `openimages/` (train only) – public ball images with their boxes (git-ignored, from `download_balls.py`).
- `robot_heading/prelabels.json`, `detection/prelabels.json` – the pre-labels as the scripts wrote them. An image whose
  labels are still exactly these has not been reviewed and is left out of training.
- Images are in Git LFS.

Earlier datasets are only in the git history: the first single-camera robot/ball dataset (`dataset/{train,val,test}`)
and the wheel-point dataset (frames labelled with the four wheel centres) are both in commit `5794c00`
(`git checkout 5794c00 -- dataset` brings them back, together with the scripts of that commit).

## Robot heading
```
uv run python training/scripts/make_robot_crops.py [--step 10] [--heading-step 30] [--grid 4 3]
```
Goes through every recording, sorts each robot sighting by room, camera, split, where in the frame it is (a `--grid`
cell) and which way it faces (`--heading-step` degrees), and writes the clearest sighting of every such view that has
no crop yet, pre-labelled. Every crop thus shows the robot at least a little differently from all the others.
Labelling: key `1` front (middle of the front, where the gripper is), key `2` rear (middle of the back), both at the
same height on the robot; a green arrow shows rear -> front. Delete both points if the heading cannot be seen or there
is no robot (a false detection: it is kept as a negative example).
Training: [../training/README.md](../training/README.md).

## Detection
```
uv run python training/scripts/make_detection_frames.py [--val-every 4] [--train-every 8] [--train-rooms room_2]
uv run python training/scripts/download_balls.py      # public ball images -> detection/train/coco/, openimages/
```
Frames at fixed time steps (val: every camera's val stretch; train: the rooms given), pre-labelled with the current
detector. Labelling: key `3` box around the whole robot, key `4` box around every ball. Training:
[../training/README.md](../training/README.md).

## Labelling (X-AnyLabeling)
```
powershell -File training/labeling/label.ps1 dataset/detection/val/recordings
```
The settings are in `training/labeling/xanylabeling.yaml`; your global X-AnyLabeling settings are not changed.

## Pairing two cameras
```
uv run python training/scripts/sync_cameras.py VIDEO_CAMERA_1 VIDEO_CAMERA_2 --guess <s> --search <s>
```
It matches moving people and the robot between the two views and reports the offset, with how clear the match is.
The guess comes from the file names (`basler_record_<date>_<hhmmss>`): the two recording PCs' clocks differ by whole
hours, so compare minutes and seconds. Write the result into `recordings.yaml`.

## Recordings
| recording | cameras | offset | floor map |
|---|---|---|---|
| `room_1_20261012_1800` | 1, 2 | camera_1 = camera_2 − 23 frames (camera_2 is 198 s long) | `a4` / `b4` |
| `room_2_20261110_1709` | 1, 2 | not found: no overlap in time | none |
| `room_1_20261012_1658`, `_1712`, `_1719` | 1 | – | none (camera_1 was moved after `_1712`) |
| `room_2_20261110_1635` | 1 | – | none |
