# Dataset

Labelled robot images. Most come from **paired** recordings: two cameras filming the same room at the same time, so
labels made in one view can be checked against, or carried over to, the other view through the floor. Single-camera
images add robot views the paired part lacks (other rooms, places and angles).

## Files
- `recordings.yaml` – the recordings: the video of each camera, the floor map (camera position in
  `tracking/calibration/`), and the frame offset between the two cameras.
- `splits.yaml` – train / val / test as stretches of time per recording (per camera when the cameras are not paired);
  both cameras of a paired moment are in the same split.
- `<recording>/camera_1/`, `<recording>/camera_2/` – images and X-AnyLabeling labels (`.json`). Images are in Git LFS.
  Inside a recording's paired stretch an image is named `t<camera_1 frame>.jpg` in both folders: the same name is the
  same moment. Outside it (and in recordings without a partner) the name is the camera's own frame number.
- `backgrounds/` – background photos for synthetic data (git-ignored).

The first, single-camera robot/ball dataset is in `../dataset_old/` (git-ignored, kept locally; see
[../training/README.md](../training/README.md) for how to restore it from git).

## Adding images
```
uv run python training/scripts/make_paired_frames.py <recording> [--n 150]
```
Picks moments where the robot is seen by both cameras, as varied as possible, per split, and writes both views
(lens-corrected, as `tracking/` sees them).
```
uv run python training/scripts/make_single_frames.py <recording> <camera> [--n 50]
```
Single-camera images: picks robot views unlike the ones already in the folder (look and place in the image), skipping
the paired stretch, and pre-labels the wheels with the current pose model as ordinary points, to be corrected by hand.
Near-identical views are also left out when the training set is built (`build_pose.py`), so labelling more frames of a
still robot adds nothing.

## Labelling (X-AnyLabeling)
```
powershell -File training/labeling/label.ps1 dataset/room_1_20261012_1800/camera_1
```
Keys `1`-`4` switch to point mode with the wheel label already set (`wheel_1` ... `wheel_4`): press the key, click the
wheel, next key. Key `5` draws the robot box. Numbering, seen from above with the robot facing away (front = gripper
side): `wheel_1` front right, `wheel_2` rear right, `wheel_3` front left, `wheel_4` rear left; the front pair is
`wheel_1` + `wheel_3`, the rear pair `wheel_2` + `wheel_4`. Click the centre of the wheel (all four centres
are at the same height, so the heading from them is exact on the floor map; the wheel radius is corrected in code for
the position); leave out a wheel you cannot see. The settings are in `training/labeling/xanylabeling.yaml`; your global X-AnyLabeling
settings are not changed.

## Pairing two cameras
```
uv run python training/scripts/sync_cameras.py VIDEO_CAMERA_1 VIDEO_CAMERA_2 --guess <s> --search <s>
```
It matches moving people and the robot between the two views and reports the offset, with how clear the match is.
The guess comes from the file names (`basler_record_<date>_<hhmmss>`): the two recording PCs' clocks differ by whole
hours, so compare minutes and seconds. Write the result into `recordings.yaml`.

## Status
| recording | offset | floor map | notes |
|---|---|---|---|
| `room_1_20261012_1800` | camera_1 = camera_2 − 23 frames | `low` / `opposite` | paired for the first 198 s (camera_2 length) |
| `room_2_20261110_1709` | not found | none | no overlap in time found between the two videos |
| `room_1_20261012_1658`, `_1712`, `_1719` | single camera | none | camera_1 only |
| `room_2_20261110_1635` | single camera | none | camera_1 only |

Test sets: `room_1_20261012_1800` 150–198 s (paired, but the robot stands in places also seen in training) and the last
part of both `room_2_20261110_1709` videos (a room the paired data does not have).
