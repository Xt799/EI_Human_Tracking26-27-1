# Dataset

Labelled images from **paired** recordings: two cameras filming the same room at the same time. Pairing lets labels made
in one view be checked against, or carried over to, the other view through the floor.

## Files
- `recordings.yaml` – the recordings: the video of each camera, the floor map (camera position in
  `tracking/calibration/`), and the frame offset between the two cameras.
- `splits.yaml` – train / val / test as stretches of time per recording; both cameras of a moment are in the same split.
- `<recording>/camera_1/`, `<recording>/camera_2/` – images and X-AnyLabeling labels (`.json`). Images are in Git LFS.
  A paired image is named `t<camera_1 frame>.jpg` in both folders: the same name is the same moment.
- `backgrounds/` – background photos for synthetic data (git-ignored).

The first, single-camera robot/ball dataset is in `../dataset_old/` (git-ignored, kept locally; see
[../training/README.md](../training/README.md) for how to restore it from git).

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
