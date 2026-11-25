# EI Human Tracking

## Repository layout
- `tracking/` – the application: tracking of people (owner / stranger), the robot with its heading and the ball from
  two cameras, in room coordinates; see [tracking/README.md](tracking/README.md).
- `training/` – fine-tuning of the robot and ball detector and the robot heading model (YOLO26n); see [training/README.md](training/README.md).
- `dataset/` – our labelled images (robot heading crops, robot and ball boxes); see [dataset/README.md](dataset/README.md).
- `videos/` – raw recordings, not in the repository.
- `cap_detection/` – cap detection scripts (Basler camera, YOLO pose, ROS 2 position publisher).

## Live cameras

Follow the [Windows live camera manual](tracking/WINDOWS_LIVE_CAMERAS.md) to install the drivers, download the trained weights, calibrate one or more Baslers and track people, the robot and the ball.

## Recorded playback
```
git lfs pull --include="training/runs/**/weights/*.pt"
uv sync                                                    # once, shared .venv (CUDA 12.8 torch)
uv run python tracking/track.py --video <camera_1.mp4> --position a4 --video <camera_2.mp4> --position b4 --offset 23
uv run python tracking/track.py --video <video.mp4>        # one camera, image only
```
- `--position`: the camera position the video before it was recorded from (its floor map in
  `tracking/calibration/`); it turns pixels into metres. Without it only the camera image is shown.
- `--offset N`: skip the first N frames of the second video, so the two show the same moment.

Windows: the camera views and the room seen from above. People with their body heading and owner / stranger label,
the robot with an arrow for the way it faces, and the ball. Keys: `p` pause, `c` continue, `q` quit. More:
[tracking/README.md](tracking/README.md).

Our room_1 recording (both cameras, 198 s):
```
uv run python tracking/track.py --video videos/room_1/camera_1/basler_record_20261012_180049.mp4 --position a4 --video videos/room_1/camera_2/AAA_basler_record_20261012_160049_FIXED.mp4 --position b4 --offset 23
```

Models:
- `training/runs/detector/weights/best.pt` – the robot and ball detector.
- `training/runs/heading/weights/last.pt` – the robot heading model (front and rear point on a crop around the robot).
- `tracking/models/yolo11s-pose.pt` – people with their keypoints (stock Ultralytics model).

## Owner and our recordings

Without an enrolled owner everyone is labelled unknown. Enrol the owner once, from a clip in which their face is visible; the owner of all our recordings is enrolled from `videos/enrol_20261123_144905.mp4`:
```
uv run python tracking/enrol_owner.py videos/enrol_20261123_144905.mp4
```

Our two-camera recordings are not in git; copy them into `videos/`. Except for pair 4, the two videos of a pair start on the same frame, so no `--offset` is needed:

| Pair | `--video` and `--position` | Room |
| --- | --- | --- |
| 1 | `videos/a1.mp4` `a1`, `videos/b1.mp4` `b1` | 3 × 3 m rectangle |
| 2 | `videos/a2.mp4` `a2`, `videos/b2.mp4` `b2` | 3 × 3 m rectangle |
| 3 | `videos/a3.mp4` `a3`, `videos/b3.mp4` `b3` | 3 × 3 m rectangle |
| 4 | `videos/room_1/camera_1/basler_record_20261012_180049.mp4` `a4`, `videos/room_1/camera_2/AAA_basler_record_20261012_160049_FIXED.mp4` `b4`, `--offset 23` | room 1, 5.3 × 6.1 m |
| 5 | `videos/a5.mp4` `a5`, `videos/b5.mp4` `b5` | 3 × 4 m taped rectangle with measured marks. The best calibrated pair |

```
uv run python tracking/track.py --video videos/a5.mp4 --position a5 --video videos/b5.mp4 --position b5
```

Every run writes `results/<start time>_positions.csv`: one row per person and object per frame, with the position in metres, the owner / stranger label and the heading. The columns, how the tracking works and its limitations: [tracking/README.md](tracking/README.md).
