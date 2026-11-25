# Tracking: people, owner and stranger, robot and its heading, ball

Two fixed cameras in opposite corners of the lab record the room at the same time. For every frame we detect the people with a pose model and the robot and the ball with the YOLO26n model from `training/` (and which way the robot faces with the heading model), convert each detection's position on the floor from pixels into metres, and merge the two views into one set of people and objects. The output is a trajectory for every person and object in room coordinates, with each person labelled as the owner or a stranger.

## Live cameras

Use one or more serial-selected Basler cameras to track people, robot heading and the ball in metres. For Windows installation, model downloads, camera discovery, calibration, testing and troubleshooting, see the [Windows live camera manual](WINDOWS_LIVE_CAMERAS.md).

## Recorded playback

```
uv run python tracking/calibrate.py a1 b1 --video videos/a1.mp4 videos/b1.mp4 --rect 4.0 3.0
uv run python tracking/enrol_owner.py videos/<clip with the owner's face>.mp4
uv run python tracking/track.py --video videos/a1.mp4 --position a1 --video videos/b1.mp4 --position b1 [--offset <frames>]
uv run python tracking/track.py --video videos/a1.mp4 [--position a1]
```
- `--video`: a video file, once or twice (two cameras).
- `--position`: right after its `--video`, the camera position it was recorded from: the name of its floor map in
  `calibration/` (`a1`, `b1`, ...). It turns pixels into metres. Without it the camera is shown in the image only
  (people, the robot with its heading arrow, the ball; no metres, no room map, no owner label).
- `--offset N`: the two videos start together after skipping the first N frames of the second one (negative N: of the
  first one), for recordings that did not start at the same moment.

The recordings are not in git: copy them into `videos/` at the repo root. They come in pairs, `a<n>` and `b<n>`, filmed at the same time by the two cameras; each pair has its own camera positions.

1. `calibrate.py` makes the floor maps of a pair, once per camera position: [calibration/README.md](calibration/README.md).
2. `enrol_owner.py` records the owner's face from a short clip in which their face is visible (any single video; use a different stretch than the one you track). It shows the faces it found per id and asks which id is the owner.
3. `track.py` takes one or two videos with their camera positions. Two recordings must start on the same frame or be given the `--offset` between them (the `a` recordings were re-timed to 30 fps to match their `b` partners; the untouched originals are in `videos/original/`). Windows open for each camera view and a top-down room map. Positions go to `results/<start time>_positions.csv` at the repo root, one row per frame per tracked thing; every run writes a new file. `REALTIME=True` drops frames to follow the video clock. Set it to `False` to process every frame.

The environment is the shared one from the repo root (`uv sync`). The face models (about 280 MB) download into `~/.insightface` on first use.

## How it works

**Metres.** A floor map per camera position, a 3×3 matrix fitted once from clicked floor points, turns any pixel on the floor into metres from a fixed origin. In any room the points are the corners of a measured rectangle on the floor; in the tiled lab they are tile corners. The two cameras of a pair are calibrated together, so they share the origin and the room bounds; the maps are in `calibration/`. A map is only valid for points on the floor, so a person is placed at the feet; a camera that does not see their feet does not place them. A new room or camera position needs new maps: [calibration/README.md](calibration/README.md).

**Merging.** Each camera tracks people on its own with ids that last while they stay in view. In room coordinates, a track within 0.6 m of a person known from the other camera is the same person; otherwise it is a new person. The position is the average of the cameras that see the feet. A camera that only sees someone's upper body, because they stand right in front of it, does not place them. The robot's box ends at its edge nearest the camera, so each camera's point is pushed back by the robot's radius before averaging. When one camera loses someone the other keeps them, and a track that comes back under a new id rejoins by position.

**Owner.** Nobody is identified by a fixed feature; the owner is enrolled per session, by face. Whenever a person's face is turned to one of the cameras, the face is encoded and compared with the enrolled face vectors every half second; with two opposite cameras one of them usually sees it. The best match above a threshold becomes the owner and keeps the label while tracked, even with the face hidden; everyone else is a stranger. On our recordings the owner scores about 0.6 against his own faces and other people about 0.0. A track that scores like the owner is never merged with a person that scores like a stranger, however close they stand, and a track whose score stops matching its person leaves it, so an id swapped inside one camera is corrected at the next face check. A new person is never created from a box within 1.2 m of a person the same camera already tracks unless its face score says otherwise: that is a second box on the same person. The owner file is not in git.

## Robot heading

`robot_heading.py`, with the heading model from `../training/` (`training/runs/heading/weights/last.pt`). For the robot
box each camera finds, a square crop of 2× the box goes through a keypoint model that finds a `front` point (where the
gripper is) and a `rear` point. Both are at the same height on the robot, so mapping them to the floor with the
camera's floor map keeps the direction rear → front exact (a plane parallel to the floor maps onto it only scaled about
the camera's foot point). The robot's heading is the two cameras' directions weighted by the model's confidence, then
smoothed over time like a person's (`HEADING_ALPHA`, `HEADING_HOLD_S`). It shows up as an arrow on the robot in each
camera view (what that camera measured), `facing … deg` in its label, an arrow on the room map and `heading_deg` in
the robot's CSV rows. On room_1 the two cameras agree within about 1–7° (median).

## Body orientation

Every person also gets the direction their **body** faces, from the shoulders and hips; it is the torso's direction, not where the head turns or the eyes look. It shows up as an arrow from the person's dot on the room map, as an arrow on the floor in each camera view (what that camera measured), as `facing … deg` in the camera labels and as `heading_deg` in the CSV: the angle in the floor frame, 0° along +x, 90° along +y (the room map's axes), empty when no camera could measure it. The code is in `orientation.py`.

### The math

**1. Raised points through the floor map.** The floor map H sends a pixel to the point where that pixel's viewing ray meets the floor. That is the right place only for points that really are on the floor. Take the camera at height `Hc`, standing above the floor point `C₀`, and a body point P at height `h` above floor point `P₀`. Its ray meets the floor further out, at

```
P' = C₀ + k · (P₀ − C₀),     k = Hc / (Hc − h)

   camera ●
          | \
          |   \  ray
       Hc |     ● P (shoulder, height h)
          |     | \
          |   h |   \
   ───────●─────●─────●──────── floor
         C₀    P₀     P'  = where H puts the shoulder
```

So H pushes a raised point away from the camera's floor point by the factor `k` (similar triangles). The position is wrong, but the factor depends only on the height. The two shoulders are at the same height, so both get the same `k`:

```
R' − L' = (C₀ + k(R₀ − C₀)) − (C₀ + k(L₀ − C₀)) = k · (R₀ − L₀)
```

The left→right shoulder line on the floor comes out stretched by `k` but in **exactly the right direction**, with no need for the camera height or the person's height. The same holds for the hips, at their own `k`. Both lines are turned into unit vectors and added.

If the camera is lower than the shoulders (`h > Hc`), `k` is negative: the ray goes up and only meets the floor "behind" the camera, and the line comes out reversed. In homogeneous coordinates, `H · (u, v, 1) = (x·w, y·w, w)` with `w` having one sign for every floor pixel and the other sign above the horizon. So the line is negated when the keypoints' `w` has the opposite sign from a floor pixel (the bottom middle of the frame). A pair with one point above the horizon and one below is skipped.

**2. From the side line to the facing direction.** With `v = R − L` (the person's own right minus their own left, as the pose model labels them) and "up" out of the floor, the body faces `forward = up × v`. In 2D that is `v` turned by 90°: `(−v_y, v_x)` if the floor's x/y axes, seen from above, turn like standard maths axes (counter-clockwise), and `(v_y, −v_x)` if they turn like the image's (x right, y down). Which one applies depends on the order of the corners A→B→C→D during calibration, so the code reads it from H: `s = sign(det J)`, where J is the Jacobian of H at a floor pixel (how the floor coordinates change for one pixel right and one pixel up). A camera looks down at the floor, so its image is an unmirrored view from above. `s = +1` means the floor frame turns like the image, `s = −1` that it is mirrored. Then

```
forward = s · (v_y, −v_x) / |v|
```

**3. How far to trust a camera.** The weight is the shoulder width in pixels divided by the torso length (shoulder middle to hip middle; 0.3 × the box height if the hips are hidden), capped at 1 when it reaches `FRONTAL_RATIO`. Square to the camera, facing it or with their back to it, the shoulders are wide and the direction is well defined. Side-on, the near shoulder hides the far one, the line is a few pixels long and a one-pixel error turns it by tens of degrees. Below `MIN_WIDTH_RATIO` the camera gives no heading.

**4. Left/right swaps.** Pose models sometimes swap the person's left and right, mostly when seen from behind, which turns the arrow by 180°. The face catches this. If the nose and both eyes are confidently visible, the person faces this camera, so `forward · towards_camera` must be positive. If none of them is visible, the person faces away, so it must be negative. A contradiction flips the arrow. The check only acts when the arrow points clearly towards or away from the camera (|cosine| > `FACE_CHECK_DOT`): in a profile view the face settles nothing.

**5. Smoothing and two cameras.** Angles cannot be averaged directly: the mean of 359° and 1° is 180°, the opposite direction. Directions are averaged as unit vectors and normalised again, and 359° and 1° average to 0°. Each camera's track smooths its heading over time this way (`HEADING_ALPHA` for the newest value). The person's heading is the weighted vector sum of the cameras that measured it this frame, so a camera that sees the person side-on gives way to the one that sees them face or back on. With no measurement for `HEADING_HOLD_S`, the heading is dropped.

### Limits

- It is the torso direction. A person looking over their shoulder still "faces" where their chest points; bending, twisting at the waist or lying down give a poor line.
- Arms raised across the chest or a carried object can move the shoulder keypoints.
- A left/right swap in a near-profile view cannot be caught: if only one camera sees the person, and side-on, the arrow can point backwards for a moment.

### Settings (top of `orientation.py`)

| Constant | Meaning |
| --- | --- |
| `MIN_WIDTH_RATIO` | shoulder width / torso length below which a camera's view is too side-on to measure |
| `FRONTAL_RATIO` | the ratio of a person square to the camera: full weight from here up |
| `TORSO_OF_BOX` | torso length as a share of the box height when the hips are hidden |
| `HIDDEN_CONF` | nose and both eyes below this confidence: the face is turned away |
| `FACE_CHECK_DOT` | how clearly (cosine) the arrow must point to or from the camera before the face may flip it |
| `HEADING_ALPHA` | weight of the newest measurement in the smoothed heading |
| `HEADING_HOLD_S` | how long a heading survives without a measurement |

## Files

- `track.py` – the tracking loop for live Baslers or one or two videos: people per camera, merging, owner label, robot with its heading and ball, drawing, CSV.
- `enrol_owner.py` – records who the owner is; writes `owner/` (face vectors and two check images, not in git).
- `calibrate.py` – builds the floor maps of a camera pair from clicked floor points (a measured rectangle, or tile corners), on live or recorded frames.
- `common.py` – shared: paths, lens correction, people detection, floor maps and room bounds, keypoint names, constants.
- `faces.py` – face visibility from the pose keypoints, face vectors, similarity, the owner file.
- `fusion.py` – merges the cameras' tracks into people and decides the owner; combines the cameras' headings, also the robot's.
- `robot_heading.py` – which way the robot faces: the crop around its box, the heading model, the direction on the floor.
- `orientation.py` – which way each person's body faces, from the shoulders and hips (see Body orientation).
- `room_map.py` – the top-down drawing of the room, with a heading arrow per person and for the robot.
- `sources.py` – the video reader (one video paced, a second following it) and serial-selected Basler capture workers and live frame synchronization.
- `calibration/` – lens correction files, the floor maps and the frames they were made from.
- `models/yolo11s-pose.pt` – the pose model. Robot and ball weights are in `../training/runs/detector/weights/best.pt`, the robot heading model in `../training/runs/heading/weights/last.pt`.
- `tools/basler_record_video.py` – the group's recorder; it made the recordings above. It also writes `<video>_times.csv`, the wall-clock time of every saved frame, for syncing two cameras.
