# Tracking: people, owner and stranger, robot, ball

Two fixed cameras in opposite corners of the lab record the room at the same time. For every frame we detect the people with a pose model and the robot and the ball with the YOLO26n model from `training/`, convert each detection's position on the floor from pixels into metres, and merge the two views into one set of people and objects. The output is a trajectory for every person and object in room coordinates, with each person labelled as the owner or a stranger.

## Running

```
uv run python tracking/calibrate.py a1 b1 --video videos/a1.mp4 videos/b1.mp4 --rect 4.0 3.0
uv run python tracking/enrol_owner.py videos/<clip with the owner's face>.mp4
uv run python tracking/track.py videos/a1.mp4 a1 videos/b1.mp4 b1
```

The recordings are not in git: copy them into `videos/` at the repo root. They come in pairs, `a<n>` and `b<n>`, filmed at the same time by the two cameras; each pair has its own camera positions.

1. `calibrate.py` makes the floor maps of a pair, once per camera position: [calibration/README.md](calibration/README.md).
2. `enrol_owner.py` records the owner's face from a short clip in which their face is visible (any single video; use a different stretch than the one you track). It shows the faces it found per id and asks which id is the owner.
3. `track.py` takes the pair and the camera position of each video. The two videos of a pair must start on the same frame and have the same frame count (the `a` recordings were re-timed to 30 fps to match their `b` partners; the untouched originals are in `videos/original/`). Three windows open: the two camera views and a top-down room map. Positions go to `results/<start time>_positions.csv` at the repo root, one row per frame per tracked thing; every run writes a new file. Every frame is processed, so playback is slower than real time; `REALTIME` at the top of the file switches to dropping frames instead.

The environment is the shared one from the repo root (`uv sync`). The face models (about 280 MB) download into `~/.insightface` on first use.

## How it works

**Metres.** A floor map per camera position, a 3×3 matrix fitted once from clicked floor points, turns any pixel on the floor into metres from a fixed origin. In any room the points are the corners of a measured rectangle on the floor; in the tiled lab they are tile corners. The two cameras of a pair are calibrated together, so they share the origin and the room bounds; the maps are in `calibration/`. A map is only valid for points on the floor, so a person is placed at the feet, or at the knees or hips when the feet are hidden. A new room or camera position needs new maps: [calibration/README.md](calibration/README.md).

**Merging.** Each camera tracks people on its own with ids that last while they stay in view. In room coordinates, a track within 0.6 m of a person known from the other camera is the same person; otherwise it is a new person. The position is the average of the cameras that see the feet. A camera that only sees someone's upper body, because they stand right in front of it, does not place them. The robot's box ends at its edge nearest the camera, so each camera's point is pushed back by the robot's radius before averaging. When one camera loses someone the other keeps them, and a track that comes back under a new id rejoins by position.

**Owner.** Nobody is identified by a fixed feature; the owner is enrolled per session, by face. Whenever a person's face is turned to one of the cameras, the face is encoded and compared with the enrolled face vectors every half second; with two opposite cameras one of them usually sees it. The best match above a threshold becomes the owner and keeps the label while tracked, even with the face hidden; everyone else is a stranger. On our recordings the owner scores about 0.6 against his own faces and other people about 0.0. A track that scores like the owner is never merged with a person that scores like a stranger, however close they stand, and a track whose score stops matching its person leaves it, so an id swapped inside one camera is corrected at the next face check. A new person is only ever created from a track placed by the feet, and not from a box within 1.2 m of a person the same camera already tracks unless its face score says otherwise: that is a second box on the same person. A knee or hip point within a second of a foot point does not move anyone, because it lands up to a metre further from the camera. The owner file is not in git.

## Files

- `track.py` – the tracking loop for a pair of videos: people per camera, merging, owner label, robot and ball, drawing, CSV.
- `enrol_owner.py` – records who the owner is; writes `owner/` (face vectors and two check images, not in git).
- `calibrate.py` – builds the floor maps of a camera pair from clicked floor points (a measured rectangle, or tile corners), on a frame picked from each video.
- `common.py` – shared: paths, lens correction, people detection, floor maps and room bounds, keypoint names, constants.
- `faces.py` – face visibility from the pose keypoints, face vectors, similarity, the owner file.
- `fusion.py` – merges the cameras' tracks into people and decides the owner.
- `room_map.py` – the top-down drawing of the room.
- `sources.py` – the video reader (one video paced, a second following it) and an unused Basler camera class kept from last year.
- `calibration/` – lens correction files, the floor maps and the frames they were made from.
- `models/yolo11s-pose.pt` – the pose model. Robot and ball weights are in `../training/runs/yolo26n_e30/weights/best.pt`.
- `tools/basler_record_video.py` – the group's original recorder, unchanged; it made the recordings above.
