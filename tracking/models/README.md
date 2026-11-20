# models/

- `yolo11s-pose.pt` – the stock Ultralytics pose model (person detection with 17 keypoints), used by `track.py` and `enrol_owner.py`. It is committed to the repo, like the detector in `training/`, so a fresh clone runs without downloads.
- Robot and ball are detected with `../../training/runs/yolo26n_e100/weights/best.pt` (classes `robot` and `ball`), trained on our recordings.
- Faces are found and encoded with InsightFace's `buffalo_l` model pack (face detector + ArcFace), which the `insightface` package downloads into `~/.insightface/models/` the first time `enrol_owner.py` or `track.py` runs. It is not in the repo.
