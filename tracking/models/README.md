# models/

- `yolo11s-pose.pt` – the stock Ultralytics pose model (person detection with 17 keypoints), used by `track.py` and `enrol_owner.py`. Git stores this weight directly.
- Robot and ball are detected with `../../training/runs/detector/weights/best.pt` (classes `robot` and `ball`), trained on our recordings. The robot heading model uses `../../training/runs/heading/weights/last.pt`. Download both trained weights with Git LFS as described in the [Windows manual](../WINDOWS_LIVE_CAMERAS.md).
- Faces are found and encoded with InsightFace's `buffalo_l` model pack (face detector + ArcFace), which the `insightface` package downloads into `~/.insightface/models/` the first time `enrol_owner.py` or `track.py` runs. It is not in the repo.
