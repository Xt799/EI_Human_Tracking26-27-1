# models/

- `yolo11s-pose.pt` – the stock Ultralytics pose model (person detection with 17 keypoints), used by `people_tracking.py`. It is committed to the repo, like the detector in `training/`, so a fresh clone runs without downloads.
- Robot and ball are detected with `../../training/runs/yolo26n_e30/weights/best.pt` (classes `robot` and `ball`), trained on our recordings.
