# EI Human Tracking

## Repository layout
- `cap_detection/` – cap detection scripts (Basler camera, YOLO pose, ROS 2 position publisher).
- `dataset/` – our labelled images (robot heading crops, robot and ball boxes); see [dataset/README.md](dataset/README.md).
- `videos/` – raw recordings, not in the repository: `videos/room_<n>/camera_<n>/*.mp4`.
- `training/` – fine-tuning of the robot and ball detector and the robot heading model (YOLO26n); see [training/README.md](training/README.md).
- `inference/` – live view of the detector on a video.
- `tracking/` – multi-camera tracking of people, the robot and the ball; see [tracking/README.md](tracking/README.md).

## Robot and ball detection (YOLO26n)
Model that detects the robot and the ball in video: fine-tuning (`training/`) and live view (`inference/`).

```
uv sync                                        # once, shared .venv (CUDA 12.8 torch)
uv run python inference/live_view.py --video <video.mp4>   # live view
```
Training pipeline, data and results: [training/README.md](training/README.md).
Tracking of people (owner / stranger), the robot and the ball from two cameras, in metres: [tracking/README.md](tracking/README.md).

- `training/runs/detector/weights/best.pt` – the robot and ball detector.
- `training/runs/heading/weights/last.pt` – the robot heading model (front and rear point on a crop around the robot).
- The pretrained `yolo26n.pt` is downloaded automatically on the first training run (`training/`).
