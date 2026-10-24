# EI Human Tracking

## Repository layout
- `cap_detection/` – cap detection scripts (Basler camera, YOLO pose, ROS 2 position publisher).
- `dataset/` – X-AnyLabeling labels (`.json`) and images (in Git LFS): `train/`, `val/`, `test/`. The `videos/` folder (raw videos) is not in the repository.
- `training/` – fine-tuning of the robot and ball detector (YOLO26n); see [training/README.md](training/README.md).
- `inference/` – live view of the detector on a video.

## Robot and ball detection (YOLO26n)
Model that detects the robot and the ball in video: fine-tuning (`training/`) and live view (`inference/`).

```
uv sync                                        # once, shared .venv (CUDA 12.8 torch)
uv run python inference/live_view.py --video <video.mp4>   # live view
```
Training pipeline, data and results: [training/README.md](training/README.md).

- `training/runs/yolo26n_e30/weights/best.pt` – the trained model.
- The pretrained `yolo26n.pt` is downloaded automatically on the first training run (`training/`).
