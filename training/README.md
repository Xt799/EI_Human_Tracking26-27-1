# Fine-tuning the robot and ball detector (YOLO26n)

We train on full frames (1920 px), labelling only the robot and the ball; people are not labelled. Always run commands from the project root.
Install the shared environment once: `uv sync` (CUDA 12.8 torch, also works on RTX 50xx).

## Data (`../dataset/`, X-AnyLabeling `.json` + image, classes: robot, ball)
- `train/` – 16 original images (`frame_17-*`) + `auto_*` images pre-labelled from the videos and corrected by hand. `train/{ball,robot}/cutout|raw` are the cutouts for the synthetic generator.
- `val/` – validation (last ~18% of the test video, 24 frames)
- `test/` – test (from the largest video, 87 frames). Never used for training.
- `backgrounds/` – background images for the synthetic data (downloads coco128 if empty)

## Pipeline
1. `uv run python training/scripts/build_full.py` – YOLO dataset in `training/ds_full/` (`--with-test`: exceptional use only; every image appears once per epoch)
2. `uv run python training/scripts/gen_synthetic.py --n 300 [--preview]` – adds synthetic images (robot/ball cutouts on other backgrounds, partly out of frame or occluded) to the training set
3. `uv run python training/scripts/train.py --epochs 30 --cache none --workers 4 --name <name>` – output: `training/runs/<name>/weights/best.pt`
4. `uv run python training/scripts/evaluate.py training/runs/<name>/weights/best.pt [--set test|val] [--imgsz 2560] [--conf 0.5]` – mAP, TP/FP/FN, duplicates
5. `uv run python inference/live_view.py [--weights ...] [--imgsz 2560] [--speed 1]` – live view (largest video by default)

New training images from the videos: `uv run python training/scripts/make_trainset.py [--n 400]` – picks diverse frames from the NON-test videos,
pre-labels them with `yolo26n_e30` at three input sizes, and writes them as `dataset/train/auto_*` + `review/` + `frames_to_check.txt` for review.
Correct them in X-AnyLabeling; the script does not overwrite corrected labels (unless `--force` is given).

## Results (`runs/yolo26n_e30`, YOLO26n, test set, conf 0.4)
| input | robot P / R | ball P / R | frames with >1 robot / ball |
|---|---|---|---|
| 1920 | 0.93 / 0.93 | 0.98 / 0.66 | 2 / 0 |
| 2560 | 0.96 / 0.86 | 0.95 / 0.82 | 2 / 1 |
| 3200 | 1.00 / 0.78 | 0.97 / 0.82 | 0 / 2 |
Most missed balls are the distant ~20 px ball at the start of the video; larger input sizes help.
For comparison, YOLO11n (same data, 1920): robot 0.97 / 0.96, ball 1.00 / 0.71, no duplicate frames – in this experiment YOLO11n was slightly better.

## Notes
- `--cache ram` runs out of memory with dataloader workers at large image sizes, hence `--cache none --workers 4`.
