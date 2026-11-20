# Fine-tuning the robot and ball detector (YOLO26n)

We train on full frames (1920 px), labelling only the robot and the ball; people are not labelled. Always run commands from the project root.
Install the shared environment once: `uv sync` (CUDA 12.8 torch, also works on RTX 50xx).

## Data (`../dataset_old/`, X-AnyLabeling `.json` + image, classes: robot, ball)
This is the first, single-camera dataset. It is git-ignored and kept locally only; to get it back, check out
`dataset/{train,val,test}` from a commit before it was moved (for example `git checkout 24b8932 -- dataset`) and move
the three folders to `dataset_old/`.
- `train/` – 16 original images (`frame_17-*`) + `auto_*` images pre-labelled from the videos and corrected by hand. `train/{ball,robot}/cutout|raw` are the cutouts for the synthetic generator.
- `val/` – validation (last ~18% of the test video, 24 frames)
- `test/` – test (from the largest video, 87 frames). Never used for training.
- `backgrounds/` – background images for the synthetic data (downloads coco128 if empty)

## Pipeline
1. `uv run python training/scripts/build_full.py` – YOLO dataset in `training/ds_full/` (`--with-test`: exceptional use only; every image appears once per epoch)
2. `uv run python training/scripts/gen_synthetic.py --n 300 [--preview]` – adds synthetic images (robot/ball cutouts on other backgrounds, partly out of frame or occluded) to the training set
3. `uv run python training/scripts/train.py --epochs 100 --cache none --workers 4 --name <name>` (YOLO26 needs at least ~100 epochs; `--optimizer`, `--lr0`, `--close-mosaic` are configurable) – output: `training/runs/<name>/weights/best.pt`
4. `uv run python training/scripts/evaluate.py training/runs/<name>/weights/best.pt [--set test|val] [--imgsz 2560] [--conf 0.5]` – mAP, TP/FP/FN, duplicates
5. `uv run python inference/live_view.py [--weights ...] [--imgsz 2560] [--speed 1]` – live view (largest video by default)

New training images from the videos: `uv run python training/scripts/make_trainset.py [--n 400]` – picks diverse frames from the NON-test videos,
pre-labels them with `yolo26n_e100` at three input sizes, and writes them as `dataset/train/auto_*` + `review/` + `frames_to_check.txt` for review.
Correct them in X-AnyLabeling; the script does not overwrite corrected labels (unless `--force` is given).

## Results (`runs/yolo26n_e100`, YOLO26n, 100 epochs, MuSGD; test set, conf 0.4)
| input | robot P / R | ball P / R | false positives (robot / ball) | frames with >1 detection |
|---|---|---|---|---|
| 1920 | 0.97 / 0.99 | 1.00 / 0.74 | 2 / 0 | 0 |
| **2560** | **0.97 / 0.99** | **0.97 / 1.00** | 2 / 2 | 0 |
| 3200 | 0.96 / 0.96 | 0.93 / 1.00 | 3 / 6 | 3 |
2560 px is the best trade-off (it also finds the distant ~20 px ball). mAP50-95 is 0.78 on test and 0.83 on val.

## YOLO26 settings
YOLO26 needs longer training: at 30 epochs (5370 iterations) `auto` picks AdamW and box accuracy is weaker (test mAP50-95 0.66).
At 100 epochs (>10000 iterations) `auto` picks YOLO26's own MuSGD optimizer, with mosaic off for the last 10 epochs (`--close-mosaic 10`) – this was the best:
| training | val mAP50-95 (2560) |
|---|---|
| 30 epochs, AdamW (auto) | 0.72 |
| 100 epochs, AdamW lr0=0.002 | 0.56 (0.81 at 1920) |
| **100 epochs, MuSGD (auto)** | **0.83** |
Compared with YOLO11n (30 epochs): at 1920, robot 0.97 / 0.96, ball 1.00 / 0.71. The training setups differ, so this is not a clean comparison.
The final model was selected on the val set; the test set was measured only once.

## Notes
- `--cache ram` runs out of memory with dataloader workers at large image sizes, hence `--cache none --workers 4`.
