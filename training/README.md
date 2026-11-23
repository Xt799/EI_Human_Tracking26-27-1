# Training the robot, ball and heading models

Two models, both fine-tuned Ultralytics YOLO26n:
- **detector** (`runs/detector/weights/best.pt`): boxes around the robot and the ball in a full frame (classes
  `0 robot`, `1 ball`). Used by `tracking/` and `inference/`, and it gives the heading model its crop.
- **heading** (`runs/heading/weights/last.pt`): a keypoint model on a crop around the robot (2x the robot box) that
  finds a `front` and a `rear` point; rear -> front is the way the robot faces.

The labelled images are in `../dataset/` (see [../dataset/README.md](../dataset/README.md), also for how to add and
label new images). Always run commands from the project root. Install the shared environment once: `uv sync`
(CUDA 12.8 torch, also works on RTX 50xx).

## Detector
```
uv run python training/scripts/download_balls.py      # once: public ball images (COCO, Open Images), git-ignored
uv run python training/scripts/build_detection.py     # -> training/ds_detect/
uv run python training/scripts/train.py --model <start.pt> --data training/ds_detect/data.yaml --imgsz 1920 --batch 6 --epochs 10 --cache none --name <name>
uv run python training/scripts/compare_detectors.py training/runs/<name>/weights/best.pt --imgsz 1920
```
- Training set: our frames (`dataset/detection/train/recordings/`) and the false-detection crops of
  `dataset/robot_heading/` (no object in them), each listed 5 times (`--own-repeat`), plus 1000 images per public
  source (`--public`), half of them those with the smallest balls. The public images have no robot in them, so they
  also teach the detector that people, clothes and furniture are not the robot.
- `compare_detectors.py`: precision / recall / mAP per class on the val set, and on how many of the false-detection
  crops a robot is found.
- `add_ball_prelabels.py WEIGHTS --imgsz 1920`: adds the balls a model finds to our frames as pre-labels, for review.
- The committed detector was fine-tuned in steps (the first single-camera model -> + more frames and the public ball
  images -> a few rounds on room_2 frames with a ball -> the command above, 10 epochs from the previous weights).
  Settings in `runs/detector/args.yaml`.
- Inference size: 1920 px or more; the ball is often only ~20 px. A detector alone flickers on such small balls
  (found in one frame, missed in the next); a tracker that keeps a ball for a few frames smooths this out.

Val set, 1920 px, conf 0.001 for mAP: robot precision 0.99, recall 1.00, mAP50 0.99; ball precision 0.96, recall 0.97,
mAP50 0.98; robot found on 0 of 20 false-detection crops.

## Heading
```
uv run python training/scripts/build_heading.py               # train = train/, val = val/  -> training/ds_heading/
uv run python training/scripts/train.py --model yolo26n-pose.pt --data training/ds_heading/data.yaml --imgsz 320 --batch 16 --epochs 120 --degrees 180 --mosaic 0 --name <name>
uv run python training/scripts/evaluate_heading.py training/runs/<name>/weights/best.pt training/ds_heading/val.txt
uv run python training/scripts/build_heading.py --all         # final model: train on train/ + val/ -> training/ds_heading_all/
```
- Only reviewed crops are used (points moved from the pre-label); crops with no points are negatives.
- `evaluate_heading.py`: angle error in the image per room, share within 10 / 20 degrees, flipped (more than 90
  degrees off), and false robots on the negatives. Other splits: `--train-room room_2` (train on one room, measure on
  the other) and `--folds 5` (cross-validation in time chunks).
- The committed model was trained on all crops (`--all`, 120 epochs), so it has no held-out score: use `last.pt`.
  The same setup trained on `train/` only scored on `val/`: median error room_1 5.9 deg, room_2 4.5 deg, 0% flipped.
- Videos to check it: `render_heading_video.py VIDEO [--split val] [--seconds a b]` (one camera) and
  `render_heading_paired.py room_1_20261012_1800` (both cameras and a top-down floor view); written to `../results/`.

## Notes
- `train.py` defaults: 1920 px, 100 epochs, `--optimizer auto` (AdamW below ~10000 iterations, YOLO26's MuSGD above).
  For a short fine-tune of a trained model, `--optimizer AdamW --lr0 0.0005` keeps more of what it knew.
- Very small training sets: the learning-rate warm-up takes about 100 iterations, so with a few images it never ends;
  list the images repeatedly.
- `--cache ram` runs out of memory with dataloader workers at large image sizes, hence `--cache none`.
- On Windows, Ultralytics' dataloader workers re-import the main script: run scripts from a file (not from stdin) and
  keep the work under `if __name__ == "__main__":`.
