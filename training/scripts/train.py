"""Fine-tune a YOLO26 model.  Run:  uv run python training/scripts/train.py [--epochs 100]

Default: the robot/ball detector on full 1920 px frames (training/ds_detect, from build_detection.py). --model, --data,
--degrees and --mosaic train other models, e.g. the heading model on robot crops: --model yolo26n-pose.pt
--data training/ds_heading/data.yaml --imgsz 320 --batch 16 --degrees 180 --mosaic 0 (see training/README.md).
"""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--name", default="yolo26n")
    ap.add_argument("--model", default="yolo26n.pt", help="pretrained weights (file in training/ or a name Ultralytics downloads)")
    ap.add_argument("--data", default=str(ROOT / "ds_detect" / "data.yaml"))
    ap.add_argument("--optimizer", default="auto", help="auto picks AdamW below 10000 iterations and MuSGD (the YOLO26 optimizer) above")
    ap.add_argument("--lr0", type=float, default=None, help="initial learning rate (default: the optimizer's own)")
    ap.add_argument("--degrees", type=float, default=8, help="random rotation range (180 = any angle)")
    ap.add_argument("--mosaic", type=float, default=1.0)
    ap.add_argument("--close-mosaic", type=int, default=10, help="last N epochs without mosaic")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache", default="ram", choices=["ram", "disk", "none"], help="use none/disk for very large --imgsz (RAM cache copies blow up with dataloader workers)")
    a = ap.parse_args()
    device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    extra = {} if a.lr0 is None else {"lr0": a.lr0}
    YOLO(str(ROOT / a.model) if (ROOT / a.model).exists() else a.model).train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, workers=a.workers, device=device,
        project=str(ROOT / "runs"), name=a.name, exist_ok=True,
        mosaic=a.mosaic, close_mosaic=a.close_mosaic, optimizer=a.optimizer, **extra, scale=0.25, degrees=a.degrees, fliplr=0.5, flipud=0.0,
        hsv_v=0.35, hsv_s=0.4, translate=0.1, cache=False if a.cache == "none" else a.cache, save_period=5, plots=True)


if __name__ == "__main__":   # required on Windows (dataloader worker processes)
    main()
