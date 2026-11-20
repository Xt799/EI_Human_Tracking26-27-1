"""Fine-tune YOLO26n on the full 1920 px frames.  Run:  uv run python training/scripts/train.py [--epochs 100]"""
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
    ap.add_argument("--optimizer", default="auto", help="auto picks AdamW below 10000 iterations and MuSGD (the YOLO26 optimizer) above")
    ap.add_argument("--lr0", type=float, default=None, help="initial learning rate (default: the optimizer's own)")
    ap.add_argument("--close-mosaic", type=int, default=10, help="last N epochs without mosaic")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache", default="ram", choices=["ram", "disk", "none"], help="use none/disk for very large --imgsz (RAM cache copies blow up with dataloader workers)")
    a = ap.parse_args()
    device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    extra = {} if a.lr0 is None else {"lr0": a.lr0}
    YOLO(str(ROOT / "yolo26n.pt")).train(
        data=str(ROOT / "ds_full" / "data.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, workers=a.workers, device=device,
        project=str(ROOT / "runs"), name=a.name, exist_ok=True,
        mosaic=1.0, close_mosaic=a.close_mosaic, optimizer=a.optimizer, **extra, scale=0.25, degrees=8, fliplr=0.5, flipud=0.0,
        hsv_v=0.35, hsv_s=0.4, translate=0.1, cache=False if a.cache == "none" else a.cache, save_period=5, plots=True)


if __name__ == "__main__":   # required on Windows (dataloader worker processes)
    main()
