# Robot- és labdafelismerés (YOLO26n)

Videóról robotot és labdát detektáló modell: finomhangolás (`training/`) és élő megjelenítés (`inference/`).

```
uv sync                                        # egyszer, közös .venv (CUDA 12.8-as torch)
uv run python inference/live_view.py --video <videó.mp4>   # élő megjelenítés
```
A tanítás menete, az adatok és az eredmények: [training/README.md](training/README.md).

- `dataset/` – X-AnyLabeling címkék (`.json`) és képek (Git LFS-ben): `train/`, `val/`, `test/`. A `videos/` mappa (nyers videók) nincs a tárolóban.
- `training/runs/yolo26n_e30/weights/best.pt` – a betanított modell.
- Az előtanított `yolo26n.pt` az első tanításkor magától letöltődik (`training/`).
