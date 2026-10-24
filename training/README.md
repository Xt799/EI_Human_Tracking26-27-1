# Robot- és labdafelismerő finomhangolása (YOLO26n)

Teljes képkockán tanítunk (1920 px), csak a robotot és a labdát; az embereket nem címkézzük. A futtatás mindig a projekt gyökeréből történik,
a közös környezetet egyszer kell telepíteni: `uv sync` (CUDA 12.8-as torch, RTX 50xx-hez is).

## Adatok (`../dataset/`, X-AnyLabeling `.json` + kép, osztályok: robot, ball)
- `train/` – 16 eredeti kép (`frame_17-*`) + a videókból előcímkézett és kézzel javított `auto_*` képek. `train/{ball,robot}/cutout|raw` a szintetikus generátor kivágásai.
- `val/` – validáció (a teszt videó utolsó ~18%-a, 24 kocka)
- `test/` – teszt (a legnagyobb videóból, 87 kocka). Tanításra nem használjuk.
- `backgrounds/` – háttérképek a szintetikus adathoz (üresen coco128-at tölt le)

## Folyamat
1. `uv run python training/scripts/build_full.py` – YOLO-adathalmaz a `training/ds_full/` mappába (`--with-test`: csak kivételes; minden kép epochonként egyszer szerepel)
2. `uv run python training/scripts/gen_synthetic.py --n 300 [--preview]` – szintetikus képek (kivágott robot/labda más háttéren, részben kilógva/eltakarva) hozzáfűzése a tanításhoz
3. `uv run python training/scripts/train.py --epochs 30 --cache none --workers 4 --name <név>` – eredmény: `training/runs/<név>/weights/best.pt`
4. `uv run python training/scripts/evaluate.py training/runs/<név>/weights/best.pt [--set test|val] [--imgsz 2560] [--conf 0.5]` – mAP, TP/FP/FN, duplikátumok
5. `uv run python inference/live_view.py [--weights ...] [--imgsz 2560] [--speed 1]` – élő megjelenítés (alapból a legnagyobb videó)

Új tanítóképek a videókból: `uv run python training/scripts/make_trainset.py [--n 400]` – sokféle kockát választ a NEM teszt videókból,
`yolo26n_e30` három bemeneti méretével előcímkézi, és `dataset/train/auto_*` + `review/` + `frames_to_check.txt` néven kiírja átnézésre.
Ezt X-AnyLabeling-ben kell javítani; a script nem ír felül javított címkéket (`--force` nélkül).

## Eredmény (`runs/yolo26n_e30`, YOLO26n, teszt halmaz, conf 0.4)
| bemenet | robot P / R | labda P / R | kocka >1 robottal / labdával |
|---|---|---|---|
| 1920 | 0.93 / 0.93 | 0.98 / 0.66 | 2 / 0 |
| 2560 | 0.96 / 0.86 | 0.95 / 0.82 | 2 / 1 |
| 3200 | 1.00 / 0.78 | 0.97 / 0.82 | 0 / 2 |
A kimaradt labdák nagy része a videó elején levő ~20 px-es, távoli labda; nagyobb bemenettel jobb.
Összehasonlításul YOLO11n-nel (azonos adat, 1920): robot 0.97 / 0.96, labda 1.00 / 0.71, nincs duplikált kocka – ebben a kísérletben a YOLO11n kicsit jobb volt.

## Megjegyzések
- `--cache ram` nagy képméretnél memóriahibát ad a dataloader workerekkel, ezért `--cache none --workers 4`.
