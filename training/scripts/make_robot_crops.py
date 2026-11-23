"""Build the robot heading dataset: one crop per robot view that is new, pre-labelled with a front and a rear point.

    uv run python training/scripts/make_robot_crops.py [--step 10] [--heading-step 30] [--grid 4 3]

1. Every --step-th frame of every camera of every recording in dataset/recordings.yaml (lens-corrected, as tracking/
   sees it) goes through the robot model; frames outside the splits of dataset/splits.yaml are skipped.
2. Each robot sighting is put into a bin: room, camera, split, the cell of a --grid over the frame the robot is in (the
   angle the camera sees it from), and its heading in the image in --heading-step degree steps (estimated with the
   current heading model (training/runs/heading); a wrong estimate only costs some variety).
3. One crop per bin: the clearest sighting (detector and heading model most confident), so every image shows the robot
   at least a little differently from all the others, and a still or repeated view is taken once.
4. Crop: a square of CROP_SCALE x the robot box around it, native pixels: dataset/robot_heading/<split>/
   <recording>_<camera>_t<frame>_x<left>_y<top>.jpg, the name giving the frame and where the crop is in it.
5. Pre-labels: the heading model's front and rear points, to be corrected by hand. They are also recorded in
   dataset/robot_heading/prelabels.json: a crop whose points are still exactly these has not been reviewed and is not used for
   training. A crop that is already there is kept with its labels; rerunning only adds crops for bins that have none.
"""
import argparse
import json

import cv2
import numpy as np

from dataset import (CROP_SCALE, HEADING_DIR, HEADING_MODEL, crop_name, crop_square, find_video, heading_deg, load_splits,
                     parse_name, recordings, split_of)
import common

DET_CONF = 0.4


def point(label, p):
    return {"label": label, "score": None, "points": [[float(p[0]), float(p[1])]], "group_id": None, "description": "",
            "difficult": False, "shape_type": "point", "flags": {}, "attributes": {}, "kie_linking": []}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--heading-step", type=float, default=30)
    ap.add_argument("--grid", type=int, nargs=2, default=[4, 3], metavar=("COLS", "ROWS"))
    a = ap.parse_args()
    from ultralytics import YOLO
    detector = YOLO(common.OBJECT_MODEL_PATH)
    heading = YOLO(str(HEADING_MODEL))
    recs = recordings()
    splits = load_splits()

    best = {}                       # bin -> (score, recording, camera, frame, box, front, rear)
    taken = set()                   # bins that already have a crop on disk
    for rec_name, rec in recs.items():
        room = rec["room"]
        for cam, info in rec["cameras"].items():
            # the split timeline: a paired camera_2 frame f is camera_1 frame f + offset
            to_timeline = (lambda f, o=rec["offset_frames"]: f + o) if cam != "camera_1" and rec.get("offset_frames") is not None else (lambda f: f)
            cap = cv2.VideoCapture(find_video(info["video"]))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fix = common.lens_correction(w, h)

            def bin_of(box, ang, split):
                cx, cy = (box[0] + box[2]) / 2 / w, (box[1] + box[3]) / 2 / h
                cell = (min(int(cx * a.grid[0]), a.grid[0] - 1), min(int(cy * a.grid[1]), a.grid[1] - 1))
                return room, cam, split, cell, int(((ang + a.heading_step / 2) % 360) // a.heading_step)

            for js in HEADING_DIR.glob(f"*/{rec_name}_{cam}_t*.json"):
                _, _, frame, ox, oy = parse_name(js.stem)
                d = json.loads(js.read_text(encoding="utf-8"))
                pts = {s["label"]: np.array(s["points"][0]) for s in d["shapes"]}
                if "front" in pts and "rear" in pts:
                    side = d["imageWidth"] / CROP_SCALE
                    c = (ox + d["imageWidth"] / 2, oy + d["imageHeight"] / 2)
                    box = (c[0] - side / 2, c[1] - side / 2, c[0] + side / 2, c[1] + side / 2)
                    taken.add(bin_of(box, heading_deg(pts["front"], pts["rear"]), split_of(splits, rec_name, cam, to_timeline(frame))))
            seen = 0
            for f in range(0, n, a.step):
                split = split_of(splits, rec_name, cam, to_timeline(f))
                if split is None:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, img = cap.read()
                if not ok:
                    break
                img = fix(img)
                r = detector.predict(img, imgsz=1920, conf=DET_CONF, verbose=False)[0]
                boxes = [(c, b) for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy())
                         if detector.names[int(k)] == "robot"]
                if not boxes:
                    continue
                det_conf, box = max(boxes, key=lambda x: x[0])
                crop, (ox, oy, fc) = crop_square(img, box, CROP_SCALE)
                p = heading.predict(crop, imgsz=320, conf=0.05, verbose=False)[0]
                if len(p.boxes) == 0:
                    continue
                k = int(p.boxes.conf.argmax())
                fr = p.keypoints.xy[k].cpu().numpy() / fc + (ox, oy)
                key = bin_of(box, heading_deg(*fr), split)
                if key in taken:
                    continue
                score = float(det_conf) * float(p.keypoints.conf[k].mean())
                if key not in best or score > best[key][0]:
                    best[key] = (score, rec_name, cam, f, box, fr[0], fr[1])
                seen += 1
            print(f"{rec_name} {cam}: {seen} robot sightings, {sum(1 for b in best.values() if b[1:3] == (rec_name, cam))} views", flush=True)

    counts = {}
    pre_path = HEADING_DIR / "prelabels.json"
    prelabels = json.loads(pre_path.read_text(encoding="utf-8")) if pre_path.exists() else {}
    for key, (_, rec_name, cam, f, box, front, rear) in sorted(best.items(), key=lambda x: (x[1][1], x[1][2], x[1][3])):
        cap = cv2.VideoCapture(find_video(recs[rec_name]["cameras"][cam]["video"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        img = common.lens_correction(img.shape[1], img.shape[0])(img)
        crop, (ox, oy, _) = crop_square(img, box, CROP_SCALE)
        out = HEADING_DIR / key[2]
        out.mkdir(parents=True, exist_ok=True)
        name = crop_name(rec_name, cam, f, ox, oy)
        cv2.imwrite(str(out / f"{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        d = {"version": "4.0.6", "flags": {}, "checked": False, "imagePath": f"{name}.jpg", "imageData": None,
             "imageHeight": crop.shape[0], "imageWidth": crop.shape[1], "description": "",
             "shapes": [point("front", front - (ox, oy)), point("rear", rear - (ox, oy))]}
        (out / f"{name}.json").write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
        prelabels[name] = {s["label"]: [round(v, 2) for v in s["points"][0]] for s in d["shapes"]}
        counts[key[2]] = counts.get(key[2], 0) + 1
    pre_path.write_text(json.dumps(prelabels, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(best)} new crops ({', '.join(f'{k} {v}' for k, v in sorted(counts.items()))}); {len(taken)} views were already there")


if __name__ == "__main__":
    main()
