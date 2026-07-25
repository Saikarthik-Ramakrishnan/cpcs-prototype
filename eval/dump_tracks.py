"""Run detection + tracking once over a clip and dump every tracked box.

This is the expensive half of the pipeline. Freezing its output to a CSV lets
eval/replay.py and eval/tune.py evaluate counter changes in seconds instead of
minutes, and guarantees every candidate is scored on identical detections.

Usage:
    python eval/dump_tracks.py --source test_1.mp4 --model yolov8s.pt \
                               --imgsz 960 --out eval/dumps/test_1_s960.csv
"""

import argparse
import csv
import json
import os
import sys
import time

import cv2
from ultralytics import YOLO

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--device", default=None,
                    help="cpu / mps / 0 ; default lets ultralytics choose")
    ap.add_argument("--tracker", default=os.path.join(_ROOT,
                                                      "bytetrack_cpcs.yaml"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.source)
    rows, i, w, h = [], 0, 0, 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        i += 1
        h, w = frame.shape[:2]
        kw = {"device": args.device} if args.device else {}
        res = model.track(frame, classes=[0], conf=args.conf,
                          imgsz=args.imgsz, persist=True,
                          tracker=args.tracker, verbose=False, **kw)
        b = res[0].boxes
        if b.id is not None:
            for box, tid, cf in zip(b.xyxy.tolist(), b.id.tolist(),
                                    b.conf.tolist()):
                x1, y1, x2, y2 = box
                rows.append([i, int(tid), round((x1 + x2) / 2, 1),
                             round((y1 + y2) / 2, 1), round(x2 - x1, 1),
                             round(y2 - y1, 1), round(cf, 3)])
    cap.release()

    with open(args.out, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["frame", "tid", "cx", "cy", "w", "h", "conf"])
        wtr.writerows(rows)
    meta = {"source": os.path.basename(args.source), "model": args.model,
            "imgsz": args.imgsz, "conf": args.conf, "n_frames": i,
            "width": w, "height": h}
    with open(os.path.splitext(args.out)[0] + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{args.out}: {i} frames, {len(rows)} boxes, "
          f"{len(set(r[1] for r in rows))} tracks, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
