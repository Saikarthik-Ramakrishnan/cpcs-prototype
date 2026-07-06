"""
CPCS multi-camera recorder — Phase 1 of on-bus testing.

Purpose: capture synchronized footage from 2-3 USB cameras on a live bus,
with per-frame timestamps, WITHOUT running any inference. Board-agnostic:
runs on x86 or ARM, needs no NPU, no model, no acceleration. The only job
is to get clean real-world footage to validate the counting pipeline against.

Each camera writes to its own timestamped .mp4 plus a frames.csv logging
(frame_index, epoch_time) so recordings can be aligned and replayed later.

Run:
    # list what cameras the board sees
    python cpcs_recorder.py --list

    # record 2 cameras at indices 0 and 2 into ./recordings/
    python cpcs_recorder.py --cams 0 2 --route "47A" --bus "DL-1PC-4432"

    # cap resolution / fps for weaker boards or USB bandwidth limits
    python cpcs_recorder.py --cams 0 2 --width 1280 --height 720 --fps 15

Press Ctrl-C to stop. Files land in --outdir (default ./recordings).
"""

import argparse
import csv
import os
import signal
import sys
import time
from datetime import datetime

import cv2

_stop = False


def _handle_sigint(sig, frame):
    global _stop
    _stop = True


def list_cameras(max_index=8):
    print("scanning camera indices 0..%d ..." % max_index)
    found = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frm = cap.read()
            if ret and frm is not None:
                h, w = frm.shape[:2]
                found.append((i, w, h))
                print(f"  index {i}: OK  {w}x{h}")
            cap.release()
    if not found:
        print("  no cameras found. Check USB connections and permissions "
              "(try: ls /dev/video*).")
    return found


def open_cam(idx, width, height, fps):
    cap = cv2.VideoCapture(idx)
    # MJPG lets many USB cameras hit higher fps at 1080p than raw YUV
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true",
                    help="list available cameras and exit")
    ap.add_argument("--cams", type=int, nargs="+", default=[0],
                    help="camera indices to record, e.g. --cams 0 2")
    ap.add_argument("--outdir", default="recordings")
    ap.add_argument("--route", default="unknown")
    ap.add_argument("--bus", default="unknown")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--preview", action="store_true",
                    help="show a live window (only if the board has a display)")
    args = ap.parse_args()

    if args.list:
        list_cameras()
        return

    signal.signal(signal.SIGINT, _handle_sigint)
    os.makedirs(args.outdir, exist_ok=True)
    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(args.outdir, f"{session}_{args.route}_{args.bus}")
    os.makedirs(session_dir, exist_ok=True)

    # open every camera
    caps, writers, logs, log_writers = {}, {}, {}, {}
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for idx in args.cams:
        cap = open_cam(idx, args.width, args.height, args.fps)
        if not cap.isOpened():
            print(f"ERROR: could not open camera {idx}; skipping.")
            continue
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_path = os.path.join(session_dir, f"cam{idx}.mp4")
        writers[idx] = cv2.VideoWriter(vid_path, fourcc, args.fps, (aw, ah))
        log_path = os.path.join(session_dir, f"cam{idx}_frames.csv")
        logs[idx] = open(log_path, "w", newline="")
        log_writers[idx] = csv.writer(logs[idx])
        log_writers[idx].writerow(["frame_index", "epoch_time", "iso_time"])
        caps[idx] = cap
        print(f"cam {idx}: {aw}x{ah} -> {vid_path}")

    if not caps:
        print("no cameras opened, aborting.")
        return

    # write a session manifest
    with open(os.path.join(session_dir, "session.txt"), "w") as f:
        f.write(f"route={args.route}\nbus={args.bus}\n")
        f.write(f"cams={args.cams}\nrequested={args.width}x{args.height}@{args.fps}\n")
        f.write(f"started={datetime.now().isoformat(timespec='seconds')}\n")

    print("\nRECORDING. Press Ctrl-C to stop.")
    print("Tip: at each bus stop, note the time — it helps label stops later.\n")

    counts = {idx: 0 for idx in caps}
    t0 = time.time()
    last_report = t0

    while not _stop:
        now = time.time()
        for idx, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                continue
            writers[idx].write(frame)
            log_writers[idx].writerow(
                [counts[idx], f"{now:.4f}",
                 datetime.now().isoformat(timespec="milliseconds")])
            counts[idx] += 1
            if args.preview:
                cv2.imshow(f"cam {idx}", frame)

        if args.preview and (cv2.waitKey(1) & 0xFF == ord("q")):
            break

        if now - last_report >= 5.0:
            elapsed = now - t0
            rates = ", ".join(
                f"cam{idx}={counts[idx]/elapsed:.1f}fps" for idx in caps)
            print(f"  {elapsed:6.0f}s  {rates}")
            last_report = now

    # cleanup
    for idx in caps:
        caps[idx].release()
        writers[idx].release()
        logs[idx].close()
    if args.preview:
        cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print("\nstopped.")
    for idx in caps:
        print(f"  cam{idx}: {counts[idx]} frames  "
              f"({counts[idx]/max(elapsed,1):.1f} fps avg)")
    print(f"\nsession saved to: {session_dir}")
    print("next: copy this folder to your dev machine and run each cam*.mp4 "
          "through cpcs_poc.py")


if __name__ == "__main__":
    main()
